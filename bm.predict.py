from omegaconf import DictConfig, OmegaConf
import hydra, logging, os
import torch
import numpy as np
import trimesh
import nibabel as nib
from multiprocessing import Pool
from skimage.measure import marching_cubes
from src.data import mri_reader, NormalizeMRIVoxels, InvertAffine
from src.models import DeepCSRNetwork, load_checkpoint
from src.utils import make_3d_grid, TicToc, save_nib_image
import nvidia_smi
import time
import datetime
import socket
from csv import writer

# A logger for this file

logger = logging.getLogger(__name__)

nvidia_smi.nvmlInit()
deviceCount = nvidia_smi.nvmlDeviceGetCount()
hostname = socket.gethostname()

def write_time2csv(model_name, t_sec=None, subj=None, loading=False, memory=False, percentUsed=None, total=None, free=None, used=None):
    base_path = '/data/users2/washbee/speedrun/'
    # Consolidate filename determination
    if loading:
        filename = 'bm.loading'
    else:
        filename = 'bm.events'

    if memory:
        filename += '.memory'

    filename += '.csv'
    filename = base_path + filename

    # Define initial list
    List = [model_name, t_sec, hostname, subj]
    
    # If memory flag is true, append memory related info
    if memory:
        List = [model_name, percentUsed, total, free, used, hostname, subj]

    if not os.path.exists(filename):
        # Create the file
        with open(filename, 'w') as file:
            # Perform any initial operations on the file, if needed
            print("File created.")

    with open(filename, 'a') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(List)


def printSpaceUsage(info_flag = False):
    msgs = ""
    for i in range(deviceCount):
        nvidia_smi.nvmlInit()
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        if info_flag:
            return (100*info.free/info.total), info.total, info.free, info.used
        
        msgs += '\n'
        msgs += "Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(i, nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total, info.total, info.free, info.used)
        nvidia_smi.nvmlShutdown()

    msgs+="\nMax Memory occupied by tensors: "+ str(torch.cuda.max_memory_allocated(device=None))
    msgs+="\nMax Memory Cached: "+ str(torch.cuda.max_memory_cached(device=None))
    msgs+="\nCurrent Memory occupied by tensors: "+ str(torch.cuda.memory_allocated(device=None))
    msgs+="\nCurrent Memory cached occupied by tensors: "+str(torch.cuda.memory_cached(device=None))
    msgs+="\n"
    return str(msgs)


def printModelSize(model):
    # print(dir(model))
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    print('\n\n\n\n')
    print('model size: {:.3f}MB'.format(size_all_mb))
    print('\n\n\n\n')


def mesh_extraction(local_args):
    surf_name, isrpr_vol, isrpr_affine, cfg = local_args
    isrpr_affine = np.eye(4) if isrpr_affine is None else isrpr_affine
    local_timer = TicToc(); local_timer_dict = {}    

    # save predictions
    if cfg.outputs.save_all:
        isrpr_vol_path = os.path.join(cfg.outputs.output_dir, 'isrpr_vol_{}_{}.nii.gz'.format(cfg.inputs.mri_id, surf_name))
        save_nib_image(isrpr_vol_path, isrpr_vol)
        logger.info("predicted implicit surface volume for surface {} saved to {}".format(surf_name, isrpr_vol_path))

    # post processing
    min_w, min_h, min_d = 0, 0, 0
    if cfg.generator.isrpr_vol_post_process:  
        logger.info('Post-processing predicted implicit surface representation for surface {} ...'.format(surf_name)) 
        local_timer.tic('PostProcessingImplicictSurface')    
        from skimage.measure import label, regionprops
        from skimage.morphology import binary_dilation, cube, convex_hull_image
        isrpr_mask, isrpr_num = label((isrpr_vol >= cfg.generator.iso_surface_level).astype(np.int32), background=0, return_num=True, connectivity=2)   
        largest_label = np.argmax([np.sum(isrpr_mask == l) for l in range(1, isrpr_num+1)]) + 1 
        isrpr_mask = binary_dilation((isrpr_mask == largest_label), cube(5))        
        isrpr_vol[np.logical_and(~isrpr_mask, isrpr_vol >= cfg.generator.iso_surface_level)] = isrpr_vol[isrpr_mask].min()
        min_w, min_h, min_d, max_w, max_h, max_d = [reg_prop for reg_prop in regionprops(isrpr_mask.astype(np.int32)) if reg_prop.label == 1][0]['bbox']
        min_w, min_h, min_d = max(min_w - 10, 0), max(min_h - 10, 0), max(min_d - 10, 0)
        max_w, max_h, max_d = min(max_w + 10, isrpr_mask.shape[0]), min(max_h + 10, isrpr_mask.shape[1]), min(max_d + 10, isrpr_mask.shape[2])
        isrpr_vol = isrpr_vol[min_w:max_w, min_h:max_h, min_d:max_d]       
        local_timer_dict['PostProcessingImplicictSurface'] = local_timer.toc('PostProcessingImplicictSurface')
        logger.info("Post-processed predicted implicit surface representation for surface {} has {} voxels and was computed in {:.4f} secs".format(
            surf_name, isrpr_vol.shape, local_timer_dict['PostProcessingImplicictSurface']))

        if cfg.outputs.save_all:
            isrpr_vol_path = os.path.join(cfg.outputs.output_dir, 'isrpr_vol_postproc_{}_{}.nii.gz'.format(cfg.inputs.mri_id, surf_name))
            save_nib_image(isrpr_vol_path, isrpr_vol)
            logger.info("Post-processed predicted implicit surface volume for surface {} saved to {}".format(surf_name, isrpr_vol_path))

    # volumetric post-processing
    if cfg.generator.isrpr_vol_smooth > 0.0:
        from skimage.filters import gaussian        
        logger.info("Smoothing predicted implicit surface representation for surface {} with Gaussian kernel radius of {}...".format(
            surf_name, cfg.generator.isrpr_vol_smooth))
        local_timer.tic('ImplicictSurfaceSmooth')
        isrpr_vol = gaussian(isrpr_vol, sigma=cfg.generator.isrpr_vol_smooth, mode='nearest', multichannel=False)
        local_timer_dict["{}_ImplicictSurfaceSmooth".format(surf_name)] = local_timer.toc('ImplicictSurfaceSmooth')
        logger.info("Smoothed predicted implicit surface for {} in {:.4f} secs".format(
            surf_name, local_timer_dict["{}_ImplicictSurfaceSmooth".format(surf_name)]))

        if cfg.outputs.save_all:
            isrpr_vol_path = os.path.join(cfg.outputs.output_dir, 'isrpr_vol_smoothed_{}_{}.nii.gz'.format(cfg.inputs.mri_id, surf_name))
            save_nib_image(isrpr_vol_path, isrpr_vol)
            logger.info("Smoothed predicted implicit surface volume for surface {} saved to {}".format(surf_name, isrpr_vol_path))

    # topology fixing
    if cfg.generator.fix_topology:        
        from nighres.shape import topology_correction
        logger.info("fixing topology of surface with nighres toolbox...".format(surf_name))
        local_timer.tic('TopologyFix')
        isrpr_nib = nib.Nifti1Image(-1. * isrpr_vol, isrpr_affine)
        propagation = 'object->background' if surf_name in ['lh_pial', 'rh_pial'] else 'background->object'             
        isrpr_nib = topology_correction(isrpr_nib, shape_type='signed_distance_function', propagation=propagation)['corrected']
        isrpr_vol, isrpr_affine, isrpr_header = -1. * isrpr_nib.get_fdata(), isrpr_nib.affine, isrpr_nib.header
        local_timer_dict["{}_TopologyFix".format(surf_name)] = local_timer.toc('TopologyFix')
        logger.info("Fixed topology of surface {} using nighres in {:.4f} secs".format(surf_name, local_timer_dict["{}_TopologyFix".format(surf_name)]))
        if cfg.outputs.save_all:
            isrpr_vol_path = os.path.join(cfg.outputs.output_dir, 'isrpr_vol_smoothed_topofixed_{}_{}.nii.gz'.format(cfg.inputs.mri_id, surf_name))
            save_nib_image(isrpr_vol_path, isrpr_vol, isrpr_affine, isrpr_header)
            logger.info("Fixed topology predicted implicit surface volume for surface {} saved to {}".format(surf_name, isrpr_vol_path))
   
    # iso-surface extraction (marching cubes)
    logger.info("extracting {} iso-surface of surface {}...".format(cfg.generator.iso_surface_level, surf_name))
    local_timer.tic('IsoSurfaceExtraction')
    bbox_size = torch.from_numpy(np.array(cfg.generator.bbox_size)).float()
    isrpr_vol = np.pad(isrpr_vol, 1, 'constant', constant_values=-1e6)
    vertices, triangles, _, _ = marching_cubes(isrpr_vol, cfg.generator.iso_surface_level, gradient_direction='ascent')
    # Normalize to bounding box
    vertices = vertices - 1.0 + np.array([min_w, min_h, min_d]).reshape(1,3)    
    vertices /= np.array([cfg.generator.resolution-1, cfg.generator.resolution-1, cfg.generator.resolution-1])    
    vertices = bbox_size * (vertices - 0.5)        
    local_timer_dict["{}_IsoSurfaceExtraction".format(surf_name)] = local_timer.toc('IsoSurfaceExtraction')
    surface_path = os.path.join(cfg.outputs.output_dir, '{}_{}.stl'.format(cfg.inputs.mri_id, surf_name))
    mesh = trimesh.Trimesh(vertices, triangles, process=False) 
    mesh.export(surface_path)
    logger.info("Surface {} extracted in {:.4f} secs and saved to {}".format(
        surf_name, local_timer_dict["{}_IsoSurfaceExtraction".format(surf_name)], surface_path))

    # mesh post-processing
    return mesh, local_timer_dict



@hydra.main(config_path="configs", config_name='predict')
def predict_app(cfg):    
    a = datetime.datetime.now()

    
    # log for GPU utilization
    GPU_msgs = []

    ### Set Stage
    stage = '0 - override config'
    msgs = printSpaceUsage()
    GPU_msgs.append(stage + msgs + '\n\n\n')
    
    # override configuration with a user defined config file
    if cfg.user_config is not None:
        user_config = OmegaConf.load(cfg.user_config)
        cfg = OmegaConf.merge(cfg, user_config)
    logger.info('Predicting surfaces with DeepCSR\nConfig:\n{}'.format(OmegaConf.to_yaml(cfg)))

    # timer
    timer = TicToc(); timer_dict = {}; timer.tic('Total')

    ### Set Stage
    stage = '0 - read MRI'
    msgs = printSpaceUsage()
    GPU_msgs.append(stage + msgs + '\n\n\n')


    # read MRI
    timer.tic('ReadData')
    normalizer, inverter_affine = NormalizeMRIVoxels('mean_std'), InvertAffine('mri_affine')
    mri_header, mri_vox, mri_affine = mri_reader(cfg.inputs.mri_vol_path)
    mri_vox = torch.from_numpy(np.expand_dims(normalizer({'mri_vox': mri_vox})['mri_vox'], 0)).float().to(cfg.model.device)
    mri_affine_inv = torch.from_numpy(np.expand_dims(inverter_affine({'mri_affine': mri_affine})['mri_affine'], 0)).float().to(cfg.model.device)
    timer_dict['ReadData'] = timer.toc('ReadData')
    logger.info("MRI {} read with {} dimensions in {:.4f} secs".format(cfg.inputs.mri_vol_path, mri_vox.shape, timer_dict['ReadData']))
    #print('mri_header',mri_header,type(mri_header))
    #print('mri_vox',mri_vox,type(mri_vox),mri_vox.shape)
    #print('mri_affine',mri_affine,type(mri_affine))
    #print('mri_affine_inv',mri_affine_inv,type(mri_affine_inv))
    #exit()
    ### Set Stage
    stage = '0 - setup model'
    msgs = printSpaceUsage()
    GPU_msgs.append(stage + msgs + '\n\n\n')
    
    # setup model 
    timer.tic('ModelSetup')
    
    print('\n*********before loading model\n', printSpaceUsage())
    model = DeepCSRNetwork(cfg.model.hypercol, len(cfg.inputs.model_surfaces)).to(cfg.model
.device)
    print('\n*********after loading model\n', printSpaceUsage())
    
    model.eval()
    timer_dict['ModelSetup'] = timer.toc('ModelSetup')
    logger.info("{:.4f} secs for DeepCSR model setup:\n{}".format(timer_dict['ModelSetup'], model))        
    model_num_params = sum(p.numel() for p in model.parameters())    
    logger.info('Total number of parameters: {}'.format(model_num_params))
     
    ### Set Stage
    stage = '0 - load model weights'
    msgs = printSpaceUsage()
    GPU_msgs.append(stage + msgs + '\n\n\n')
    
    # load model weights    
    timer.tic('ModelLoadWeights')
    best_ite, best_val_loss = load_checkpoint(cfg.inputs.model_checkpoint, model=model)
    timer_dict['ModelLoadWeights'] = timer.toc('ModelLoadWeights')
    logger.info("Model weights at iteration {} and validation loss {:.4f} loaded from {} in {:.4f} secs".format(
        best_ite, best_val_loss, cfg.inputs.model_checkpoint, timer_dict['ModelLoadWeights']))

    printModelSize(model)
    ### Set Stage
    stage = '0 - generate grid of points at desired resolution'
    msgs = printSpaceUsage()
    GPU_msgs.append(stage + msgs + '\n\n\n')
     
    # generate grid of points at desired resolution
    timer.tic('ImplicitSurfacePrediction')    
    logger.info("predicting implicit surfaces ...")
    bbox_size = torch.from_numpy(np.array(cfg.generator.bbox_size)).float()
    query_points = bbox_size * make_3d_grid((-0.5,)*3, (0.5,)*3, (cfg.generator.resolution,)*3)
    logger.info("{} query points generated to predict implicit surfaces".format(query_points.shape[0]))
     
    ### Set Stage
    stage = '0 - implicit surface prediction in batches and resuing computed features'
    msgs = printSpaceUsage()
    GPU_msgs.append(stage + msgs + '\n\n\n')
    
    # implicit surface prediction in batches and reusing computed features   
    torch.cuda.empty_cache()
    
    b = datetime.datetime.now()
    write_time2csv('DeepCSR', t_sec = (b-a).total_seconds(), loading=True)
    percentUsed,total,free,used = printSpaceUsage(info_flag=True)
    write_time2csv('DeepCSR',percentUsed=percentUsed, total=total, free=free, used=used,loading=True,memory=True)
    with torch.no_grad():
        a = datetime.datetime.now()

        precomp_feature_maps, pred_isrpr_vol = None, []
        query_points_batches = torch.split(query_points, cfg.generator.points_batch_size)
        for b_idx, points_batch in enumerate(query_points_batches):
            points_batch = points_batch.unsqueeze(0).to(cfg.model.device)
            #print('\n*********before model() predict\n', printSpaceUsage())
            pred_isrpr, precomp_feature_maps = model(mri_vox, points_batch, mri_affine_inv, precomp_feature_maps)                
            #print('\n*********after model() predict\n', printSpaceUsage())
            pred_isrpr_vol.append(pred_isrpr.squeeze(0).cpu())
            if (b_idx + 1) % 10 == 0:
                logger.info("predicted {}/{} batches of query points in {:.4f} secs".format(b_idx, len(query_points_batches), timer.toc('ImplicitSurfacePrediction')))

        pred_isrpr_vol = torch.cat(pred_isrpr_vol, dim=0)
        pred_isrpr_vol = pred_isrpr_vol.reshape(cfg.generator.resolution, cfg.generator.resolution, cfg.generator.resolution, -1)
        pred_isrpr_vol = pred_isrpr_vol.cpu().numpy()        
        timer_dict['ImplicitSurfacePrediction'] = timer.toc('ImplicitSurfacePrediction')
        del precomp_feature_maps; torch.cuda.empty_cache();
        logger.info("Implicit surface prediction of shape {} computed in {:.4f} secs".format(pred_isrpr_vol.shape, timer_dict['ImplicitSurfacePrediction']))        
    
    ### Set Stage
    stage = '0 - after model prediction / generate meshes in parallel'
    msgs = printSpaceUsage()
    GPU_msgs.append(stage + msgs + '\n\n\n')

    # generate meshes in parallel
    logger.info("extracting meshes...")
    timer.tic('MeshExtraction')    
    with Pool(len(cfg.inputs.model_surfaces)) as p:
        args_iter = [(surf_name, pred_isrpr_vol[:,:,:, surf_idx], None, cfg) for surf_idx, surf_name in enumerate(cfg.inputs.model_surfaces)]        
        out_iter = p.map(mesh_extraction, args_iter)
    timer_dict['MeshExtraction'] = timer.toc('MeshExtraction')
    logger.info("Surfaces extracted in {:.4f} secs".format(timer_dict['MeshExtraction']))

    stage = '0 - timer statistics to disk'
    msgs = printSpaceUsage()
    GPU_msgs.append(stage + msgs + '\n\n\n')
    
    # Print GPU messages
    for msg in GPU_msgs:
        print(msg)

# # FOR DEBUG
    # dummy_args = (cfg.inputs.model_surfaces[0], pred_isrpr_vol[:,:,:, 0], None, cfg)
    # mesh_extraction(dummy_args)
    # dummy_args.stop()

     # timer statistics to disk
    timer_dict['Total'] = timer.toc('Total')
    for _, local_timer in out_iter: timer_dict.update(local_timer)
    logger.info("Timer summary:")
    with open(os.path.join(cfg.outputs.output_dir, "{}_timer.txt".format(cfg.inputs.mri_id)), 'w') as file:
        for key, value in timer_dict.items():
            file.write('{},{}\n'.format(key, value))
            logger.info('\t{} => {:.4f} secs'.format(key, value))
    
    
    b = datetime.datetime.now()
    write_time2csv('DeepCSR', t_sec = (b-a).total_seconds())
    percentUsed,total,free,used = printSpaceUsage(info_flag=True)
    write_time2csv('DeepCSR',memory=True, percentUsed=percentUsed,total=total,free=free,used=used)
    
    logger.info("Total Surface prediction finished in {:.4f} seconds".format(timer_dict['Total']))


if __name__ == "__main__":
    predict_app()
