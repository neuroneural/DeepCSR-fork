from omegaconf import DictConfig, OmegaConf
import hydra, logging, os, itertools, glob, warnings
warnings.filterwarnings("ignore")
import torch
import numpy as np
import trimesh
import nibabel as nib
from src.data import mesh_reader, pointcloud_reader
from src.utils import TicToc
from joblib import Parallel, delayed


# A logger for this file
logger = logging.getLogger(__name__)

def eval_job_map(mri_id, surf_name, cfg):

    timer = TicToc(); timer.tic('total')
    logger.info('>>' * 5 + " Evaluating mri {} and surface {}".format(mri_id, surf_name))

    try:
        # load ground truth
        gt_pcl, gt_pcl_path, gt_mesh_path = None, None, None
        if cfg.evaluator.use_precomp_pcl:
            # load precomputed point cloud for ground-truth
            gt_pcl_path = os.path.join(cfg.inputs.gt_folder, mri_id, 'pointcloud.val.npz')
            gt_pcl = pointcloud_reader(gt_pcl_path, [surf_name])[0]
            logger.info("Precomputed Point cloud with {} dimensions loaded from {}".format(gt_pcl.shape, gt_pcl_path))
        else:
            # load ground-truth mesh
            gt_mesh_path = os.path.join(cfg.inputs.gt_folder, mri_id, '{}.stl'.format(surf_name))
            gt_mesh = mesh_reader(gt_mesh_path)
            gt_mesh = trimesh.Trimesh(vertices=gt_mesh[0], faces=gt_mesh[1])
            gt_mesh.remove_duplicate_faces(); gt_mesh.remove_unreferenced_vertices();                    
            logger.info("GT mesh loaded from {} with {} vertices and {} faces".format(
                gt_mesh_path, gt_mesh.vertices.shape, gt_mesh.faces.shape))
            # sample point cloud for ground-truth mesh
            timer.tic()
            gt_pcl = gt_mesh.sample(cfg.evaluator.num_sampled_points)
            sample_time = timer.toc()
            logger.info("Point cloud with {} dimensions sampled from ground-truth mesh in {:.4f} secs".format(gt_pcl.shape, sample_time))        

        # load predicted mesh
        pred_mesh_path = os.path.join(cfg.inputs.pred_folder, mri_id, '{}_{}.stl'.format(mri_id, surf_name))
        pred_mesh = mesh_reader(pred_mesh_path)
        pred_mesh = trimesh.Trimesh(vertices=pred_mesh[0], faces=pred_mesh[1])
        pred_mesh.remove_duplicate_faces(); pred_mesh.remove_unreferenced_vertices();
        logger.info("Predicted mesh loaded from {} with {} vertices and {} faces".format(
            pred_mesh_path, pred_mesh.vertices.shape, pred_mesh.faces.shape)) 
        # sampling point cloud in predicted mesh
        timer.tic()
        pred_pcl = pred_mesh.sample(cfg.evaluator.num_sampled_points)
        sample_time = timer.toc()
        logger.info("Point cloud with {} dimensions sampled from predicted mesh in {:.4f} secs".format(pred_pcl.shape, sample_time))

        # compute point to mesh distances and metrics
        logger.info("computing point to mesh distances...")
        timer.tic()
        _, P2G_dist, _ = trimesh.proximity.closest_point(pred_mesh, gt_pcl)
        _, G2P_dist, _ = trimesh.proximity.closest_point(gt_mesh, pred_pcl)
        dist_comp_time = timer.toc()
        logger.info("point to mesh distances computed in {:.4f} secs".format(dist_comp_time))
        #Average symmetric surface distance
        logger.info("computing metrics...")
        assd = ((P2G_dist.sum() + G2P_dist.sum()) / float(P2G_dist.size + G2P_dist.size))
        logger.info("\t > Average symmetric surface distance {:.4f}".format(assd))
        # Hausdorff distance
        hd = max(np.percentile(P2G_dist, 90), np.percentile(G2P_dist, 90))
        logger.info("\t > Hausdorff surface distance {:.4f}".format(hd)) 

        # log and metrics write csv
        cols_str = ';'.join(['MRI_ID', 'SURF_NAME', 'PRED_MESH', 'GT_MESH', 'GT_PCL', 'NUM_POINTS', 'ASSD', 'HD'])
        mets_str = ';'.join([mri_id, surf_name, pred_mesh_path, str(gt_mesh_path), str(gt_pcl_path), str(cfg.evaluator.num_sampled_points), str(assd), str(hd)])
        logger.info('REPORT_COLS;{}'.format(cols_str))
        logger.info('REPORT_VALS;{}'.format(mets_str))
        met_csv_file_path = os.path.join(cfg.outputs.output_dir, "{}_{}_{}.csv".format(cfg.outputs.metrics_csv_prefix, mri_id, surf_name))    
        with open(met_csv_file_path, 'w') as output_csv_file:                        
            output_csv_file.write(mets_str+'\n')
        logger.info('>>' * 5 + " Evaluation for {} and {} finished in {:.4f} secs".format(mri_id, surf_name, timer.toc('total')))
    except:
        pass



@hydra.main(config_path="configs", config_name='eval')
def eval_app(cfg):    

    timer = TicToc(); timer.tic('total_eval')

    # override configuration with a user defined config file
    if cfg.user_config is not None:
        user_config = OmegaConf.load(cfg.user_config)
        cfg = OmegaConf.merge(cfg, user_config)
    logger.info('Evaluating predicted surfaces with DeepCSR\nConfig:\n{}'.format(OmegaConf.to_yaml(cfg)))

    # # run evaluation in parallel using job lib
    mri_ids = [mri_id.strip() for mri_id in open(cfg.inputs.mri_ids, 'r').readlines()]
    Parallel(n_jobs=cfg.evaluator.num_jobs, verbose=3)(delayed(eval_job_map)(mri_id, surf_name, cfg) for mri_id, surf_name in itertools.product(mri_ids, cfg.inputs.surfaces))
    
    # join the results
    eval_temp_files = sorted(glob.glob(os.path.join(cfg.outputs.output_dir, "{}*.csv".format(cfg.outputs.metrics_csv_prefix))))
    eval_met_csv_file = os.path.join(cfg.outputs.output_dir, "{}_ALL.csv".format(cfg.outputs.metrics_csv_prefix))
    with open(eval_met_csv_file, 'w') as eval_all_file:
        eval_all_file.write(';'.join(['MRI_ID', 'SURF_NAME', 'PRED_MESH', 'GT_MESH', 'GT_PCL', 'NUM_POINTS', 'ASSD', 'HD']) + '\n')
        for eval_temp_file in eval_temp_files:
            with open(eval_temp_file, 'r') as temp_file:
                eval_all_file.write(temp_file.readline())
            os.remove(eval_temp_file)

    logger.info("Evaluation finished in {:.4f} secs".format(timer.toc('total_eval')))
    logger.info("Results saved in {}.".format(eval_met_csv_file))


if __name__ == "__main__":
    eval_app()