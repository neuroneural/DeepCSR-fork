from omegaconf import DictConfig, OmegaConf
import hydra, logging, subprocess, os
from datetime import datetime
from src.utils import TicToc
import trimesh
import numpy as np
from mesh_contains.inside_mesh import check_mesh_contains
from scipy.spatial import cKDTree

# A logger for this file
logger = logging.getLogger(__name__)


# point cloud computation
# meshes in order: lh_pial, lh_white, rh_pial, rh_white
def export_pointcloud(meshes, surfnames, num_points, outfilepath, compress=True):    
    points_acc, normals_acc, ids_acc = [], [], []
    pointcloud_size = int(np.ceil(num_points / float(len(meshes))))
    for surfId, mesh in enumerate(meshes):
        points, face_idx = mesh.sample(pointcloud_size, return_index=True)
        normals = mesh.face_normals[face_idx]
        ids = np.ones(points.shape[0]) * surfId

        # Compress
        if compress:
            dtype = np.float16
        else:
            dtype = np.float32

        points_acc.append(points.astype(dtype))
        normals_acc.append(normals.astype(dtype))
        ids_acc.append(ids.astype(dtype))
    points = np.concatenate(points_acc, axis=0)
    normals = np.concatenate(normals_acc, axis=0)
    ids = np.concatenate(ids_acc, axis=0)

    logger.info('Writing pointcloud: %s' % outfilepath)
    np.savez_compressed(outfilepath, points=points, normals=normals, ids=ids, surfnames=surfnames)

# meshes in order: lh_pial, lh_white, rh_pial, rh_white
def export_points(meshes, surfnames, num_points, bbox_ratio, bbox_size, pertub, dist_mode, outfilepath, compress=True):    
    for name, mesh in zip(surfnames, meshes):
        if not mesh.is_watertight:
            logger.info('Warning: mesh %s is not watertight! Check for consistency.' % name)                
            
    kdtrees = [cKDTree(mesh.triangles_center) for mesh in meshes]        
    n_points_uniform = int(num_points * bbox_ratio)
    n_points_surface = num_points - n_points_uniform
    n_points_white = int(n_points_surface * 0.6)
    n_points_pial = int(n_points_surface * 0.3)
    n_points_cortex = n_points_surface - n_points_pial - n_points_white

    # sampling the bouding box
    points_uniform = np.random.rand(n_points_uniform, 3)
    points_uniform = bbox_size * (points_uniform - 0.5)

    # sampling in the surface
    n_lp, n_rp = near_split(n_points_pial, 2)
    n_lw, n_rw = near_split(n_points_white, 2)
    n_points_surface = [n_lp, n_lw, n_rp, n_lw]
    points_surface = np.concatenate([mesh.sample(n_points) for mesh, n_points in zip(meshes, n_points_surface)], axis=0)
    points_surface += pertub * np.random.randn(points_surface.shape[0], 3)

    # sampling the cortex
    lp_mesh, lw_mesh, rp_mesh, rw_mesh = meshes[0], meshes[1], meshes[2], meshes[3]
    lp_tree, lw_tree, rp_tree, rw_tree = kdtrees[0], kdtrees[1], kdtrees[2], kdtrees[3]
    n_points_cortex = near_split(n_points_cortex, 2)
    lc_points = sample_cortex(lw_mesh, lp_tree, n_points_cortex[0])
    rc_points = sample_cortex(rw_mesh, rp_tree, n_points_cortex[1])
        
    # computing occupancies and distances
    points = np.concatenate([points_uniform, points_surface, lc_points, rc_points], axis=0).astype(np.float64)    
    occupancies = np.stack([check_mesh_contains(mesh, points) for mesh in meshes], axis=-1)    

    # distance computation
    if dist_mode == 'point2center':
        distances = np.stack([tree.query(points)[0] for tree in kdtrees], axis=-1)
    elif dist_mode == 'point2plane':                        
        # 1 - find 10 closest faces        
        closest_faces = np.zeros((points.shape[0], 10, 4, 3, 3))
        for s in range(len(meshes)):            
            tries = kdtrees[s].query(points, k=10)[1]            
            tries = meshes[s].triangles[tries.ravel()].reshape((points.shape[0], 10, 3, 3))
            closest_faces[:, :, s, :, :] = tries
        points_query = np.tile(points[:, None, None, :], (1, 10, 4, 1)).reshape(-1,3)
        # 2 - compute closest point in a triangle from a given point
        distances = trimesh.triangles.closest_point(closest_faces.reshape(-1, 3, 3), points_query)        
        # 3 - compute the distance between the closest point in the triangle and the given points                
        distances = np.sqrt(np.sum((points_query - distances) ** 2, axis=1)).reshape(points.shape[0], 10, 4).min(axis=1)        
        del closest_faces; del points_query;        
    else:
        raise ValueError("Distance mode is not well set chose: {} or {}".format('point2center', 'point2plane'))

    # Compress
    if compress:
        points = points.astype(np.float16)
        occupancies = np.packbits(occupancies)
    else:
        points = points.astype(np.float32)

    # save file
    print('Writing points: %s' % outfilepath)
    np.savez_compressed(outfilepath, points=points, occupancies=occupancies, distances=distances, surfnames=surfnames, surf2idx=list(range(len(surfnames))))


def sample_cortex(white_mesh, pial_kdtree, num_samples):
    white_points = white_mesh.sample(num_samples)
    white_pial_dist, pial_points = pial_kdtree.query(white_points)
    pial_points = pial_kdtree.data[pial_points]
    mean, var = (white_points + pial_points) / 2.0, white_pial_dist / 4.0
    cortex_points = (var.reshape(-1, 1) * np.random.randn(mean.shape[0], 3)) + mean
    return cortex_points


def near_split(x, num_bins):
    quotient, remainder = divmod(x, num_bins)
    return [quotient + 1] * remainder + [quotient] * (num_bins - remainder)


@hydra.main(config_path="configs", config_name='preprop')
def preprop_app(cfg):
    logger.info('Data Preprocessing Routine\nConfig:\n{}'.format(OmegaConf.to_yaml(cfg)))

    # helpers
    timer = TicToc()
    timer.tic(tag='all')
    
    # creating output directory
    sample_output_dir = os.path.join(cfg.outputs.output_dir, cfg.inputs.sample_id)
    if not os.path.exists(sample_output_dir):
        os.makedirs(sample_output_dir)
    surfaces = ['lh_pial', 'lh_white', 'rh_pial', 'rh_white']

    logger.info('1 - Registering input images...')
    out_mri_vol_path = os.path.join(sample_output_dir, 'mri.nii.gz')
    out_affine_path = os.path.join(sample_output_dir, 'transform_affine.txt')
    subprocess.call(['mri_convert', cfg.inputs.mri_vol_path, out_mri_vol_path])
    timer.tic();
    reg_aladin_cmd = ['reg_aladin', '-ref', cfg.registration.template_path, '-flo', out_mri_vol_path, '-aff', out_affine_path]
    if cfg.registration.only_rigid:
        reg_aladin_cmd.append('-rigOnly')        
    subprocess.call(reg_aladin_cmd)
    if os.path.exists(os.path.join(sample_output_dir, 'outputResult.nii.gz')): os.remove(os.path.join(sample_output_dir, 'outputResult.nii.gz'))
    subprocess.call(['reg_resample', '-ref', cfg.registration.template_path, '-flo', out_mri_vol_path, 
        '-trans', out_affine_path, '-res', out_mri_vol_path, '-inter', '3'])
    logger.info('Image registration done in {} seconds'.format(timer.toc()))


    
    logger.info('2 - Converting and warping surfaces')
    timer.tic()
    T = np.linalg.inv(np.loadtxt(out_affine_path))
    for surf_name, surf_in_path in zip(surfaces, [cfg.inputs.lh_pial_path, cfg.inputs.lh_white_path, cfg.inputs.rh_pial_path, cfg.inputs.rh_white_path]):        
        surf_out_path = os.path.join(sample_output_dir, "{}.stl".format(surf_name))
        timer.tic(tag=surf_name)
        # convert freesurfer mesh to scanner cordinates and .stl format        
        subprocess.call(['mris_convert', '--to-scanner', surf_in_path, surf_out_path])
        # read stl mesh, apply transform and export as .off
        mesh = trimesh.load(surf_out_path)        
        mesh.remove_duplicate_faces(); mesh.remove_unreferenced_vertices();
        # read fixed to moving transformation and invert to moving to fixed        
        mesh = mesh.apply_transform(T)
        mesh.export(surf_out_path)
        logger.info('Surface {} converted and warped in {} seconds'.format(surf_name, timer.toc(tag=surf_name)))
    logger.info('All surfaces conversion and warping done in {} seconds'.format(timer.toc()))    
    

    meshes = []
    for surf_name in surfaces:
        surf_out_path = os.path.join(sample_output_dir, "{}.stl".format(surf_name))
        meshes.append(trimesh.load(surf_out_path))
    
    
    logger.info('3 - Generating point clouds')
    timer.tic()
    for split in ['train', 'val']:
        # export point cloud
        timer.tic(tag=split)
        out_pcl_path = os.path.join(sample_output_dir, 'pointcloud.{}.npz'.format(split))
        export_pointcloud(meshes, surfaces, cfg.sampling.num_points, out_pcl_path, compress=cfg.outputs.compress)        
        logger.info('Point cloud sampled for {} split in {} seconds'.format(split, timer.toc(tag=split)))    
    logger.info('Point cloud sampled in {} seconds'.format(timer.toc()))


    logger.info('4 - Generating implicit surface ground-truth')
    timer.tic()
    for split in ['train', 'val']:
        timer.tic(tag=split)
        # export points
        out_points_path = os.path.join(sample_output_dir, 'points.{}.npz'.format(split))
        export_points(meshes, surfaces, cfg.sampling.num_points, cfg.sampling.bbox_ratio, cfg.sampling.bbox_size, cfg.sampling.point_pertub_sigma, 
            cfg.sampling.distance_method, out_points_path, compress=cfg.outputs.compress)        
        logger.info('Implicit surface ground-truth computed for {} split in {} seconds'.format(split, timer.toc(tag=split)))    
    logger.info('Implicit surface ground-truth computed in {}'.format(timer.toc()))

    logger.info("Preprocessing for {} finished in {} seconds".format(cfg.inputs.sample_id, timer.toc(tag='all')))


if __name__ == "__main__":
    preprop_app()
