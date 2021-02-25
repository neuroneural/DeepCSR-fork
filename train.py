from omegaconf import DictConfig, OmegaConf
import hydra, logging, os
from src.data import CSRDataset, SubsamplePoints, NormalizeMRIVoxels, InvertAffine, collate_CSRData_fn, PointsToImplicitSurface
from src.models import DeepCSRNetwork, save_checkpoint, load_checkpoint
from src.metrics import OCCBCELogits, SDFL1Loss, itersection_over_union
from src.utils import TicToc
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
from collections import defaultdict


# A logger for this file
logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name='train')
def train_app(cfg):
    
    # override configuration with a user defined config file
    if cfg.user_config is not None:
        user_config = OmegaConf.load(cfg.user_config)
        cfg = OmegaConf.merge(cfg, user_config)
    logger.info('Training DeepCSR\nConfig:\n{}'.format(OmegaConf.to_yaml(cfg)))


    # setting up dataset and data loader
    field_transforms = {
        'mri': Compose([NormalizeMRIVoxels('mean_std'), InvertAffine('mri_affine')]),
        'points': Compose([SubsamplePoints(cfg.trainer.points_per_image), PointsToImplicitSurface(cfg.dataset.implicit_rpr)]),
    } 
    train_dataset = CSRDataset(cfg.dataset.path, 'train', cfg.dataset.train_split, ['mri', 'points'], cfg.dataset.surfaces, shuffle=True, field_transform=field_transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.trainer.img_batch_size, collate_fn=collate_CSRData_fn, shuffle=False, pin_memory=True, num_workers=cfg.trainer.img_batch_size) 
    logger.info("{} subjects loaded for training".format(len(train_dataset)))

    val_dataset = CSRDataset(cfg.dataset.path, 'val', cfg.dataset.val_split, ['mri', 'points'], cfg.dataset.surfaces, shuffle=False, field_transform=field_transforms)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.trainer.img_batch_size, collate_fn=collate_CSRData_fn, shuffle=False, pin_memory=True, num_workers=cfg.trainer.img_batch_size) 
    logger.info("{} subjects loaded for validation".format(len(val_dataset)))

    # setting up model, criterion, optimizer and scheduler
    model = DeepCSRNetwork(cfg.model.hypercol, len(cfg.dataset.surfaces)).to(cfg.model.device)
    logger.info("DeepCSR model setup:\n{}".format(model))
    model_num_params = sum(p.numel() for p in model.parameters())
    logger.info('Total number of parameters: {}'.format(model_num_params))

    criterion = SDFL1Loss() if cfg.dataset.implicit_rpr == 'sdf' else OCCBCELogits()
    logger.info("Loss setup:\n{}".format(criterion))
    
    optimizer = getattr(torch.optim, cfg.optimizer.name)(model.parameters(), **cfg.optimizer.kwargs)
    logger.info("Optimizer setup:\n{}".format(optimizer))

    lr_scheduler = None
    if cfg.lr_schedule.name is not None:
        scheduler = getattr(torch.optim.lr_scheduler, cfg.lr_schedule.name)(optimizer, **cfg.lr_schedule.kwargs)
    logger.info("LR Schedule setup:\n{}".format(lr_scheduler))

    # setup tensorboard logs
    tb_logs_dir_path = os.path.join(cfg.outputs.output_dir, 'tb_logs')
    tb_logger = SummaryWriter(tb_logs_dir_path)
    logger.info("Tensorboard logs saved to {}".format(tb_logs_dir_path))

    ## train and validation loop
    timer, ite, epoch, device = TicToc(), 1, 1, cfg.model.device
    train_loss_acc, train_iou_acc = 0.0, 0.0
    best_val_loss = np.finfo(np.float32).max  

    # resume from checkpoint
    if cfg.trainer.resume_checkpoint is not None:
        logger.info("Resume training from {}".format(cfg.trainer.resume_checkpoint))
        ite, best_val_loss = load_checkpoint(cfg.trainer.resume_checkpoint, model, optimizer, lr_scheduler)
        logger.info('Resuming from {} ite with best val loss {:.4f}'.format(ite, best_val_loss))
        logger.info('Loaded Model:\n{}'.format(model))
        logger.info('Loaded optimizer:\n{}'.format(optimizer))
        logger.info('Loaded lr_scheduler:\n{}'.format(lr_scheduler))        

    timer.tic('train_step')
    while True:
        timer.tic('epoch')
        for data in train_dataloader:
            
            ### train step ###                            
            model.train()
            optimizer.zero_grad()        
            
            # read batch data            
            points = data.get('pts_loc').to(device)
            isrpr = data.get('pts_isrpr').to(device)
            mri_vox = data.get('mri_vox').to(device)
            mri_affine = data.get('mri_affine').to(device)            

            # network forward, loss and gradient computation, and back-propagation            
            pred, _ = model(mri_vox, points, mri_affine)
            train_loss = criterion(pred, isrpr)
            train_loss.backward()
            optimizer.step()
            train_loss_acc += train_loss.item()            
            
            # compute train metrics
            with torch.no_grad():
                if cfg.dataset.implicit_rpr == 'sdf':                
                    pred_bin, isrpr_bin = pred >= 0.0,  isrpr >= 0.0
                else:            
                    # obs: predicted occupancy is in logits
                    pred_bin, isrpr_bin = pred >= 0.0,  isrpr >= 0.5
                train_iou_acc += itersection_over_union(pred_bin, isrpr_bin).item()

            # log train
            if ite % cfg.trainer.train_log_interval == 0: 
                avg_train_ite_time = timer.toc('train_step') / float(cfg.trainer.train_log_interval)              
                train_loss_acc = train_loss_acc / float(cfg.trainer.train_log_interval)
                train_iou_acc = train_iou_acc / float(cfg.trainer.train_log_interval)
                logger.info("Training: Ite={}, Loss={:.4f}, IOU={:.4f}, AvgIteTime={:.2f} secs".format(ite, train_loss_acc, train_iou_acc, avg_train_ite_time))
                tb_logger.add_scalar('train/loss', train_loss_acc, ite)
                tb_logger.add_scalar('train/iou', train_iou_acc, ite)
                train_loss_acc, train_iou_acc = 0.0, 0.0
                timer.tic('train_step')
            ### train step ###

            ### eval step ###
            if ite % cfg.trainer.evaluate_interval == 0:
                with torch.no_grad():   
                    val_loss_acc, val_iou_acc, val_loss_surf, val_iou_surf = 0.0, 0.0, defaultdict(float), defaultdict(float)
                    timer.tic('eval_step')
                    logger.info("Evaluating...")
                    for data in val_dataloader:

                        # read batch data and network prediction
                        points = data.get('pts_loc').to(device)
                        isrpr = data.get('pts_isrpr').to(device)
                        mri_vox = data.get('mri_vox').to(device)
                        mri_affine = data.get('mri_affine').to(device)            
                        pred, _ = model(mri_vox, points, mri_affine)

                        # compute general metrics and surface specific metrics
                        if cfg.dataset.implicit_rpr == 'sdf':                
                            pred_bin, isrpr_bin = pred >= 0.0,  isrpr >= 0.0
                        else:            
                            # obs: predicted occupancy is in logits
                            pred_bin, isrpr_bin = pred >= 0.0,  isrpr >= 0.5                                   
                        val_loss_acc += criterion(pred, isrpr).item() * isrpr.size(0)                                                       
                        val_iou_acc += itersection_over_union(pred_bin, isrpr_bin).item() * isrpr.size(0)                        
                        for surf_idx, surf_name in enumerate(cfg.dataset.surfaces):
                            val_loss_surf[surf_name] += criterion(pred[:,:, [surf_idx]], isrpr[:,:, [surf_idx]]).item() * isrpr.size(0)
                            val_iou_surf[surf_name] += itersection_over_union(pred_bin[:,:, [surf_idx]], isrpr_bin[:,:, [surf_idx]]).item() * isrpr.size(0)

                    # average and log metrics
                    num_val_samples = float(len(val_dataset))
                    val_loss_acc = val_loss_acc / num_val_samples
                    val_iou_acc = val_iou_acc / num_val_samples
                    val_loss_surf = {key: val_loss_surf[key] / num_val_samples for key in val_loss_surf}
                    val_iou_surf = {key: val_iou_surf[key] / num_val_samples for key in val_iou_surf}                    
                    val_elapsed_time = timer.toc('eval_step')
                    tb_logger.add_scalar('val/loss', val_loss_acc, ite)
                    tb_logger.add_scalar('val/iou', val_iou_acc, ite)
                    for surf_name in cfg.dataset.surfaces:
                        tb_logger.add_scalar('val_surf/{}_loss'.format(surf_name), val_loss_surf[surf_name], ite)
                        tb_logger.add_scalar('val_surf/{}_iou'.format(surf_name), val_iou_surf[surf_name], ite)
                    logger.info("Evaluation: Ite={}, Loss={:.4f}, IOU={:.4f}, EvalTime={:.2f} secs, LossPerSurf={}, IOUPerSurf={}".format(
                        ite, val_loss_acc, val_iou_acc, val_elapsed_time, val_loss_surf, val_iou_surf))                    

                    # if found the best val loss so far ->  checkpoint best
                    if val_loss_acc <= best_val_loss:
                        best_val_loss = val_loss_acc
                        ckp_file = os.path.join(cfg.outputs.output_dir, 'best_model.pth')
                        save_checkpoint(ite, model, optimizer, lr_scheduler, best_val_loss, ckp_file)
                        logger.info("Best model found with val_loss={:.4f} !!! checkpoint to {}".format(best_val_loss, ckp_file))

                    # snapshot last batch
                    np_points, np_isrpr, np_pred = points.cpu().numpy(), isrpr.cpu().numpy(), pred.cpu().numpy() 
                    vis_folder_path = os.path.join(cfg.outputs.output_dir, 'visualize', 'vis_ite{:06d}'.format(ite))
                    os.makedirs(vis_folder_path, exist_ok=True)
                    for i in range(points.shape[0]):
                        vis_sample_file = os.path.join(vis_folder_path, 'sample_{:04d}.npz'.format(i))                        
                        np.savez_compressed(vis_sample_file, points=np_points[i], isrpr=np_isrpr[i], predictions=np_pred[i], isrpr_type=[cfg.dataset.implicit_rpr])
                    logger.info('visualization of predictions saved into {}'.format(vis_folder_path))

                    # learning rate scheduler step
                    if lr_scheduler is not None:                        
                        lr_scheduler.step()

            ### eval step ###

            ### checkpoint step ###            
            if ite % cfg.trainer.checkpoint_interval == 0:
                checkpoints_dir_path = os.path.join(cfg.outputs.output_dir, 'checkpoints')
                os.makedirs(checkpoints_dir_path, exist_ok=True)
                ckp_file = os.path.join(checkpoints_dir_path, 'model_ite{:06d}.pth'.format(ite))
                save_checkpoint(ite, model, optimizer, lr_scheduler, best_val_loss, ckp_file)
                logger.info("checkpoint saved to {}".format(ckp_file))
            ### checkpoint step ### 

            # next iteration
            ite = ite + 1        
        logger.info("Epoch {} finished ({:.2f} secs)".format(epoch, timer.toc('epoch')))
        epoch = epoch + 1


if __name__ == "__main__":
    train_app()
