import csv
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import SimpleITK as sitk
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from utils.file_ops import *

from loss.seg_loss import DC_and_CE_loss
from utils.train_utils import load_MA, maybe_to_torch, to_cuda, run_online_evaluation, finish_online_evaluation
from utils.log_and_save import progress_bar, Visualizer, plot_progress, save_checkpoint

from loader.roi_loader3d import DataLoader3D
from models.unet_roi import ROI_UNet
import argparse

def train_epoch(net, criterion, mse_fn, num_fn, optimizer, dataloader, amp_grad_scaler, Basic_Args=None):
    net.train()
    train_loss = 0
    total = 0

    for i in range(Basic_Args.num_batches_per_epoch):
        data_dict = next(dataloader)
        data = data_dict['data']
        labels = data_dict['seg']

        data = maybe_to_torch(data)
        labels = maybe_to_torch(labels)
        
        if torch.cuda.is_available():
            data = to_cuda(data)
            labels = to_cuda(labels)

        batch_size = data.size(0)

        optimizer.zero_grad()
        with autocast():
            outputs = net(data)
            loss = criterion(outputs, labels)
            total_loss = loss

            train_loss += total_loss.item() * batch_size
            total += batch_size

        amp_grad_scaler.scale(total_loss).backward()
        amp_grad_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 12)
        amp_grad_scaler.step(optimizer)
        amp_grad_scaler.update()

        progress_bar(i, Basic_Args.num_batches_per_epoch, 'train loss:{}, dcce loss:{}'.format(round(total_loss.item(), 4), round(loss.item(), 4)))

    msg = 'Train loss: %.5f' % (train_loss / total)
    print(msg)

    return train_loss / total


def evaluate(net, criterion, num_fn, dataset, Basic_Args=None):
    is_training = net.training
    net.eval()

    total_loss = 0.0
    total = 0

    online_eval_foreground_dc = []
    online_eval_tp = []
    online_eval_fp = []
    online_eval_fn = []

    with torch.no_grad():
        for i in range(Basic_Args.num_val_batches_per_epoch):
            data_dict = next(dataset)
            data = data_dict['data']
            labels = data_dict['seg']
            
            data = maybe_to_torch(data)
            labels = maybe_to_torch(labels)
            
            if torch.cuda.is_available():
                data = to_cuda(data)
                labels = to_cuda(labels)

            batch_size = data.size(0)
            with autocast():
                outputs = net(data)
                loss = criterion(outputs, labels)

            total_loss += loss.item() * batch_size
            total += batch_size

            tmp_1, tmp_2, tmp_3, tmp_4 = run_online_evaluation(outputs, labels)
            online_eval_foreground_dc.append(tmp_1)
            online_eval_tp.append(tmp_2)
            online_eval_fp.append(tmp_3)
            online_eval_fn.append(tmp_4)

    global_dc_per_class = finish_online_evaluation(online_eval_tp, online_eval_fp, online_eval_fn)
    results = {
        'val_loss': total_loss / total,
        'val_mean_dice': np.mean(global_dc_per_class),
        'val_dice_per_class': global_dc_per_class
    }

    print('Val   loss: {:.5f}, Val   Dice: {:.5f}'.format(results['val_loss'], results['val_mean_dice']))

    if len(global_dc_per_class) != 1:
        dc_msg = 'Dice per class:  '
        for i in range(len(global_dc_per_class)):
            dc_msg += '{:.5f}, '.format(global_dc_per_class[i])
        dc_msg = dc_msg[:-2]
        print(dc_msg)

    net.train(is_training)
    
    return results


def train(Basic_Args):
    maybe_mkdir_p(Basic_Args.model_save_path)

    ###======================= logging =====================###
    print('Basic_Args:')
    print('model_save_root: {}'.format(Basic_Args.model_save_path))
    print('resume: {}, LR: {}, Weight_Decay: {}, Patch_Size: {}, BATCH_SIZE: {}'.format(
                Basic_Args.resume, Basic_Args.lr, Basic_Args.weight_decay, Basic_Args.patch_size, Basic_Args.batch_size))


    LOG_CSV = os.path.join(Basic_Args.model_save_path, 'log_{}.csv'.format(args.seed))
    LOG_CSV_HEADER = ['epoch', 'train loss', 'val loss', 'val Dice', 'val Dice per class']
    if not os.path.exists(LOG_CSV):
        with open(LOG_CSV, 'w') as f:
            csv_writer = csv.writer(f, delimiter=',')
            csv_writer.writerow(LOG_CSV_HEADER)

    amp_grad_scaler = None
    # mean average
    train_loss_MA = None
    val_eval_criterion_MA = None
    best_MA_tr_loss_for_patience = None
    best_epoch_based_on_MA_tr_loss = None
    best_val_eval_criterion_MA = None


    # record
    BEST_EPOCH = 0
    all_tr_losses = [[]]
    all_val_losses = [[]]
    all_val_eval_metrics = [[]]

    ###======================= data loader =====================###
    train_dataloader = DataLoader3D(Basic_Args, split_flag='train')
    val_dataloader = DataLoader3D(Basic_Args, split_flag='val')


    ###=========================== model maker =======================###
    model = ROI_UNet(num_classes=2, act='leakyrelu', pool_op_kernel_sizes=None, conv_kernel_sizes=None, args=Basic_Args)
    model = model.cuda()

    ###======================= loss and optimizer =====================###
    criterion = DC_and_CE_loss({'smooth': 1e-5}, {}).cuda()
    mse_fn = torch.nn.MSELoss(reduction='mean').cuda()  # for density map
    num_fn = torch.nn.L1Loss(reduction='mean').cuda() # for count num

    optimizer = torch.optim.Adam(model.parameters(), lr=Basic_Args.lr, weight_decay=Basic_Args.weight_decay)
    lrscheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=Basic_Args.lr_scheduler_patience,
                                                 verbose=True, threshold=Basic_Args.lr_scheduler_eps, threshold_mode="abs")


    ###======================= resume or continue =====================###
    START_EPOCH = 0
    # fp16 for speed up training
    if amp_grad_scaler is None and torch.cuda.is_available():
            amp_grad_scaler = GradScaler()

    if Basic_Args.resume:
        # Load student checkpoint for continue.
        print('==> Resuming from checkpoint..')
        assert Basic_Args.load_snapshot_path is not None
        print('==> Resuming from {}'.format(Basic_Args.load_snapshot_path))

        ckpt_t = torch.load(Basic_Args.load_snapshot_path)
        model.load_state_dict(ckpt_t['net'])
        optimizer.load_state_dict(ckpt_t['optimizer'])
        lrscheduler.load_state_dict(ckpt_t['lrscheduler'])
        START_EPOCH = ckpt_t['epoch'] + 1
        if 'amp_grad_scaler' in ckpt_t.keys():
            amp_grad_scaler.load_state_dict(ckpt_t['amp_grad_scaler'])
        
        all_tr_losses, all_val_losses, all_val_eval_metrics = ckpt_t['plot_stuff']

        train_loss_MA, val_eval_criterion_MA = load_MA(Basic_Args, all_tr_losses, all_val_losses)
        print('After resuming, start epoch {}, best loss on val: {:.4f}, train_loss_MA: {:.4f}'.format(
            START_EPOCH, all_val_losses[0][-1], train_loss_MA))


    ###============================ train start ======================###
    best_val_loss = 999999
    for epoch in range(START_EPOCH, Basic_Args.max_epoch):
        print('** Epoch %d: %s' % (epoch, Basic_Args.model_save_path)) # task description

        ##   train  ##
        train_loss = train_epoch(model, criterion, mse_fn, num_fn, optimizer, train_dataloader, amp_grad_scaler, Basic_Args)
        train_stats = {'train_loss': train_loss}

        ## Evaluation ##
        val_eval = evaluate(model, criterion, num_fn, val_dataloader, Basic_Args)

        ### update_MA
        if train_loss_MA is None or epoch < 10: # first 10 epoch do not use mean average for fast convergence
            train_loss_MA = train_stats['train_loss']
        else:
            train_loss_MA = Basic_Args.train_loss_MA_alpha * train_loss_MA + (1 - Basic_Args.train_loss_MA_alpha) * \
                                 train_stats['train_loss']

        if val_eval_criterion_MA is None or epoch < 10:
            val_eval_criterion_MA = val_eval['val_loss']
        else:
            val_eval_criterion_MA = Basic_Args.val_eval_criterion_alpha * val_eval_criterion_MA +\
                                    (1 - Basic_Args.val_eval_criterion_alpha) * val_eval['val_loss']

        # lrscheduler
        lrscheduler.step(train_loss_MA)

        print("LR:{}, train_loss_MA: {:.6f}, val_eval_criterion_MA(loss): {:.6f}".format(
                        optimizer.param_groups[0]['lr'], train_loss_MA, val_eval_criterion_MA))
        
        all_tr_losses[0].append(train_loss)
        all_val_losses[0].append(val_eval['val_loss'])
        all_val_eval_metrics[0].append(val_eval['val_mean_dice'])

        if val_eval['val_loss'] <= best_val_loss:
            best_val_loss = val_eval['val_loss']
            save_checkpoint(Basic_Args, model, optimizer, lrscheduler, epoch, all_tr_losses, all_val_losses,
                                all_val_eval_metrics, amp_grad_scaler, fname='model_best_wo_MA.pth')


        ## save checkpoint
        if best_val_eval_criterion_MA is None:
            best_val_eval_criterion_MA = val_eval_criterion_MA
        if val_eval_criterion_MA <= best_val_eval_criterion_MA:
            best_val_eval_criterion_MA = val_eval_criterion_MA
            BEST_EPOCH = epoch
            print('saving best model for epoch {} ...'.format(BEST_EPOCH))
            save_checkpoint(Basic_Args, model, optimizer, lrscheduler, epoch, all_tr_losses, all_val_losses,
                                all_val_eval_metrics, amp_grad_scaler, fname='model_best.pth')

        if epoch % (Basic_Args.save_every) == (Basic_Args.save_every - 1):
            print("saving latest model for epoch {} ... ".format(epoch))
            save_checkpoint(Basic_Args, model, optimizer, lrscheduler, epoch, all_tr_losses, all_val_losses,
                                all_val_eval_metrics, amp_grad_scaler, fname='model_every.pth')
        if epoch % 100 == 99:
            print("saving scheduled checkpoint file...")
            save_checkpoint(Basic_Args, model, optimizer, lrscheduler, epoch, all_tr_losses, all_val_losses,
                                all_val_eval_metrics, amp_grad_scaler, index=True)
        if epoch > Basic_Args.max_epoch - 1:
            save_checkpoint(Basic_Args, model, optimizer, lrscheduler, epoch, all_tr_losses, all_val_losses,
                                all_val_eval_metrics, amp_grad_scaler, index=True)

        ## save log_csv
        def _convert_scala(x):
            if hasattr(x, 'item'):
                x = x.item()
            return x
        log_tr = ['train_loss']
        log_val = ['val_loss', 'val_mean_dice']
        log_vector = [epoch] + [train_stats.get(k, 0) for k in log_tr] + [val_eval.get(k, 0) for k in log_val]
        per_class = val_eval['val_dice_per_class']
        log_vector += per_class
        log_vector = list(map(_convert_scala, log_vector))

        with open(LOG_CSV, 'a') as f:
            logwriter = csv.writer(f, delimiter=',')
            logwriter.writerow(log_vector)

        # manage patience
        continue_training = True
        if Basic_Args.patience is not None:
            if best_MA_tr_loss_for_patience is None:
                best_MA_tr_loss_for_patience = train_loss_MA
            if best_epoch_based_on_MA_tr_loss is None:
                best_epoch_based_on_MA_tr_loss = epoch

            # Now see if the moving average of the train loss has improved. If yes then reset patience, else
            # increase patience
            if train_loss_MA + Basic_Args.train_loss_MA_eps < best_MA_tr_loss_for_patience:
                best_MA_tr_loss_for_patience = train_loss_MA
                best_epoch_based_on_MA_tr_loss = epoch

            # if patience has reached its maximum then finish training (provided lr is low enough)
            if epoch - best_epoch_based_on_MA_tr_loss > Basic_Args.patience:
                if optimizer.param_groups[0]['lr'] > Basic_Args.lr_threshold:
                    print("My patience ended, but I believe I need more time (lr > 1e-6)")
                    best_epoch_based_on_MA_tr_loss = epoch - Basic_Args.patience // 2
                else:
                    print("My patience ended")
                    continue_training = False
            else:
                print("My patience running, the condition of endding {}/{}".format(epoch - best_epoch_based_on_MA_tr_loss, Basic_Args.patience))

        print("Best epoch: {}, best epoch on MA: {}".format(BEST_EPOCH, best_epoch_based_on_MA_tr_loss))
        print("")

        if not continue_training:
            # allows for early stopping
            break

    # print(' * %s' % LOGDIR)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--model_save_path', type=str, default=None)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--load_snapshot_path', type=str,
                        default=None, help='load snapshot for fine tuninng or testing!')
    parser.add_argument('--task_name', type=str, default='roi')
    parser.add_argument('--split_prop', type=float, default=0.8)
   
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--oversample_foreground_percent', type=float, default=0.33)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--patch_size', default=[64, 64, 64])

    parser.add_argument('--max_epoch', type=int, default=1000)
    parser.add_argument('--save_every', type=int, default=25)
    parser.add_argument('--num_batches_per_epoch', type=int, default=250)
    parser.add_argument('--num_val_batches_per_epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=3e-5)

    parser.add_argument('--lr_scheduler_patience', type=int, default=30)
    parser.add_argument('--lr_scheduler_eps', type=float, default=1e-3)
    parser.add_argument('--train_loss_MA_alpha', type=float, default=0.93)
    parser.add_argument('--train_loss_MA_eps', type=float, default=5e-4)
    parser.add_argument('--val_eval_criterion_alpha', type=float, default=0.9)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--lr_threshold', type=float, default=1e-6)

    args = parser.parse_args()

    train(args)
