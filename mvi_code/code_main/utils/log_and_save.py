import os
import sys
import time
from datetime import datetime
import shutil
import numpy as np
from visdom import Visdom
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt


def save_checkpoint_v2(Basic_Args, model, optim, lrscheduler, epoch, all_tr_losses, all_val_losses,
                        all_val_eval_metrics, amp_grad_scaler, fname=None, index=False):
    # Save checkpoint.
    if isinstance(model, nn.DataParallel):
        model = model.module

    if lrscheduler != None:
        state = {
            'net': model.state_dict(),
            'optimizer': optim.state_dict(),
            'lrscheduler': lrscheduler.state_dict(),
            'epoch': epoch,
            'plot_stuff': (all_tr_losses, all_val_losses, all_val_eval_metrics),
            'rng_state': torch.get_rng_state(), # 随机数生成状态
            'amp_grad_scaler': amp_grad_scaler.state_dict()
        }
    else:
        state = {
            'net': model.state_dict(),
            'optimizer': optim.state_dict(),
            'epoch': epoch,
            'plot_stuff': (all_tr_losses, all_val_losses, all_val_eval_metrics),
            'rng_state': torch.get_rng_state(),
            'amp_grad_scaler': amp_grad_scaler.state_dict()
        }

    if fname is None:
        if index:
            ckpt_name = 'ckpt_epoch' + str(epoch) + '.pth'
        else:
            ckpt_name = 'ckpt_' + '.pth'
    else:
        ckpt_name = fname

    ckpt_path = os.path.join(Basic_Args.model_save_root, ckpt_name)
    torch.save(state, ckpt_path)



def save_checkpoint_v3(Basic_Args, model, optim, lrscheduler, epoch, all_tr_losses, all_val_losses,
                        all_val_eval_metrics, amp_grad_scaler, thresholds, max_entropy, fname=None, index=False):
    # for PLOP
    if isinstance(model, nn.DataParallel):
        model = model.module

    if lrscheduler != None:
        state = {
            'net': model.state_dict(),
            'optimizer': optim.state_dict(),
            'lrscheduler': lrscheduler.state_dict(),
            'epoch': epoch,
            'plot_stuff': (all_tr_losses, all_val_losses, all_val_eval_metrics),
            'rng_state': torch.get_rng_state(), # 随机数生成状态
            'amp_grad_scaler': amp_grad_scaler.state_dict(),
            'thresholds': thresholds,
            'max_entropy': max_entropy
        }
    else:
        state = {
            'net': model.state_dict(),
            'optimizer': optim.state_dict(),
            'epoch': epoch,
            'plot_stuff': (all_tr_losses, all_val_losses, all_val_eval_metrics),
            'rng_state': torch.get_rng_state(),
            'amp_grad_scaler': amp_grad_scaler.state_dict(),
            'thresholds': thresholds,
            'max_entropy': max_entropy
        }

    if fname is None:
        if index:
            ckpt_name = 'ckpt_epoch' + str(epoch) + '.pth'
        else:
            ckpt_name = 'ckpt_' + '.pth'
    else:
        ckpt_name = fname

    ckpt_path = os.path.join(Basic_Args.model_save_root, ckpt_name)
    torch.save(state, ckpt_path)



class Logger(object):
    """Reference: https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514"""
    def __init__(self, logdir, train=True, resume=False, seed=None):
        if train and not resume:
            if len(os.listdir(logdir)) != 0:
                print(logdir)
                ans = input("Folder is not empty. All data inside folder will be deleted. "
                                "Will you proceed [y/N]? ")
                if ans in ['y', 'Y']:
                    shutil.rmtree(logdir)
                else:
                    exit(1)
        self.seed = seed
        self.resume = resume
        if train:
            self.set_dir(logdir, 'log_{}.txt'.format(seed))
        else:
            self.set_dir(logdir, 'log_test.txt')

    def set_dir(self, logdir, log_fn='log.txt'):
        self.logdir = logdir
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        if os.path.exists(os.path.join(logdir, log_fn)):
            print(os.path.join(logdir, log_fn))
            ans = input("log file exsists. log file will be deleted. "
                            "Will you proceed [y/N]? ")
            if ans in ['y', 'Y']:
                os.remove(os.path.join(logdir, log_fn))
            else:
                exit(1)
        self.log_file = open(os.path.join(logdir, log_fn), 'a')

    def log(self, string):
        self.log_file.write('[%s] %s' % (datetime.now(), string) + '\n')
        self.log_file.flush()

        print('[%s] %s' % (datetime.now(), string))
        sys.stdout.flush()

    def log_dirname(self, string):
        self.log_file.write('%s (%s)' % (string, self.logdir) + '\n')
        self.log_file.flush()

        print('%s (%s)' % (string, self.logdir))
        sys.stdout.flush()


# _, term_width = os.popen('stty size', 'r').read().split()
# term_width = int(term_width)
term_width = 80
TOTAL_BAR_LENGTH = 20.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append(' Step:%s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def tensor2image_img(tensor):
    image = 255 * (tensor.cpu().float().numpy() * 0.5 + 0.5)
    if image.shape[0] == 1:
        image = np.tile(image, (3,1,1))
    return image.astype(np.uint8)


def tensor2image_seg(tensor):
    def threshold_predictions(predictions, thr=127):
        thresholded_preds = predictions[:]
        low_values_indices = thresholded_preds < thr
        thresholded_preds[low_values_indices] = 0
        low_values_indices = thresholded_preds >= thr
        thresholded_preds[low_values_indices] = 255
        return thresholded_preds
    image = 255 * tensor.cpu().float().numpy()
    if image.shape[0] == 1:
        image = np.tile(image, (3,1,1))
        image = threshold_predictions(image)

    return image.astype(np.uint8)


class Visualizer():
    def __init__(self, env):
        self.viz = Visdom(port=10104, env=env)
        self.epoch = 0
        self.losses = {}
        self.loss_windows = {}
        self.image_windows = {}

    def log(self, losses=None, images=None):
        for i, loss_name in enumerate(losses.keys()):
            self.losses[loss_name] = losses[loss_name]

        # Draw images
        if images is not None:
            for image_name, tensor in images.items():
                if image_name not in self.image_windows:
                    if 'seg' in image_name:
                        self.image_windows[image_name] = self.viz.image(tensor2image_seg(tensor.data), opts={'title':image_name})
                    else:
                        self.image_windows[image_name] = self.viz.image(tensor2image_img(tensor.data), opts={'title':image_name})
                else:
                    if 'seg' in image_name:
                        self.viz.image(tensor2image_seg(tensor.data), win=self.image_windows[image_name], opts={'title':image_name})
                    else:
                        self.viz.image(tensor2image_img(tensor.data), win=self.image_windows[image_name], opts={'title':image_name})

        # Plot losses
        if losses is not None:
            for loss_name, loss in self.losses.items():
                if loss_name not in self.loss_windows:
                    self.loss_windows[loss_name] = self.viz.line(X=np.array([self.epoch]), Y=np.array([loss]), 
                                                                    opts={'xlabel': 'epochs', 'ylabel': loss_name, 'title': loss_name})
                else:
                    self.viz.line(X=np.array([self.epoch]), Y=np.array([loss]), win=self.loss_windows[loss_name], update='append')
            self.epoch += 1


def plot_progress(epoch, all_losses, all_losses_color, all_losses_label, output_dir, ylim=[0., 1.],
                  all_metric=None, all_metric_color=None, all_metric_label=None):
        """
        Should probably by improved
        :return:
        """
        font = {'weight': 'normal',
                'size': 18}

        matplotlib.rc('font', **font)

        fig = plt.figure(figsize=(30, 24))
        ax = fig.add_subplot(111)
        ax.set_ylim(ylim)
        if epoch > 4:
            ax.set_ylim([ylim[0], all_losses[0][4]])
        if all_metric is not None:
            ax2 = ax.twinx()

        x_values = list(range(epoch + 1))
        for i in range(len(all_losses)):
            ax.plot(x_values, all_losses[i], color=all_losses_color[i], ls='-', label=all_losses_label[i])

        if all_metric is not None:
            for i in range(len(all_metric)):
                ax2.plot(x_values, all_metric[i], color=all_metric_color[i], ls='--', label=all_metric_label[i])

        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        if all_metric is not None:
            ax2.set_ylabel("evaluation metric")
        ax.legend()
        if all_metric is not None:
            ax2.legend(loc=9)

        fig.savefig(output_dir)
        plt.close()

