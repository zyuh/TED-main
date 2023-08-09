import os
import sys
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.nn as nn
from dataloaders import Cap_2d_full2021_fpn
from utils.losses2d import DiceLoss
from utils.util import *

from utils.util import _recover_from_flatten, gen_seqlabel_cartcoord_torch, vis, _smooth_flatten, _complete_flatten, post, cart2polar, binary_dice_loss, dice_loss_1d
from val_2d import cal_cap_metric, cal_1d_dice, cal_1d_dice_loader, cal_metric_binary
import segmentation_models.segmentation_models_pytorch as smp
import cv2
from skimage import measure
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score
import itertools
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,
                    default='./tednet/train_data', help='Name of Experiment')
parser.add_argument('--load_snapshot_path', type=str,
                    default=None, help='load snapshot for fine tuninng or testing!')
parser.add_argument('--exp', type=str, default='cap_try', help='model_name')
parser.add_argument('--max_iterations', type=int, default=20000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=300, help='maximum epoch number to train')
parser.add_argument('--is_filteroutliers', action='store_true', help='filter outliers in new train data')
parser.add_argument('--batch_size', type=int, default=32, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--n_classes', type=int, default=5, help='random seed')
parser.add_argument('--gpu', type=str, default='1', help='GPU to use')
parser.add_argument('--n_gpu', type=int, default=1, help='number of gpu to use')
parser.add_argument('--ring_width', type=int, default=150, help='max ring width') # default 0, but not meet 5x5 blur
parser.add_argument('--is_hcccap', action='store_true', help='jointly train hcc and cap')
parser.add_argument('--save_snapshot', action='store_false', help='save snapshot per N epochs')
parser.add_argument('--attention_type', type=str, default='none', help='att')
parser.add_argument('--is_tumor_seg', action='store_true', help='jointly train tumor seg and seq')
parser.add_argument('--list_path', type=str, default='./tednet/vein_code/code_main/lists', help='list_path')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--data_type', type=str, default='cap', help='cap, fen')
parser.add_argument('--is_test', action='store_true', help='inference')
parser.add_argument('--is_testvertex', action='store_true', help='using predicted vertex to sampling feature in test phase')
parser.add_argument('--is_no_coorconv', action='store_true', help='ablation: is_no_coorconv')
parser.add_argument('--is_no_fpn', action='store_true', help='ablation: is_no_fpn')
parser.add_argument('--is_fasttrain', action='store_true', help='fast training, w/o validation and testing')
parser.add_argument('--resume_epoch', type=int, default=0, help='resume_epoch')
parser.add_argument('--n_ray', type=int, default=90, help='number of rays')
parser.add_argument('--is_recover_rect', action='store_true', help='recover_rect')
parser.add_argument('--test_epoch', type=int, default=299, help='test_epoch')
args = parser.parse_args()

train_dataset = args.dataset

# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

batch_size = args.batch_size
max_iterations = args.max_iterations
base_lr = args.base_lr
if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


CELoss = nn.CrossEntropyLoss()

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.dataset = args.dataset.strip("\r")
    snapshot_path = "../model/" + args.exp
    snapshot_path += '_d' + args.data_type
    snapshot_path += '_lr' + str(args.base_lr) + '_bs' + str( args.batch_size) + '_epo' + str(args.max_epochs)
    snapshot_path = snapshot_path + '_ncls' + str(args.n_classes)
    snapshot_path = snapshot_path+ '_'+ args.attention_type if args.attention_type != 'none' else snapshot_path
    snapshot_path = snapshot_path + '_joint' if args.is_tumor_seg else snapshot_path
    snapshot_path = snapshot_path + '_noFPN' if args.is_no_fpn else snapshot_path
    snapshot_path = snapshot_path + '_nocoorconv' if args.is_no_coorconv else snapshot_path
    snapshot_path = snapshot_path + '_ray' + str(args.n_ray) if args.n_ray != 90 else snapshot_path

    list_path = args.list_path
    # make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    log_filename = snapshot_path + "/log.txt" if not args.is_test else snapshot_path + "/log_test.txt"
    if args.is_testvertex:
        log_filename = snapshot_path + "/log_testvertex.txt"
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    net = smp.PHCC_FPN('resnet50', in_channels=1, classes=args.n_classes, encoder_weights='imagenet', attention=None if args.attention_type=='none' else args.attention_type, is_tumor_seg=args.is_tumor_seg, is_testvertex=args.is_testvertex, is_no_coorconv=args.is_no_coorconv, is_no_fpn=args.is_no_fpn)
    net = net.cuda()
    if args.n_gpu > 1:
        net = nn.DataParallel(net)
    # print(net)
    optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    # build data loader
    Cap_2d = Cap_2d_full2021_fpn
    cfg_train = Cap_2d.Config(datapath=args.dataset, mode='train', data_type=args.data_type, batch=args.batch_size if not args.is_recover_rect else int(np.ceil(args.batch_size / 4)), n_classes=args.n_classes, is_recover_rect=args.is_recover_rect,  list_path=list_path, is_boundary=False, is_hcccap=args.is_hcccap, is_ctc=False, is_transformer=False, is_multidoc=False, doc_id=1000, is_transformer_seg=False)
    db_train = Cap_2d.Data(cfg_train)
    cfg_val = Cap_2d.Config(datapath=args.dataset, mode='test', data_type=args.data_type, batch=int(np.ceil(args.batch_size / 4)) if args.batch_size>=4 else 1, n_classes=args.n_classes, is_recover_rect=args.is_recover_rect, list_path=list_path, is_boundary=False, is_hcccap=args.is_hcccap, is_ctc=False, is_transformer=False, is_multidoc=False, doc_id=1000, is_transformer_seg=False)
    db_val = Cap_2d.Data(cfg_val)
    cfg_test = Cap_2d.Config(datapath=args.dataset, mode='test', data_type=args.data_type, batch=int(np.ceil(args.batch_size / 4)) if args.batch_size>=4 else 1, n_classes=args.n_classes, is_recover_rect=args.is_recover_rect, list_path=list_path, is_boundary=False, is_hcccap=args.is_hcccap, is_ctc=False, is_transformer=False, is_multidoc=False, doc_id=1000, is_transformer_seg=False)
    db_test = Cap_2d.Data(cfg_test)
    print("Length dataset | Training: {} Val: {} Test {}!".format(db_train.__len__(), db_val.__len__(), db_test.__len__()))
    if not args.is_recover_rect:
        trainloader = DataLoader(db_train, collate_fn=db_train.collate, batch_size=cfg_train.batch, shuffle=True, pin_memory=True, num_workers=4)
    else:
        trainloader = DataLoader(db_train, collate_fn=db_test.testcollate, batch_size=cfg_val.batch, shuffle=False, num_workers=4)

    valloader = DataLoader(db_val, collate_fn=db_val.testcollate, batch_size=cfg_val.batch, shuffle=False, num_workers=4)
    testloader = DataLoader(db_test, collate_fn=db_test.testcollate, batch_size=cfg_val.batch, shuffle=False, num_workers=4)
    dataloaders, image_datasets = {'train': trainloader, 'val': valloader, 'test': testloader}, {'train': db_train, 'val': db_val, 'test': db_test}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    logging.info("{} train iterations per epoch".format(len(trainloader)))
    best_performance, best_performance_acc, best_performance_f1mi, best_performance_f1ma = 0.0, 0.0, 0.0, 0.0
    start_epoch = 0
    iteration, iter_num_train, iter_num_val, iter_num_test, max_iterations = 0, 0, 0, 0, args.max_epochs * len(trainloader)
    vis_train_dir, vis_val_dir, vis_test_dir = os.path.join(snapshot_path, 'vis_train'), os.path.join(snapshot_path, 'vis_val'), os.path.join(snapshot_path, 'vis_test')
    if args.is_recover_rect:
        vis_test_dir += '_rect'
    vis_error_dir = os.path.join(snapshot_path, 'vis_error')
    os.makedirs(vis_train_dir, exist_ok=True)
    os.makedirs(vis_val_dir, exist_ok=True)
    os.makedirs(vis_test_dir, exist_ok=True)
    os.makedirs(vis_error_dir, exist_ok=True)

    if args.resume_epoch != 0:
        start_epoch = args.resume_epoch
        iteration = args.resume_epoch * len(trainloader)
        net.load_state_dict(torch.load(os.path.join(snapshot_path, 'epoch_'+str(args.resume_epoch)+'.pth')))
        lr_ = base_lr * (1.0 - iteration / max_iterations) ** 0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_

    if args.is_test:
        epoch_num = args.test_epoch# 299
        phase = 'test'
        if args.is_recover_rect:
            phase = 'test'
            vis_test_ori_dir = os.path.join(snapshot_path, 'vis_test_ori_multitest'+phase)
            os.makedirs(vis_test_ori_dir, exist_ok=True)
        vis_test_dir = vis_test_dir.replace('vis_test', 'vis_test_infer5_multitest')
        os.makedirs(vis_test_dir, exist_ok=True)
        # load_snapshot_path = os.path.join(snapshot_path, 'epoch_'+str(epoch_num)+'.pth')
        state_dict = torch.load(args.load_snapshot_path)
        net.load_state_dict(state_dict)
        net.eval()

        epoch_f1_macro, epoch_f1_micro, epoch_acc, epoch_pr, epoch_re = [], [], [], [], []
        epoch_pr_micro, epoch_re_micro = [], []
        epoch_dice_multi, epoch_acc_multi, epoch_pr_multi, epoch_re_multi, epoch_auc_multi = [], [], [], [], []
        epoch_metrics_tumor = []
        epoch_seq_label, epoch_seq_pred = [], []  # (k, 90)
        flag_found = False
        for i_batch, sampled_batch in tqdm(enumerate(dataloaders[phase])):
            iter_num_test += 1
            net.eval()
            inputs, label, label_hcc, cap_degree, cap_semantic, rect, name = sampled_batch
            # for na in name:
            #     if na.find('1267_capfen_slice010') != -1:
            #         flag_found = True
            # if not flag_found:
            #     continue

            inputs, label = inputs.cuda().float(), label.cuda().long()
            args.vis_dir, args.name = snapshot_path, name
            seq_label, vertexs = gen_seqlabel_cartcoord_torch(cap_degree, cap_semantic, args)  # (b, 90), (b, 90, 2)
            if args.n_ray < 90:
                ray_interval = int(90 / args.n_ray)
                vertexs_tmp = torch.zeros_like(vertexs, device=seq_label.device)
                for i in range(0, 90, ray_interval):
                    j = int(i / ray_interval)
                    vertexs_tmp[:, j, :] = vertexs[:, i, :]
                vertexs = vertexs_tmp.clone()

            seq_label = seq_label.cuda().long()  # (b, 90)
            out_seq, out_tumor = net(inputs, vertexs)  # (b, 3, n_ray)
            seq_pred = torch.argmax(torch.softmax(out_seq, dim=1), dim=1)  # (b, n_ray)
            if args.n_ray < 90:
                seq_pred = nn.Upsample(size=90, mode='nearest')(seq_pred.float().unsqueeze(1)).squeeze(1)

            # recover for visualization
            inputs_vis = nn.Upsample(scale_factor=1 / 4, mode='bilinear')(inputs).cpu()
            pred_recover = torch.zeros((inputs.size()[0], int(inputs.size()[2] / 4), int(inputs.size()[3] / 4)))
            label_recover = torch.zeros_like(pred_recover)
            degree_map = torch.zeros_like(pred_recover)
            mask_tumor = torch.argmax(torch.softmax(out_tumor, dim=1), dim=1).cpu().detach().numpy()
            flatten_seq = seq_pred.unsqueeze(1).unsqueeze(3).repeat(1,1,1, args.ring_width)
            flatten_seq = nn.Upsample(size=(360, args.ring_width), mode='nearest')(flatten_seq.float()).cpu().detach().numpy()
            flatten_seq_label = seq_label.unsqueeze(1).unsqueeze(3).repeat(1,1,1, args.ring_width)
            flatten_seq_label = nn.Upsample(size=(360, args.ring_width), mode='nearest')(flatten_seq_label.float()).cpu().detach().numpy()
            for b in range(len(out_seq)):
                degree_map_d = torch.zeros_like(cap_degree[b]).float()
                for i in range(90):
                    pred_recover[b, vertexs[b, i, 0], vertexs[b, i, 1]] = torch.argmax(torch.softmax(out_seq[b], dim=0), dim=0)[i] + 1
                    label_recover[b, vertexs[b, i, 0], vertexs[b, i, 1]] = seq_label[b][i] + 1
                    degree_map_d[cap_degree[b] == i + 1] = seq_pred[b][i].cpu().detach() + 1.0
                if np.sum(mask_tumor[b]>0) != 0:
                    M = cv2.moments((mask_tumor[b]>0).astype(np.float))
                else:
                    continue  # tumor ?
                cY, cX = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                dilation = cv2.dilate(mask_tumor[b].astype(np.uint8), kernel=np.ones((3, 3), np.uint8), iterations=1)
                erosion = cv2.erode(mask_tumor[b].astype(np.uint8), kernel=np.ones((3, 3), np.uint8), iterations=1)
                hcc_cvt_edge_b = (cv2.GaussianBlur((dilation - erosion).astype(np.float), (5, 5), 0) > 0).astype(np.float)
                # hcc_cvt_edge_b = (cv2.GaussianBlur((dilation - erosion).astype(np.float), (11,11), 0) > 0).astype(np.float)
                _, r1, theta1 = cart2polar(hcc_cvt_edge_b, M, hcc_cvt_edge_b, recover_flag=True, args=args)  # take 0.6 s
                pred_recover_tumor_b = _recover_from_flatten(flatten_seq[b][0]+1, np.zeros_like(mask_tumor[b]), M, r1, theta1)
                # vis(pred_recover[b].numpy(), 'pred_recover', name[b], vis_test_dir)
                # vis(label_recover[b].numpy(), 'label_recover', name[b], vis_test_dir)

                # vis(label_hcc[b].cpu().detach().numpy(), 'label_tumor', name[b], vis_test_dir)
                # vis(torch.argmax(torch.softmax(out_tumor[b], dim=0), dim=0).cpu().detach().numpy(), 'pred_tumor', name[b], vis_test_dir)

                epoch_seq_pred.append(seq_pred[b].cpu().detach().numpy())
                epoch_seq_label.append(seq_label[b].cpu().detach().numpy())

                # ########################### recover to original size ##############################################
                if args.is_recover_rect:
                    # plot original image and label
                    data_path = args.dataset + '/' + name[b][0:6] + '/' + name[b] + '.npz'
                    data = np.load(data_path)
                    img, lab = data['image'], data['label_cap']  # assuming all slices containing label_cap?
                    img = np.clip(img, -100, 240)
                    # image = (image - np.mean(image)) / np.std(image)
                    img = (img - img.min()) / (img.max() - img.min()) * 255.0
                    print(os.path.join(vis_test_dir, name[b] + '_oriimg.png'))
                    cv2.imwrite(os.path.join(vis_test_dir, name[b] + '_oriimg.png'), img.astype(np.uint8))
                    vis(lab, 'orilabel', name[b], vis_test_dir)
                    # print('image:',np.shape(image), 'label:',np.shape(label))

                    p0, p1, p2, p3 = rect[b][0]
                    pred_recover_tmor_ori_b = np.zeros((512, 512))
                    try:
                        pred_recover_tumor_b = cv2.resize(pred_recover_tumor_b, dsize=(p1-p0, p3-p2), interpolation=cv2.INTER_NEAREST)
                        pred_recover_tmor_ori_b[p0:p1, p2:p3] = pred_recover_tumor_b
                    except:
                        print("name {} rect {}".format(name[b], rect[b][0]))

                    vis(pred_recover_tmor_ori_b, 'pred_recover_predtumor_orisize', name[b], vis_test_ori_dir)
                else:
                    pass
                vis(degree_map_d, 'pred_recover_grid', name[b], vis_test_dir)
                vis(cap_semantic[b].cpu().detach().numpy(), 'cap_semantic', name[b], vis_test_dir)
                vis(label[b].cpu().detach().numpy(), 'label_cap', name[b], vis_test_dir)
                vis(pred_recover_tumor_b, 'pred_recover_predtumor', name[b], vis_test_dir)
                vis((mask_tumor[b] > 0).astype(np.float), 'predtumor', name[b], vis_test_dir)
                vis(flatten_seq[b][0] + 1, 'pred_seq', name[b], vis_test_dir)
                vis(flatten_seq_label[b][0] + 1, 'label_seq', name[b], vis_test_dir)
                image = inputs[b][0].cpu().detach().numpy()
                image = (image - image.min()) / (image.max() - image.min()) * 255.0
                cv2.imwrite(os.path.join(vis_test_dir, name[b] + '_img.png'), image.astype(np.uint8))

            # if args.is_recover_rect:
            #     continue
            # evaluating
            seq_label, seq_pred = seq_label.cpu().detach().numpy(), seq_pred.cpu().detach().numpy()
            for b in range(len(seq_label)):
                f1_macro = f1_score(seq_label[b], seq_pred[b], average='macro')
                f1_micro = f1_score(seq_label[b], seq_pred[b], average='micro')
                precision = precision_score(seq_label[b], seq_pred[b], average='macro')
                recall = recall_score(seq_label[b], seq_pred[b], average='macro')
                precision_micro = precision_score(seq_label[b], seq_pred[b], average='micro')
                recall_micro = recall_score(seq_label[b], seq_pred[b], average='micro')
                acc = accuracy_score(seq_label[b], seq_pred[b])
                dice_multi, acc_multi, pr_multi, re_multi, auc_multi = [], [], [], [], []

                for i in range(3):
                    dice_multi.append(cal_1d_dice(seq_label[b] == i, seq_pred[b] == i))
                    acc_multi.append(accuracy_score(seq_label[b] == i, seq_pred[b] == i))
                    pr_multi.append(precision_score(seq_label[b] == i, seq_pred[b] == i))
                    re_multi.append(recall_score(seq_label[b] == i, seq_pred[b] == i))
                    # auc_multi.append(roc_auc_score(seq_label[b] == i, seq_pred[b] == i))
                epoch_f1_macro.append(f1_macro)
                epoch_f1_micro.append(f1_micro)
                epoch_pr.append(precision)
                epoch_re.append(recall)
                epoch_pr_micro.append(precision_micro)
                epoch_re_micro.append(recall_micro)
                epoch_acc.append(acc)
                epoch_dice_multi.append(dice_multi)
                epoch_acc_multi.append(acc_multi)
                epoch_pr_multi.append(pr_multi)
                epoch_re_multi.append(re_multi)
                # epoch_auc_multi.append(auc_multi)
                epoch_metrics_tumor.append(cal_metric_binary(label_hcc[b].cpu().detach().numpy(),
                                                             torch.argmax(torch.softmax(out_tumor[b], dim=0),
                                                                          dim=0).cpu().detach().numpy()))  # (b, 3), dsc, hd, asd

        logging.info(' Testing | avg acc : {} dice : {} f1_mi : {} f1_ma : {} pr_mi {} pr_ma {} re_mi {} re_ma {}'.format(np.mean(epoch_acc), np.mean(epoch_dice_multi), np.mean(epoch_f1_micro),
                np.mean(epoch_f1_macro), np.mean(epoch_pr_micro), np.mean(epoch_pr), np.mean(epoch_re_micro), np.mean(epoch_re)))
        logging.info("Testing tumor| dsc: {} asd {}".format(np.mean(np.array(epoch_metrics_tumor)[:, 0]), np.mean(np.array(epoch_metrics_tumor)[:, 2])))
        epoch_auc_multi, epoch_acc_multi, epoch_pr_multi, epoch_re_multi = np.array(epoch_auc_multi), np.array(epoch_acc_multi), np.array(epoch_pr_multi), np.array(epoch_re_multi)
        print("epoch_seq_label", np.shape(epoch_seq_label), np.max(np.array(epoch_seq_label)))  # (16, 90)
        print("epoch_seq_pred", np.shape(epoch_seq_pred), np.max(np.array(epoch_seq_pred)))  # (16, 360, 150)
        epoch_seq_label, epoch_seq_pred = np.array(epoch_seq_label), np.array(epoch_seq_pred)
        epoch_seq_label, epoch_seq_pred = epoch_seq_label.flatten(), epoch_seq_pred.flatten()
        epoch_auc = []
        for i in range(3):
            epoch_auc_tmp = roc_auc_score(epoch_seq_label == i, epoch_seq_pred == i)
            epoch_auc.append(epoch_auc_tmp)
            logging.info('Testing class {} | auc {} acc {} pr {} re {}'.format(i, epoch_auc_tmp, np.mean(epoch_acc_multi[:, i]), np.mean(epoch_pr_multi[:, i]), np.mean(epoch_re_multi[:, i])))
        logging.info('average auc {}'.format(np.mean(epoch_auc)))

    else:
        iterator = tqdm(range(start_epoch, args.max_epochs), ncols=70)
        writer = SummaryWriter(snapshot_path + '/log')
        print("Starting joint sequential and tumor segmentatiion trainer")
        phase_list = ['train', 'val'] if not args.is_fasttrain else ['train']
        for epoch_num in iterator:
            # add test set to recover full cap, test_interval=20 (select best val model)
            for phase in phase_list:
                if phase == 'train':
                    loss_train =[]
                    for i_batch, sampled_batch in tqdm(enumerate(dataloaders[phase])):
                        net.train(True)
                        iteration += 1
                        inputs, label, label_hcc, cap_degree, cap_semantic, _, name = sampled_batch
                        inputs, label, label_hcc = inputs.cuda().float(), label.cuda().long(), label_hcc.cuda().long()
                        args.vis_dir, args.name = snapshot_path, name
                        seq_label, vertexs = gen_seqlabel_cartcoord_torch(cap_degree, cap_semantic, args)  # (b, 90), (b, 90, 2)
                        if args.n_ray < 90:
                            ray_interval = int(90 / args.n_ray)
                            seq_label_tmp = torch.zeros_like(seq_label, device=seq_label.device)
                            vertexs_tmp = torch.zeros_like(vertexs, device=seq_label.device)
                            for i in range(0, 90, ray_interval):
                                j = int(i / ray_interval)
                                seq_label_tmp[:, j] = seq_label[:, i]
                                vertexs_tmp[:, j, :] = vertexs[:, i, :]
                            seq_label_90 = seq_label.clone()
                            seq_label = seq_label_tmp.clone()
                            vertexs = vertexs_tmp.clone()

                        seq_label = seq_label.cuda().long()
                        if args.is_tumor_seg:
                            # print(inputs.shape, vertexs.shape)
                            out_seq, out_tumor = net(inputs, vertexs)  # (b, 3, 90)
                            dice_loss_2d = DiceLoss(n_classes=2)
                            loss_tumor = 0.5 * nn.CrossEntropyLoss()(out_tumor, label_hcc) + 0.5 * dice_loss_2d(out_tumor, label_hcc.unsqueeze(1))
                            loss_seq_ce, loss_seq_dice = CELoss(out_seq, seq_label), dice_loss_1d(out_seq, seq_label)
                            loss_seq = 0.5 * loss_seq_ce + 0.5 * loss_seq_dice
                            loss = 0.7 * loss_seq + 0.3 * loss_tumor
                        else:
                            out_seq = net(inputs, vertexs)  # (b, 3, 90)
                            loss_seq_ce, loss_seq_dice = CELoss(out_seq, seq_label), dice_loss_1d(out_seq, seq_label)
                            loss = 0.5 * loss_seq_ce + 0.5 * loss_seq_dice
                        optimizer.zero_grad()
                        loss.backward()

                        optimizer.step()
                        lr_ = base_lr * (1.0 - iteration / max_iterations) ** 0.9
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_

                        writer.add_scalar('lr', lr_, iteration)
                        writer.add_scalar('train/loss', loss, iteration)
                        writer.add_scalar('train/loss_seq_ce', loss_seq_ce, iteration)
                        writer.add_scalar('train/loss_seq_dice', loss_seq_dice, iteration)
                        if args.is_tumor_seg:
                            writer.add_scalar('train/loss_tumor', loss_tumor, iteration)
                        # logging.info('iteration %d | epoch %d: loss : %f' % (iteration, epoch_num, loss.item()))
                        loss_train.append(loss.item())

                        if not args.is_fasttrain:
                            pred_recover = torch.zeros((inputs.size()[0], int(inputs.size()[2]/4), int(inputs.size()[3]/4)))
                            label_recover = torch.zeros_like(pred_recover)
                            if epoch_num % 10 == 0 and epoch_num > 0:
                                for b in range(len(out_seq)):
                                    for i in range(90):
                                        pred_recover[b, vertexs[b, i, 0], vertexs[b, i, 1]] = torch.argmax(torch.softmax(out_seq[b], dim=0), dim=0)[i] + 1
                                        label_recover[b, vertexs[b, i, 0], vertexs[b, i, 1]] = seq_label[b][i] + 1
                                    vis(pred_recover[b].numpy(), 'pred_recover', name[b], vis_train_dir)
                                    vis(label_recover[b].numpy(), 'label_recover', name[b], vis_train_dir)
                                    vis(label[b].cpu().detach().numpy(), 'label_cap', name[b], vis_train_dir)
                                    vis(label_hcc[b].cpu().detach().numpy(), 'label_tumor', name[b], vis_train_dir)
                                    vis(torch.argmax(torch.softmax(out_tumor[b], dim=0),dim=0).cpu().detach().numpy(), 'pred_tumor', name[b], vis_train_dir)

                            if np.mod(iteration + 1, 5) == 0:
                                image = inputs[0, 0:1, :, :]
                                image = (image - torch.min(image)) / (torch.max(image) - torch.min(image))
                                image = image.repeat(3, 1, 1)  # channel 0
                                writer.add_image('train/Image', image, iteration)
                                if epoch_num % 10 == 0 and epoch_num > 0:
                                    writer.add_image('train/seq_pred_recover', pred_recover[0].unsqueeze(0) * 80, iteration)
                                    writer.add_image('train/seq_label_recover', label_recover[0].unsqueeze(0) * 80, iteration)
                                out_seq = torch.argmax(torch.softmax(out_seq, dim=1), dim=1, keepdim=True)
                                out_seq = out_seq.unsqueeze(3).repeat((1,1,1,40)) + 1
                                writer.add_image('train/seq_pred', out_seq[0, ...] * 80, iteration)
                                seq_label = seq_label.unsqueeze(1).unsqueeze(3).repeat((1,1,1,40)) + 1
                                writer.add_image('train/seq_label', seq_label[0, ...] * 80, iteration)
                                out_tumor_vis = torch.argmax(torch.softmax(out_tumor, dim=1), dim=1, keepdim=True)
                                writer.add_image('train/tumor_pred', out_tumor_vis[0, ...] * 255, iteration)
                                writer.add_image('train/tumor_label', label_hcc[0, ...].unsqueeze(0) * 255, iteration)

                            # to do: visualize recover prediction/label on top of image

                    logging.info("epoch: %d | average train loss: %f" % (epoch_num, np.mean(loss_train)))
                    if args.save_snapshot:
                        if np.mod(epoch_num + 1, 50) == 0:
                            save_mode_path = os.path.join(
                                snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
                            torch.save(net.state_dict(), save_mode_path)
                            logging.info("save model to {}".format(save_mode_path))
                if phase == 'val':
                    epoch_f1_macro, epoch_f1_micro, epoch_acc, epoch_pr, epoch_re = [], [], [], [], []
                    epoch_pr_micro, epoch_re_micro = [], []
                    epoch_dice_multi, epoch_acc_multi, epoch_pr_multi, epoch_re_multi = [], [], [], []
                    epoch_metrics_tumor = []
                    for i_batch, sampled_batch in enumerate(dataloaders[phase]):
                        iter_num_val += 1
                        net.eval()
                        inputs, label, label_hcc, cap_degree, cap_semantic, _, name = sampled_batch
                        inputs, label = inputs.cuda().float(), label.cuda().long()
                        args.vis_dir, args.name = snapshot_path, name
                        seq_label, vertexs = gen_seqlabel_cartcoord_torch(cap_degree, cap_semantic, args)  # (b, 90), (b, 90, 2)
                        seq_label = seq_label.cuda().long() # (b, 90)
                        if args.n_ray < 90:
                            ray_interval = int(90 / args.n_ray)
                            seq_label_tmp = torch.zeros_like(seq_label, device=seq_label.device)
                            vertexs_tmp = torch.zeros_like(vertexs, device=seq_label.device)
                            for i in range(0, 90, ray_interval):
                                j = int(i / ray_interval)
                                seq_label_tmp[:, j] = seq_label[:, i]
                                vertexs_tmp[:, j, :] = vertexs[:, i, :]
                            seq_label_90 = seq_label.clone()
                            seq_label = seq_label_tmp.clone()
                            vertexs = vertexs_tmp.clone()

                        out_seq, out_tumor = net(inputs, vertexs)  # (b, 3, 90)
                        seq_pred = torch.argmax(torch.softmax(out_seq, dim=1),dim=1)

                        # recover for visualization
                        inputs_vis = nn.Upsample(scale_factor=1 / 4, mode='bilinear')(inputs).cpu()
                        pred_recover = torch.zeros((inputs.size()[0], int(inputs.size()[2]/4), int(inputs.size()[3]/4)))
                        label_recover = torch.zeros_like(pred_recover)
                        for b in range(len(out_seq)):
                            for i in range(90):
                                # fill out degree map!!!!
                                pred_recover[b, vertexs[b, i, 0], vertexs[b, i, 1]] = torch.argmax(torch.softmax(out_seq[b], dim=0),dim=0)[i] + 1
                                label_recover[b, vertexs[b, i, 0], vertexs[b, i, 1]] = seq_label[b][i] + 1
                                degree_map_d = (cap_degree == i + 1).float()
                                degree_map_d[degree_map_d>0] = (seq_label[b][i] + 1).float()
                            vis(pred_recover[b].numpy(), 'pred_recover', name[b], vis_val_dir)
                            vis(label_recover[b].numpy(), 'label_recover', name[b], vis_val_dir)
                            vis(label[b].cpu().detach().numpy(), 'label_cap', name[b], vis_val_dir)
                            vis(label_hcc[b].cpu().detach().numpy(), 'label_tumor', name[b], vis_train_dir)
                            vis(torch.argmax(torch.softmax(out_tumor[b], dim=0), dim=0).cpu().detach().numpy(), 'pred_tumor', name[b], vis_train_dir)

                        # add_image in val
                        image = inputs_vis[0, 0:1, :, :]
                        image = (image - torch.min(image)) / (torch.max(image) - torch.min(image))
                        image = image.repeat(3, 1, 1)  # channel 0
                        writer.add_image('val/Image', image, iter_num_val)
                        writer.add_image('val/seq_pred_recover', pred_recover[0].unsqueeze(0) * 80, iter_num_val)
                        writer.add_image('val/seq_label_recover', label_recover[0].unsqueeze(0) * 80, iter_num_val)
                        out_seq_vis = torch.argmax(torch.softmax(out_seq, dim=1), dim=1, keepdim=True)
                        out_seq_vis = out_seq_vis.unsqueeze(3).repeat((1, 1, 1, 40)) + 1
                        writer.add_image('val/seq_pred', out_seq_vis[0, ...] * 80, iter_num_val)
                        seq_label_vis = seq_label.unsqueeze(1).unsqueeze(3).repeat((1, 1, 1, 40)) + 1
                        writer.add_image('val/seq_label', seq_label_vis[0, ...] * 80, iter_num_val)
                        out_tumor_vis = torch.argmax(torch.softmax(out_tumor, dim=1), dim=1, keepdim=True)
                        writer.add_image('val/tumor_pred', out_tumor_vis[0, ...] * 255, iter_num_val)
                        writer.add_image('val/tumor_label', label_hcc[0, ...].unsqueeze(0) * 255, iter_num_val)


                        # evaluating
                        seq_label, seq_pred = seq_label.cpu().detach().numpy(), seq_pred.cpu().detach().numpy()
                        for b in range(len(seq_label)):
                            f1_macro = f1_score(seq_label[b], seq_pred[b], average='macro')
                            f1_micro = f1_score(seq_label[b], seq_pred[b], average='micro')
                            precision = precision_score(seq_label[b], seq_pred[b], average='macro')
                            recall = recall_score(seq_label[b], seq_pred[b], average='macro')
                            precision_micro = precision_score(seq_label[b], seq_pred[b], average='micro')
                            recall_micro = recall_score(seq_label[b], seq_pred[b], average='micro')

                            acc = accuracy_score(seq_label[b], seq_pred[b])
                            dice_multi, acc_multi, pr_multi, re_multi = [], [], [], []
                            for i in range(3):
                                dice_multi.append(cal_1d_dice(seq_label[b] == i, seq_pred[b] == i))
                                acc_multi.append(accuracy_score(seq_label[b] == i, seq_pred[b] == i))
                                pr_multi.append(precision_score(seq_label[b] == i, seq_pred[b] == i))
                                re_multi.append(recall_score(seq_label[b] == i, seq_pred[b] == i))
                            epoch_f1_macro.append(f1_macro)
                            epoch_f1_micro.append(f1_micro)
                            epoch_pr.append(precision)
                            epoch_re.append(recall)
                            epoch_pr_micro.append(precision_micro)
                            epoch_re_micro.append(recall_micro)
                            epoch_acc.append(acc)
                            epoch_dice_multi.append(dice_multi)
                            epoch_acc_multi.append(acc_multi)
                            epoch_pr_multi.append(pr_multi)
                            epoch_re_multi.append(re_multi)

                            epoch_metrics_tumor.append(cal_metric_binary(label_hcc[b].cpu().detach().numpy(), torch.argmax(torch.softmax(out_tumor[b], dim=0), dim=0).cpu().detach().numpy())) # (b, 3), dsc, hd, asd
                    logging.info('Val epoch {} | avg acc : {} dice : {} f1_mi : {} f1_ma : {} pr_mi {} pr_ma {} re_mi {} re_ma {} tumor_dsc: {}'.format(
                        epoch_num, np.mean(epoch_acc), np.mean(epoch_dice_multi), np.mean(epoch_f1_micro), np.mean(epoch_f1_macro), np.mean(epoch_pr_micro), np.mean(epoch_pr), np.mean(epoch_re_micro), np.mean(epoch_re), np.mean(np.array(epoch_metrics_tumor)[:, 0])))

                    epoch_acc_multi = np.array(epoch_acc_multi)
                    epoch_pr_multi = np.array(epoch_pr_multi)
                    epoch_re_multi = np.array(epoch_re_multi)
                    epoch_dice_multi = np.array(epoch_dice_multi)

                    for i in range(3):
                        writer.add_scalar('val_epo/acc_g'+str(i+1), np.mean(epoch_acc_multi[:, i]), epoch_num)
                        writer.add_scalar('val_epo/pr_g' + str(i + 1), np.mean(epoch_pr_multi[:, i]), epoch_num)
                        writer.add_scalar('val_epo/re_g' + str(i + 1), np.mean(epoch_re_multi[:, i]), epoch_num)
                        writer.add_scalar('val_epo/dsc_g' + str(i + 1), np.mean(epoch_dice_multi[:, i]), epoch_num)

                    writer.add_scalar('val_epo/avg_acc', np.mean(epoch_acc), epoch_num)
                    writer.add_scalar('val_epo/f1_mac', np.mean(epoch_f1_macro), epoch_num)
                    writer.add_scalar('val_epo/f1_mic', np.mean(epoch_f1_micro), epoch_num)
                    writer.add_scalar('val_epo/pr_mac', np.mean(epoch_pr), epoch_num)
                    writer.add_scalar('val_epo/pr_mic', np.mean(epoch_pr_micro), epoch_num)
                    writer.add_scalar('val_epo/re_mac', np.mean(epoch_re), epoch_num)
                    writer.add_scalar('val_epo/re_mic', np.mean(epoch_re_micro), epoch_num)

                    writer.add_scalar('val_epo/avg_tumor_dsc', np.mean(np.array(epoch_metrics_tumor)[:, 0]), epoch_num)

                    performance_acc= np.mean(epoch_acc)
                    if performance_acc > best_performance_acc:
                        save_mode_path = os.path.join(snapshot_path, '_acc_best_model.pth')
                        torch.save(net.state_dict(), save_mode_path)
                        logging.info(' Best model on Acc | epoch {} | avg acc : {} dice : {} f1_mi : {} f1_ma : {} pr_mi {} pr_ma {} re_mi {} re_ma {} tumor_dsc: {}'.format(
                        epoch_num, np.mean(epoch_acc), np.mean(epoch_dice_multi), np.mean(epoch_f1_micro), np.mean(epoch_f1_macro), np.mean(epoch_pr_micro), np.mean(epoch_pr), np.mean(epoch_re_micro), np.mean(epoch_re), np.mean(np.array(epoch_metrics_tumor)[:, 0])))
                        best_performance_acc = performance_acc

                    performance_f1mi= np.mean(epoch_f1_micro)
                    if performance_f1mi > best_performance_f1mi:
                        save_mode_path = os.path.join(snapshot_path, '_f1mi_best_model.pth')
                        torch.save(net.state_dict(), save_mode_path)
                        logging.info(' Best model on F1 mi | epoch {} | avg acc : {} dice : {} f1_mi : {} f1_ma : {} pr_mi {} pr_ma {} re_mi {} re_ma {} tumor_dsc: {}'.format(
                        epoch_num, np.mean(epoch_acc), np.mean(epoch_dice_multi), np.mean(epoch_f1_micro), np.mean(epoch_f1_macro), np.mean(epoch_pr_micro), np.mean(epoch_pr), np.mean(epoch_re_micro), np.mean(epoch_re), np.mean(np.array(epoch_metrics_tumor)[:, 0])))
                        best_performance_f1mi = performance_f1mi

                    performance_f1ma= np.mean(epoch_f1_macro)
                    if performance_f1ma > best_performance_f1ma:
                        save_mode_path = os.path.join(snapshot_path, '_f1ma_best_model.pth')
                        torch.save(net.state_dict(), save_mode_path)
                        logging.info(' Best model on F1 ma | epoch {} | avg acc : {} dice : {} f1_mi : {} f1_ma : {} pr_mi {} pr_ma {} re_mi {} re_ma {} tumor_dsc: {}'.format(
                        epoch_num, np.mean(epoch_acc), np.mean(epoch_dice_multi), np.mean(epoch_f1_micro), np.mean(epoch_f1_macro), np.mean(epoch_pr_micro), np.mean(epoch_pr), np.mean(epoch_re_micro), np.mean(epoch_re), np.mean(np.array(epoch_metrics_tumor)[:, 0])))
                        best_performance_f1ma = performance_f1ma

                    if args.save_snapshot:
                        if np.mod(epoch_num + 1, 50) == 0:
                            save_mode_path = os.path.join(
                                snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
                            torch.save(net.state_dict(), save_mode_path)
                            logging.info("save model to {}".format(save_mode_path))

                if phase == 'val' and epoch_num%10==0 and epoch_num>=10:
                    phase = 'test'
                    epoch_f1_macro, epoch_f1_micro, epoch_acc, epoch_pr, epoch_re = [], [], [], [], []
                    epoch_pr_micro, epoch_re_micro = [], []
                    epoch_dice_multi, epoch_acc_multi, epoch_pr_multi, epoch_re_multi = [], [], [], []
                    epoch_metrics_tumor = []
                    for i_batch, sampled_batch in enumerate(dataloaders[phase]):
                        iter_num_test += 1
                        net.eval()
                        inputs, label, label_hcc, cap_degree, cap_semantic, _, name = sampled_batch
                        inputs, label = inputs.cuda().float(), label.cuda().long()
                        args.vis_dir, args.name = snapshot_path, name
                        seq_label, vertexs = gen_seqlabel_cartcoord_torch(cap_degree, cap_semantic, args)  # (b, 90), (b, 90, 2)
                        seq_label = seq_label.cuda().long() # (b, 90)
                        out_seq, out_tumor = net(inputs, vertexs)  # (b, 3, 90)
                        seq_pred = torch.argmax(torch.softmax(out_seq, dim=1),dim=1)

                        # recover for visualization
                        inputs_vis = nn.Upsample(scale_factor=1 / 4, mode='bilinear')(inputs).cpu()
                        pred_recover = torch.zeros((inputs.size()[0], int(inputs.size()[2]/4), int(inputs.size()[3]/4)))
                        label_recover = torch.zeros_like(pred_recover)
                        for b in range(len(out_seq)):
                            for i in range(90):
                                pred_recover[b, vertexs[b, i, 0], vertexs[b, i, 1]] = torch.argmax(torch.softmax(out_seq[b], dim=0),dim=0)[i] + 1
                                label_recover[b, vertexs[b, i, 0], vertexs[b, i, 1]] = seq_label[b][i] + 1
                            vis(pred_recover[b].numpy(), 'pred_recover', name[b], vis_test_dir)
                            vis(label_recover[b].numpy(), 'label_recover', name[b], vis_test_dir)
                            vis(label[b].cpu().detach().numpy(), 'label_cap', name[b], vis_test_dir)
                            vis(label_hcc[b].cpu().detach().numpy(), 'label_tumor', name[b], vis_test_dir)
                            vis(torch.argmax(torch.softmax(out_tumor[b], dim=0), dim=0).cpu().detach().numpy(), 'pred_tumor', name[b], vis_test_dir)

                        # add_image in test
                        image = inputs_vis[0, 0:1, :, :]
                        image = (image - torch.min(image)) / (torch.max(image) - torch.min(image))
                        image = image.repeat(3, 1, 1)  # channel 0
                        writer.add_image('test/Image', image, iter_num_test)
                        writer.add_image('test/seq_pred_recover', pred_recover[0].unsqueeze(0) * 80, iter_num_test)
                        writer.add_image('test/seq_label_recover', label_recover[0].unsqueeze(0) * 80, iter_num_test)
                        out_seq_vis = torch.argmax(torch.softmax(out_seq, dim=1), dim=1, keepdim=True)
                        out_seq_vis = out_seq_vis.unsqueeze(3).repeat((1, 1, 1, 40)) + 1
                        writer.add_image('test/seq_pred', out_seq_vis[0, ...] * 80, iter_num_test)
                        seq_label_vis = seq_label.unsqueeze(1).unsqueeze(3).repeat((1, 1, 1, 40)) + 1
                        writer.add_image('test/seq_label', seq_label_vis[0, ...] * 80, iter_num_test)
                        out_tumor_vis = torch.argmax(torch.softmax(out_tumor, dim=1), dim=1, keepdim=True)
                        writer.add_image('test/tumor_pred', out_tumor_vis[0, ...] * 255, iter_num_test)
                        writer.add_image('test/tumor_label', label_hcc[0, ...].unsqueeze(0) * 255, iter_num_test)

                        # evaluating
                        seq_label, seq_pred = seq_label.cpu().detach().numpy(), seq_pred.cpu().detach().numpy()
                        for b in range(len(seq_label)):
                            f1_macro = f1_score(seq_label[b], seq_pred[b], average='macro')
                            f1_micro = f1_score(seq_label[b], seq_pred[b], average='micro')
                            precision = precision_score(seq_label[b], seq_pred[b], average='macro')
                            recall = recall_score(seq_label[b], seq_pred[b], average='macro')
                            precision_micro = precision_score(seq_label[b], seq_pred[b], average='micro')
                            recall_micro = recall_score(seq_label[b], seq_pred[b], average='micro')
                            acc = accuracy_score(seq_label[b], seq_pred[b])
                            dice_multi, acc_multi, pr_multi, re_multi = [], [], [], []
                            for i in range(3):
                                dice_multi.append(cal_1d_dice(seq_label[b] == i, seq_pred[b] == i))
                                acc_multi.append(accuracy_score(seq_label[b] == i, seq_pred[b] == i))
                                pr_multi.append(precision_score(seq_label[b] == i, seq_pred[b] == i))
                                re_multi.append(recall_score(seq_label[b] == i, seq_pred[b] == i))
                            epoch_f1_macro.append(f1_macro)
                            epoch_f1_micro.append(f1_micro)
                            epoch_pr.append(precision)
                            epoch_re.append(recall)
                            epoch_pr_micro.append(precision_micro)
                            epoch_re_micro.append(recall_micro)
                            epoch_acc.append(acc)
                            epoch_dice_multi.append(dice_multi)
                            epoch_acc_multi.append(acc_multi)
                            epoch_pr_multi.append(pr_multi)
                            epoch_re_multi.append(re_multi)
                            epoch_metrics_tumor.append(cal_metric_binary(label_hcc[b].cpu().detach().numpy(), torch.argmax(torch.softmax(out_tumor[b], dim=0), dim=0).cpu().detach().numpy())) # (b, 3), dsc, hd, asd


                    logging.info(' Testing epoch {} | avg acc : {} dice : {} f1_mi : {} f1_ma : {} pr_mi {} pr_ma {} re_mi {} re_ma {} tumor_dsc: {}'.format(
                        epoch_num, np.mean(epoch_acc), np.mean(epoch_dice_multi), np.mean(epoch_f1_micro), np.mean(epoch_f1_macro), np.mean(epoch_pr_micro), np.mean(epoch_pr), np.mean(epoch_re_micro), np.mean(epoch_re), np.mean(np.array(epoch_metrics_tumor)[:, 0])))

                    epoch_acc_multi = np.array(epoch_acc_multi)
                    epoch_pr_multi = np.array(epoch_pr_multi)
                    epoch_re_multi = np.array(epoch_re_multi)
                    epoch_dice_multi = np.array(epoch_dice_multi)
                    for i in range(3):
                        writer.add_scalar('test_epoch/acc_g'+str(i+1), np.mean(epoch_acc_multi[:, i]), epoch_num)
                        writer.add_scalar('test_epoch/pr_g' + str(i + 1), np.mean(epoch_pr_multi[:, i]), epoch_num)
                        writer.add_scalar('test_epoch/re_g' + str(i + 1), np.mean(epoch_re_multi[:, i]), epoch_num)
                        writer.add_scalar('test_epoch/dsc_g' + str(i + 1), np.mean(epoch_dice_multi[:, i]), epoch_num)

                    writer.add_scalar('test_epoch/avg_acc', np.mean(epoch_acc), epoch_num)
                    writer.add_scalar('test_epoch/f1_mac', np.mean(epoch_f1_macro), epoch_num)
                    writer.add_scalar('test_epoch/f1_mic', np.mean(epoch_f1_micro), epoch_num)
                    writer.add_scalar('test_epoch/pr_mac', np.mean(epoch_pr), epoch_num)
                    writer.add_scalar('test_epoch/pr_mic', np.mean(epoch_pr_micro), epoch_num)
                    writer.add_scalar('test_epoch/re_mac', np.mean(epoch_re), epoch_num)
                    writer.add_scalar('test_epoch/re_mic', np.mean(epoch_re_micro), epoch_num)
                    writer.add_scalar('test_epoch/avg_tumor_dsc', np.mean(np.array(epoch_metrics_tumor)[:, 0]), epoch_num)

        writer.close()