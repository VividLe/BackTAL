# -------------------------------------------------------------------------------------------------------
# Source code for "Background-Click Supervision for Temporal Action Localization"
# Written by Le Yang and Tao Zhao
# --------------------------------------------------------------------------------------------------------

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import sys

import _init_paths
from config.default import config as cfg
from config.default import update_config
import pprint
from models.network import Network
from dataset.dataset import BackTALDataset
from core.train_eval import train, evaluate
from core.functions import prepare_env, evaluate_mAP

from utils.utils import save_best_model, save_best_record_txt
from criterion.loss import BackTALLoss


def args_parser():
    parser = argparse.ArgumentParser(description='Implementation of BackTAL')
    parser.add_argument('-dataset', help='Choose dataset to run', default='THUMOS14', choices=['THUMOS14', 'ActivityNet1.2', 'HACS'])
    args = parser.parse_args()
    return args


def post_process(cfg, actions_json_file, best_mAP, epoch):
    mAP, average_mAP = evaluate_mAP(cfg, actions_json_file, os.path.join(cfg.BASIC.ROOT_DIR, cfg.DATASET.GT_FILE))
    flag_best = False
    if cfg.DATASET.NAME == "THUMOS14":
        # use mAP@0.5 as the metric
        mAP_5 = mAP[4]
        if mAP_5 > best_mAP:
            best_mAP = mAP_5
            flag_best = True
    elif cfg.DATASET.NAME == "ActivityNet1.2" or cfg.DATASET.NAME == "HACS":
        # use average mAP as the metric
        if average_mAP > best_mAP:
            best_mAP = average_mAP
            flag_best = True

    info = [epoch, average_mAP, mAP]

    return best_mAP, info, flag_best


def main():
    args = args_parser()

    if args.dataset == "THUMOS14":
        cfg_file = '../experiments/THUMOS14.yaml'
    elif args.dataset == "ActivityNet1.2":
        cfg_file = '../experiments/ActivityNet1.2.yaml'
    elif args.dataset == "HACS":
        cfg_file = '../experiments/HACS.yaml'
    else:
        print('Please select dataset from: [THUMOS14, ActivityNet1.2, HACS]')
        sys.exit(0)

    update_config(cfg_file)
    if cfg.BASIC.SHOW_CFG:
        pprint.pprint(cfg)

    # path configuration
    cfg.BASIC.LOG_DIR = os.path.join(cfg.BASIC.CKPT_DIR, cfg.BASIC.TIME + cfg.BASIC.SUFFIX, 'log')
    cfg.BASIC.BACKUP_DIR = os.path.join(cfg.BASIC.CKPT_DIR, cfg.BASIC.TIME + cfg.BASIC.SUFFIX, 'codes_backup')
    cfg.TEST.RESULT_DIR = os.path.join(cfg.BASIC.CKPT_DIR, cfg.BASIC.TIME + cfg.BASIC.SUFFIX, 'results')
    cfg.BASIC.ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')

    # prepare running environment for the whole project
    prepare_env(cfg)

    # log
    writer = SummaryWriter(log_dir=os.path.join(cfg.BASIC.ROOT_DIR, cfg.BASIC.LOG_DIR))

    # dataloader
    train_dset = BackTALDataset(cfg, cfg.DATASET.TRAIN_SPLIT)
    train_loader = DataLoader(train_dset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
                              num_workers=cfg.BASIC.WORKERS, pin_memory=cfg.BASIC.PIN_MEMORY)
    val_dset = BackTALDataset(cfg, cfg.DATASET.VAL_SPLIT)
    val_loader = DataLoader(val_dset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False,
                            num_workers=cfg.BASIC.WORKERS, pin_memory=cfg.BASIC.PIN_MEMORY)

    # network
    model = Network(cfg)
    model.cuda()

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.LR, betas=cfg.TRAIN.BETAS, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    # criterion
    criterion = BackTALLoss(cfg)

    best_mAP = -1
    for epoch in range(1, cfg.TRAIN.EPOCH_NUM+1):
        print('Epoch: %d:' % epoch)
        loss_average, loss_base_average, loss_supp_average, loss_norm_average = train(cfg, train_loader, model, optimizer, criterion)

        writer.add_scalar('loss_average/train', loss_average, epoch)
        writer.add_scalar('loss_base_average/train', loss_base_average, epoch)
        writer.add_scalar('loss_supp_average/train', loss_supp_average, epoch)
        writer.add_scalar('loss_norm_average/train', loss_norm_average, epoch)

        if cfg.BASIC.VERBOSE:
            print('loss_average %f' % loss_average)
            print('loss_base_average %f' % loss_base_average)
            print('loss_supp_average %f' % loss_supp_average)
            print('loss_norm_average %f' % loss_norm_average)

        if epoch % cfg.TEST.EVAL_INTERVAL == 0:
            actions_json_file = evaluate(cfg, val_loader, model, epoch)

            best_mAP, info, flag_best = post_process(cfg, actions_json_file, best_mAP, epoch)
            if flag_best:
                txt_file = os.path.join(cfg.BASIC.ROOT_DIR, cfg.TEST.RESULT_DIR, args.dataset + '.txt')
                save_best_record_txt(cfg, info, txt_file)
                model_file = os.path.join(cfg.BASIC.ROOT_DIR, cfg.TEST.RESULT_DIR, args.dataset + '.pth')
                save_best_model(cfg, epoch=epoch, model=model, file_path=model_file)

                epoch, average_mAP, mAP = info
                tIoU_thresh = cfg.TEST.IOU_TH

        print("Best peformance is:")
        print("average_mAP: {:.4f}".format(average_mAP))
        for i in range(len(tIoU_thresh)):
            print("mAP@{:.2f}: {:.4f}".format(tIoU_thresh[i], mAP[i]))

    writer.close()


if __name__ == '__main__':
    main()
