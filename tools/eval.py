# -------------------------------------------------------------------------------------------------------
# Source code for "Background-Click Supervision for Temporal Action Localization"
# Written by Le Yang and Tao Zhao
# --------------------------------------------------------------------------------------------------------

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
from torch.utils.data import DataLoader
import sys

import _init_paths
from config.default import config as cfg
from config.default import update_config
import pprint
from models.network import Network
from dataset.dataset import BackTALDataset
from core.train_eval import evaluate
from core.functions import prepare_env, evaluate_mAP
from utils.utils import load_weights


def args_parser():
    parser = argparse.ArgumentParser(description='Implementation of SBF-Net')
    parser.add_argument('-dataset', help='Choose dataset to run', default='THUMOS14', choices=['THUMOS14', 'ActivityNet1.2', 'HACS'])
    parser.add_argument('-weight_file', help='Path of weight_file', default='../checkpoints/THUMOS14_best.pth')

    args = parser.parse_args()
    return args


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

    # path configuration
    cfg.BASIC.BACKUP_DIR = os.path.join(cfg.BASIC.CKPT_DIR, cfg.BASIC.TIME + cfg.BASIC.SUFFIX, 'codes_backup')
    cfg.TEST.RESULT_DIR = os.path.join(cfg.BASIC.CKPT_DIR, cfg.BASIC.TIME + cfg.BASIC.SUFFIX, 'results')
    cfg.BASIC.ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')

    if cfg.BASIC.SHOW_CFG:
        pprint.pprint(cfg)

    # prepare running environment for the whole project
    prepare_env(cfg)

    val_dset = BackTALDataset(cfg, cfg.DATASET.VAL_SPLIT)
    val_loader = DataLoader(val_dset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False,
                            num_workers=cfg.BASIC.WORKERS, pin_memory=cfg.BASIC.PIN_MEMORY)

    model = Network(cfg)
    model.cuda()

    weight_file = args.weight_file

    # directly evaluate model
    epoch = -1
    model = load_weights(model, weight_file)

    output_json_file_cas = evaluate(cfg, val_loader, model, epoch)
    mAP, average_mAP = evaluate_mAP(cfg, output_json_file_cas, os.path.join(cfg.BASIC.ROOT_DIR, cfg.DATASET.GT_FILE))

    print("average_mAP: {:.4f}".format(average_mAP))
    for i in range(len(cfg.TEST.IOU_TH)):
        print("mAP@{:.2f}: {:.4f}".format(cfg.TEST.IOU_TH[i], mAP[i]))


if __name__ == '__main__':
    main()
