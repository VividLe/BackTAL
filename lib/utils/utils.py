import torch
import numpy as np
import random
import os
import shutil


def fix_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)


def backup_codes(cfg, root_dir, res_dir, backup_list):
    if os.path.exists(res_dir):
        shutil.rmtree(res_dir)
    os.makedirs(res_dir)
    for name in backup_list:
        shutil.copytree(os.path.join(root_dir, name), os.path.join(res_dir, name))
    if cfg.BASIC.VERBOSE:
        print('codes backup at {}'.format(os.path.join(res_dir, name)))


def save_best_model(cfg, epoch, model, file_path):
    state = {'epoch': epoch,
             'state_dict': model.state_dict()}
    torch.save(state, file_path)
    if cfg.BASIC.VERBOSE:
        print('save model: %s' % file_path)
    return file_path


def save_best_record_txt(cfg, info, file_path):
    epoch, average_mAP, mAP = info
    tIoU_thresh = cfg.TEST.IOU_TH

    with open(file_path, "w") as f:
        f.write("Epoch: {}\n".format(epoch))
        f.write("average_mAP: {:.4f}\n".format(average_mAP))

        for i in range(len(tIoU_thresh)):
            f.write("mAP@{:.2f}: {:.4f}\n".format(tIoU_thresh[i], mAP[i]))


def load_weights(model, weight_file):
    checkpoint = torch.load(weight_file)
    model.load_state_dict(checkpoint['state_dict'])

    return model
