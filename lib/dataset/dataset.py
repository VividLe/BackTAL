from torch.utils.data import Dataset, DataLoader
import os
import numpy as np


class BackTALDataset(Dataset):
    def __init__(self, cfg, split):
        self.split = split
        self.train_split = cfg.DATASET.TRAIN_SPLIT
        self.base_dir = os.path.join(cfg.BASIC.ROOT_DIR, cfg.DATASET.DATA_DIR, split)
        self.datas = self._load_dataset()

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        file_name = self.datas[item]
        data = np.load(os.path.join(self.base_dir, file_name))
        feat_spa = data['feat_spa']
        feat_tem = data['feat_tem']
        cls_label = data['cls_label']
        vid_name = str(data['vid_name'])

        if self.split == self.train_split:
            click_label = data['point_label']
            return feat_spa, feat_tem, cls_label, click_label, vid_name
        else:
            frame_num = data['frame_num']
            fps = data['fps']
            duration = data['duration']
            return feat_spa, feat_tem, vid_name, frame_num, fps, cls_label, duration

    def _load_dataset(self):
        data_set = os.listdir(self.base_dir)
        data_set.sort()
        datas = [i for i in data_set if i.endswith('.npz')]
        return datas


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '../../lib')
    from config.default import config as cfg
    from config.default import update_config

    cfg_file = '../experiments/THUMOS14.yaml'
    update_config(cfg_file)
    train_dset = BackTALDataset(cfg, cfg.DATASET.TRAIN_SPLIT)
    train_loader = DataLoader(train_dset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
                              num_workers=cfg.BASIC.WORKERS, pin_memory=cfg.BASIC.PIN_MEMORY)

    for feat_spa, feat_tem, cls_label in train_loader:
        print(type(feat_spa), feat_spa.size(), type(cls_label), cls_label)

