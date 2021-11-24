import torch
import torch.nn as nn
import cv2
from scipy.interpolate import interp1d
import numpy as np
import os
import json

from post_process.nms import nms


def L1Norm(x):
    return torch.abs(x).sum()


def minmax_norm(x):
    '''
    normalize the class activation map
    for each class, calculate the max value and min value,
    use it the normalize the activation sequence for this class
    Notice: delta is in shape [1, 21]
    '''
    max_val = nn.ReLU()(torch.max(x, dim=2)[0])
    max_val = torch.unsqueeze(max_val, dim=2)
    min_val = nn.ReLU()(torch.min(x, dim=2)[0])
    min_val = torch.unsqueeze(min_val, dim=2)
    delta = max_val - min_val
    delta[delta <= 0] = 1
    x_norm = (x - min_val) / delta

    return x_norm


def minmax_norm_feature(act_map, min_val=None, max_val=None):
    if min_val is None or max_val is None:
        relu = nn.ReLU()
        max_val = relu(torch.max(act_map, dim=1)[0])
        min_val = relu(torch.min(act_map, dim=1)[0])

    delta = max_val - min_val
    delta[delta <= 0] = 1
    ret = (act_map - min_val) / delta

    ret[ret > 1] = 1
    ret[ret < 0] = 0

    return ret


def upgrade_resolution(arr, scale):
    # convert data: [1, 750] --> [750, 1, 1]
    arr = np.expand_dims(arr, axis=2)
    arr = np.transpose(arr, (2, 0, 1))

    x = np.arange(0, arr.shape[0])
    f = interp1d(x, arr, kind='linear', axis=0, fill_value='extrapolate')
    scale_x = np.arange(0, arr.shape[0], 1 / scale)
    up_scale = f(scale_x)
    return up_scale


def grouping(arr):
    return np.split(arr, np.where(np.diff(arr) != 1)[0] + 1)


def convert_output_txt(proposals, cate_idx, video_name):
    localizations = list()
    for data in proposals:
        loc = [video_name, data[1], data[2], cate_idx, data[0]]
        localizations.append(loc)
    return localizations


def record_localizations_txt(actions, output_file):
    with open(output_file, 'w') as f:
        for item in actions:
            strout = '%s\t%.2f\t%.2f\t%d\t%.4f\n' % (item[0], item[1], item[2], item[3], item[4])
            f.write(strout)
    return


def convert_output_json(proposals, cate_name):
    localizations = list()
    for data in proposals:
        loc = dict()
        loc['label'] = cate_name
        loc['segment'] = [data[1], data[2]]
        loc['score'] = data[0]
        localizations.append(loc)

    return localizations


def record_localizations_json(loc_result, result_file):
    '''
    Prepare the output following the ActivityNet format
    '''
    output_dict = {'version': 'VERSION 1.3', 'results': loc_result, 'external_data': {}}
    outfile = open(result_file, 'w')
    json.dump(output_dict, outfile)
    outfile.close()
    return


def get_localization(cfg, position, cas, video_cls_score, frame_num, fps, duration):
    '''
    convert position to action proposals
    '''
    grouped_list = grouping(position)

    actions = list()
    for proposal in grouped_list:
        start_time = proposal[0] / frame_num * duration
        end_time = (proposal[-1] + 1) / frame_num * duration

        inner_score = np.mean(cas[proposal])

        outer_s = max(0, int(proposal[0] - cfg.TEST.OUTER_LAMBDA * len(proposal)))
        outer_e = min(frame_num, int(proposal[-1] + cfg.TEST.OUTER_LAMBDA * len(proposal)))
        outer_scope_list = list(range(outer_s, int(proposal[0]))) + list(range(proposal[-1]+1, outer_e))
        if len(outer_scope_list) == 0:
            outer_score = 0
        else:
            outer_score = np.mean(cas[outer_scope_list])

        score = inner_score - outer_score + cfg.TEST.CONF_GAMMA * video_cls_score

        if cfg.DATASET.NAME == "ActivityNet1.2" or cfg.DATASET.NAME == "HACS":
            dur = end_time - start_time
            start_time = start_time + dur * 0.02
            end_time = end_time - dur * 0.02

        actions.append([score, start_time, end_time])

        if cfg.DATASET.NAME == "ActivityNet1.2" or cfg.DATASET.NAME == "HACS":
            start_time = 0
            end_time = duration
            actions.append([0.6, start_time, end_time])

    return actions


def write_results(cfg, epoch, data_json):
    output_json_file = os.path.join(cfg.BASIC.ROOT_DIR, cfg.TEST.RESULT_DIR, str(epoch).zfill(3)+ '.json')
    record_localizations_json(data_json, output_json_file)

    return output_json_file


def localize_actions(cfg, cas, cls_score_video, frame_num, fps, duration, activity_list, feature1):
    '''
    cas: class activation sequence
    '''
    if cfg.DATASET.NAME == "THUMOS14":
        feat_magnitudes = torch.norm(feature1, p=2, dim=1)
        feat_magnitudes = minmax_norm_feature(feat_magnitudes)
        feat_magnitudes_np = feat_magnitudes.cpu().data.numpy()
        feat_magnitudes_np_resize = cv2.resize(feat_magnitudes_np, (frame_num, 1), interpolation=cv2.INTER_LINEAR)
        feat_magnitudes_np_resize = feat_magnitudes_np_resize[0, :]

    # only dispose classes with confident classification scores
    confident_cates = np.where(cls_score_video >= cfg.TEST.CLS_SCORE_TH)[0]

    localizations_json = list()
    for class_id in confident_cates:
        cate_name = activity_list[class_id]
        cate_score = cas[class_id:class_id+1, :]

        # interpolate to input temporal length  [1, T]
        scores = cv2.resize(cate_score, (frame_num, 1), interpolation=cv2.INTER_LINEAR)
        # dispose for one class each time
        assert scores.shape[0] == 1
        scores = scores[0, :]

        # use the watershed algorithm
        actions = list()
        for th in list(np.arange(cfg.TEST.ACT_THRESH_MIN, cfg.TEST.ACT_THRESH_MAX, cfg.TEST.ACT_THRESH_STEP)):
            cas_temp = scores.copy()
            cas_temp[np.where(cas_temp < th)] = 0
            position = np.where(cas_temp > 0)
            # position is in a list, select the first element
            position = position[0]
            if any(position):
                proposals = get_localization(cfg, position, cas_temp, cls_score_video[class_id], frame_num, fps, duration)
                actions.extend(proposals)

        # use feature magnitudes produce proposal
        if cfg.DATASET.NAME == "THUMOS14":
            for th in list(np.arange(0.3, 0.8, 0.075)):
                cas_temp = scores.copy()
                feat_magnitudes_np_temp = feat_magnitudes_np_resize.copy()
                feat_magnitudes_np_temp[np.where(feat_magnitudes_np_temp < th)] = 0
                position = np.where(feat_magnitudes_np_temp > 0)
                # position is in a list, select the first element
                position = position[0]
                if any(position):
                    proposals = get_localization(cfg, position, cas_temp, cls_score_video[class_id], frame_num, fps, duration)
                    actions.extend(proposals)

        if any(actions):
            proposals_after_filter = nms(actions, cfg.TEST.NMS_THRESHOLD)
            locs_json = convert_output_json(proposals_after_filter, cate_name)
            localizations_json.extend(locs_json)

    return localizations_json


def evaluate_score(cfg, cls_score, cas, cls_label, vid_name, frame_num, fps, duration, activity_list, feature1):
    cas_base = minmax_norm(cas)
    cas_base = torch.squeeze(cas_base, dim=0)
    cas_base = cas_base.data.cpu().numpy()

    # discard the last data
    score_np = cls_score[0, :-1].data.cpu().numpy()

    locs_json = localize_actions(cfg, cas_base, score_np, frame_num, fps, duration, activity_list, feature1)

    cls_label_np = cls_label.detach().cpu().numpy()
    score_np[np.where(score_np < cfg.TEST.CLS_SCORE_TH)] = 0
    score_np[np.where(score_np >= cfg.TEST.CLS_SCORE_TH)] = 1
    correct_pred = np.sum(cls_label_np == score_np, axis=1)

    num_correct = np.sum((correct_pred == cfg.DATASET.CLS_NUM).astype(np.float32))
    num_total = correct_pred.shape[0]

    return locs_json, num_correct, num_total
