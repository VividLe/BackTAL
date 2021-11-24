import torch
import os
import json
from post_process.functions import evaluate_score
from post_process.functions import write_results


dtype = torch.cuda.FloatTensor() if torch.cuda.is_available() else torch.FloatTensor()
dtypel = torch.cuda.LongTensor() if torch.cuda.is_available() else torch.LongTensor()


def train(cfg, data_loader, model, optimizer, criterion):
    model.train()

    loss_record = 0
    loss_dict_record ={"loss_base":0, "loss_supp":0, "loss_norm":0}

    for feat_spa, feat_tem, cls_label, click_label, vid_name in data_loader:
        feature = torch.cat([feat_spa, feat_tem], dim=1)
        feature = feature.type_as(dtype)
        cls_label = cls_label.type_as(dtype)
        click_label = click_label.type_as(dtypel)

        score_base, cas_base, score_supp, cas_supp, embedding, fore_weights = model(feature, is_train=True)
        loss_total, loss_dict = criterion(cfg, score_base, cas_base,  score_supp, cas_supp, embedding, fore_weights, cls_label, click_label, vid_name)

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        loss_record += loss_total.item()
        loss_dict_record["loss_base"] += loss_dict["loss_base"].item()
        loss_dict_record["loss_supp"] += loss_dict["loss_supp"].item()
        loss_dict_record["loss_norm"] += loss_dict["loss_norm"].item()

    loss_average = loss_record / len(data_loader)
    loss_base_average = loss_dict_record["loss_base"] / len(data_loader)
    loss_supp_average = loss_dict_record["loss_supp"] / len(data_loader)
    loss_norm_average = loss_dict_record["loss_norm"] / len(data_loader)

    return loss_average, loss_base_average, loss_supp_average, loss_norm_average


def evaluate(cfg, data_loader, model, epoch):
    gt_file = os.path.join(cfg.BASIC.ROOT_DIR, cfg.DATASET.GT_FILE)
    with open(gt_file, 'r') as f:
        gts_data = json.load(f)
    activity_list = gts_data['activity_list']

    model.eval()

    localizations_results = dict()

    for feat_spa, feat_tem, vid_name, frame_num, fps, cls_label, duration in data_loader:
        feature = torch.cat([feat_spa, feat_tem], dim=1)
        feature = feature.type_as(dtype)
        vid_name = vid_name[0]
        frame_num = frame_num.item()
        fps = fps.item()
        duration = duration.item()

        with torch.no_grad():
            score_cas, sequence_cas, feature_level1 = model(feature, is_train=False)

        locs_json, num_correct, num_total = evaluate_score(cfg, score_cas, sequence_cas, cls_label, vid_name, frame_num, fps, duration, activity_list, feature_level1)

        localizations_results[vid_name] = locs_json

    output_json_file = write_results(cfg, epoch, localizations_results)

    return output_json_file
