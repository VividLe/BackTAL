import torch
import torch.nn as nn
import torch.nn.functional as F


dtype = torch.cuda.FloatTensor() if torch.cuda.is_available() else torch.FloatTensor()
dtypel = torch.cuda.LongTensor() if torch.cuda.is_available() else torch.LongTensor()


class EmbeddingLoss(nn.Module):
    def __init__(self, k, th_similar_min, th_different_max):
        super(EmbeddingLoss, self).__init__()
        self.th_similar_min = th_similar_min
        self.th_different_max = th_different_max
        self.k = k

    def cosine_similarity(self, x1, x2, eps=1e-8):
        '''
        pair-wise cosine distance
        x1: [M, D]
        x2: [N, D]
        similarity: [M, N]
        '''
        w1 = x1.norm(p=2, dim=1, keepdim=True)
        w2 = x2.norm(p=2, dim=1, keepdim=True)
        similarity = torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)
        return similarity

    def forward(self, cfg,  embeddings, click_labels, cas_supp, label):
        loss_batch = 0
        num_batch = embeddings.size(0)
        num_batch_dynamic = num_batch
        for ibat in range(num_batch):
            click_label = click_labels[ibat, :]
            embedding = embeddings[ibat, :, :]
            mask = torch.where(click_label == cfg.DATASET.CLS_NUM)[0]
            if not mask.numel():
                # print('[]')
                num_batch_dynamic -= 1
                continue
            bg_embedding = embedding[:, mask]
            bg_embedding = bg_embedding.t()  # [M, D]
            lable_b_idxes = torch.where(label[ibat]==1)[0]

            mask_list = []
            for label_b_idx in lable_b_idxes:
                mask = torch.topk(cas_supp[ibat][label_b_idx], self.k)[1]
                mask_list.extend(mask)
            fg_embedding = embedding[:, torch.tensor(mask_list)]
            fg_embedding = fg_embedding.t()

            # all fg embedings should be similar
            sim_fg2fg = self.cosine_similarity(fg_embedding, fg_embedding)
            sim_fg2fg_hard = torch.min(sim_fg2fg, dim=1)[0]
            zero = torch.zeros_like(sim_fg2fg_hard)
            loss_fg2fg = torch.max(self.th_similar_min - sim_fg2fg_hard, zero)
            loss_fg2fg = loss_fg2fg.mean()

            # all bg embeddings should be similar
            sim_bg2bg = self.cosine_similarity(bg_embedding, bg_embedding)
            sim_bg2bg_hard = torch.min(sim_bg2bg, dim=1)[0]
            zero = torch.zeros_like(sim_bg2bg_hard)
            loss_bg2bg = torch.max(self.th_similar_min - sim_bg2bg_hard, zero)
            loss_bg2bg = loss_bg2bg.mean()

            # fg embeddings should be different with bg embeddings
            sim_fg2bg = self.cosine_similarity(fg_embedding, bg_embedding)
            # fg2bg
            sim_fg2bg_hard = torch.max(sim_fg2bg, dim=1)[0]
            zero = torch.zeros_like(sim_fg2bg_hard)
            loss_fg2bg = torch.max(sim_fg2bg_hard - self.th_different_max, zero)
            loss_fg2bg = loss_fg2bg.mean()
            # bg2fg
            sim_bg2fg_hard = torch.max(sim_fg2bg, dim=0)[0]
            zero = torch.zeros_like(sim_bg2fg_hard)
            loss_bg2fg = torch.max(sim_bg2fg_hard - self.th_different_max, zero)
            loss_bg2fg = loss_bg2fg.mean()

            loss_batch = loss_batch + loss_fg2fg + loss_bg2bg + loss_fg2bg + loss_bg2fg

        loss_batch = loss_batch / num_batch_dynamic
        return loss_batch


class BackTALLoss(nn.Module):
    def __init__(self, cfg):
        super(BackTALLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.k = int(cfg.DATASET.NUM_SEGMENTS * cfg.NETWORK.TOPK_K_R)
        self.embedding_loss = EmbeddingLoss(self.k, th_similar_min=cfg.NETWORK.TH_SIMILAR_MIN, th_different_max=cfg.NETWORK.TH_DIFFERENT_MAX)

        self.k_for_topk_bg_loss = int(cfg.DATASET.NUM_SEGMENTS * cfg.NETWORK.TOPK_K_R_FOR_LOSS)

    def _cls_loss(self, scores, labels):
        '''
        calculate classification loss
        1. dispose label, ensure the sum is 1
        2. calculate topk mean, indicates classification score
        3. calculate loss
        '''
        labels = labels / (torch.sum(labels, dim=1, keepdim=True) + 1e-10)
        clsloss = -torch.mean(torch.sum(labels * F.log_softmax(scores, dim=1), dim=1), dim=0)
        return clsloss

    def forward(self, cfg, score_base, cas_base, score_supp, cas_supp,  embedding, fore_weights, label, click_label, vid_name):
        loss_dict = {}

        label_base = torch.cat((label, torch.ones((label.shape[0], 1)).cuda()), dim=1)
        label_supp = torch.cat((label, torch.zeros((label.shape[0], 1)).cuda()), dim=1)

        loss_base = self._cls_loss(score_base, label_base)
        loss_supp = self._cls_loss(score_supp, label_supp)
        loss_norm = torch.mean(torch.norm(fore_weights, p=1, dim=2))

        loss_embedding = self.embedding_loss(cfg, embedding, click_label, cas_supp, label)

        loss_click_base = self.ce(cas_base, click_label)

        # only bg click usp, topk mean as fg pseudo
        loss_gt_fg_bg_click = []
        if cfg.DATASET.NAME == "ActivityNet1.2" or cfg.DATASET.NAME  == 'HACS':
            score_base_tmp = torch.mean(torch.topk(cas_base, self.k_for_topk_bg_loss, dim=2)[0], dim=2)

        for b in range(cas_base.shape[0]):
            cls_idxes = torch.where(label[b]==1)[0]
            loss_gt_fg_bg_click_b_c = []
            bg_click_idx = torch.where(click_label[b]==cfg.DATASET.CLS_NUM)[0]
            if not bg_click_idx.numel():
                continue
            for cls_idx in cls_idxes:
                bg_click_value = torch.mean(cas_base[b, cls_idx, bg_click_idx])
                if cfg.DATASET.NAME == "ActivityNet1.2" or cfg.DATASET.NAME  == 'HACS':
                    fg_click_value = score_base_tmp[b, cls_idx]
                else:
                    fg_click_value = score_base[b, cls_idx]

                fg_bg_click_value = torch.cat((fg_click_value.unsqueeze(0).unsqueeze(0), bg_click_value.unsqueeze(0).unsqueeze(0)), dim=1)
                label_c = torch.cat((torch.ones((1, 1)).cuda(), torch.zeros((1, 1)).cuda()), dim=1)
                loss_gt_fg_bg_click_b_c.append(torch.sum(-label_c * F.log_softmax(fg_bg_click_value, dim=1), dim=1).unsqueeze(0))
            loss_gt_fg_bg_click_b_c = torch.mean(torch.cat(loss_gt_fg_bg_click_b_c, dim=1)).unsqueeze(0).unsqueeze(0)
            loss_gt_fg_bg_click.append(loss_gt_fg_bg_click_b_c)
        loss_gt_fg_bg_click = torch.mean(torch.cat(loss_gt_fg_bg_click, dim=1))

        loss_total = loss_base + loss_supp + cfg.TRAIN.C_LOSS_NORM * loss_norm + loss_click_base + loss_gt_fg_bg_click * cfg.TRAIN.C_LOSS_TOPK_BG +  loss_embedding * cfg.TRAIN.C_LOSS_EMBEDDING

        loss_dict["loss_base"] = loss_base
        loss_dict["loss_supp"] = loss_supp
        loss_dict["loss_norm"] = loss_norm
        loss_dict["loss_total"] = loss_total

        return loss_total, loss_dict
