import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionConv1d(nn.Module):
    '''
    Convolution is equivalent with Unfold + Matrix Multiplication + Fold (or view to output shape)
    '''

    def __init__(self, kernel_size, out_channels):
        super(AttentionConv1d, self).__init__()
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.cosine_similarity = nn.CosineSimilarity(dim=1)

    def calculate_similarity(self, embedding, embedding_neighbor):
        similarity = self.cosine_similarity(embedding, embedding_neighbor)
        similarity = torch.unsqueeze(similarity, dim=1)
        return similarity

    def cal_local_attenttion(self, embedding, feature, kernel_size):
        # embedding for t-1
        embedding_l = torch.zeros_like(embedding)
        embedding_l[:, :, 1:] = embedding[:, :, :-1]
        similarity_l = self.calculate_similarity(embedding, embedding_l)

        # itself
        similarity_c = self.calculate_similarity(embedding, embedding)

        # embedding for t+1
        embedding_r = torch.zeros_like(embedding)
        embedding_r[:, :, :-1] = embedding[:, :, 1:]
        similarity_r = self.calculate_similarity(embedding, embedding_r)

        similarity = torch.cat([similarity_l, similarity_c, similarity_r], dim=1)  # [B, 3, T]

        # expand for D times
        batch, channel, temporal_length = feature.size()
        similarity_tile = torch.zeros(batch, kernel_size * channel, temporal_length).type_as(feature)
        similarity_tile[:, :channel * 1, :] = similarity[:, :1, :]
        similarity_tile[:, channel * 1:channel * 2, :] = similarity[:, 1:2, :]
        similarity_tile[:, channel * 2:, :] = similarity[:, 2:, :]

        return similarity_tile

    def forward(self, feature, embedding, weight):
        batch, channel, temporal_length = feature.size()
        inp = torch.unsqueeze(feature, dim=3)
        w = torch.unsqueeze(weight, dim=3)

        unfold = nn.Unfold(kernel_size=(self.kernel_size, 1), stride=1, padding=[1, 0])
        inp_unf = unfold(inp)
        # local attention
        attention = self.cal_local_attenttion(embedding, feature, kernel_size=self.kernel_size)
        inp_weight = inp_unf * attention
        inp_unf_t = inp_weight.transpose(1, 2)
        w_t = w.view(w.size(0), -1).t()
        results = torch.matmul(inp_unf_t, w_t)
        out_unf = results.transpose(1, 2)
        out = out_unf.view(batch, self.out_channels, temporal_length)
        return out


class FilterModule(nn.Module):
    def __init__(self, cfg):
        super(FilterModule, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=cfg.NETWORK.FEAT_DIM, out_channels=512, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU()
        )
        self.conv_2 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.conv_1(x)
        out = self.conv_2(out)
        return out


class BaseModule(nn.Module):
    def __init__(self, cfg):
        super(BaseModule, self).__init__()
        self.conv_1 = nn.Conv1d(in_channels=cfg.NETWORK.FEAT_DIM, out_channels=2048, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_1_att = AttentionConv1d(kernel_size=3, out_channels=2048)
        self.conv_2 = nn.Conv1d(in_channels=2048, out_channels=2048, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_2_att = AttentionConv1d(kernel_size=3, out_channels=2048)
        self.lrelu = nn.LeakyReLU()
        self.drop_out = nn.Dropout(cfg.NETWORK.CASMODULE_DROPOUT)

    def forward(self, x, embedding):
        feat1 = self.lrelu(self.conv_1_att(x, embedding, self.conv_1.weight))
        feat2 = self.lrelu(self.conv_2_att(feat1, embedding, self.conv_2.weight))
        feature = self.drop_out(feat2)
        return feat1, feature


class ClassifierModule(nn.Module):
    def __init__(self, cfg):
        super(ClassifierModule, self).__init__()
        self.conv = nn.Conv1d(in_channels=2048, out_channels=cfg.DATASET.CLS_NUM + 1, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        out = self.conv(x)
        return out


class EmbeddingModule(nn.Module):
    def __init__(self, cfg):
        super(EmbeddingModule, self).__init__()
        self.conv_1 = nn.Conv1d(in_channels=cfg.NETWORK.FEAT_DIM, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv1d(in_channels=512, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        out = self.lrelu(self.conv_1(x))
        out = self.conv_2(out)
        embedding = F.normalize(out, p=2, dim=1)
        return embedding


class Network(nn.Module):
    def __init__(self, cfg):
        super(Network, self).__init__()
        self.filter_module = FilterModule(cfg)
        self.base_module = BaseModule(cfg)
        self.classifier_module = ClassifierModule(cfg)
        self.softmax = nn.Softmax(dim=1)
        self.k = int(cfg.DATASET.NUM_SEGMENTS * cfg.NETWORK.TOPK_K_R)
        self.num_cls = cfg.DATASET.CLS_NUM
        self.embedding_module = EmbeddingModule(cfg)

    def forward(self, x, is_train):
        fore_weights = self.filter_module(x)
        x_supp = fore_weights * x

        embedding = self.embedding_module(x)
        _, feature_base_2 = self.base_module(x, embedding)
        cas_base = self.classifier_module(feature_base_2)

        feature_supp_1, feature_supp_2 = self.base_module(x_supp, embedding)
        cas_supp = self.classifier_module(feature_supp_2)
        score_base = torch.mean(torch.topk(cas_base, self.k, dim=2)[0], dim=2)
        score_supp = torch.mean(torch.topk(cas_supp, self.k, dim=2)[0], dim=2)

        if is_train:
            return score_base, cas_base, score_supp, cas_supp, embedding, fore_weights
        else:
            score_supp = self.softmax(score_supp)
            return score_supp, cas_supp, feature_supp_1


if __name__ == '__main__':
    import os
    import torch.optim as optim
    os.environ['CUDA_VISIBLE_DEVICES'] = '9'
    os.environ['OMP_NUM_THREADS'] = '1'
    import sys
    sys.path.insert(0, '../../lib')
    from config.default import config as cfg
    from config.default import update_config

    cfg_file = '../experiments/THUMOS14.yaml'
    update_config(cfg_file)

    data = torch.randn((2, 2048, 750)).cuda()
    embedding = torch.randn((2, 32, 750)).cuda()

    network = Network(cfg).cuda()
    score_base, _, _, _, _ = network(data, is_train=True)
    label = torch.randint_like(score_base, low=0, high=2)

    f_loss = nn.MSELoss()
    optimizer = optim.Adam(network.parameters(), lr=1e-4)

    loss = f_loss(score_base, label)
    optimizer.zero_grad()
    loss.backward()
    grad = network.embedding_module.conv_2.weight.grad
    print('after backward', grad.size(), grad.min(), grad.max())

