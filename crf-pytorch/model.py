import torch.nn as nn
import torch
import numpy as np

class CRF(nn.Module):
    def __init__(self, tag_class=4):
        super(CRF, self).__init__()
        self.tag_class = tag_class #这里4为标签种类大小
        self.trans = nn.Parameter(torch.rand((self.tag_class, self.tag_class)))

        a = 1


    def forward(self, input):
        return input
    # @staticmethod
    def loss(self, target_predict, target_true, lens):
        with torch.autograd.set_detect_anomaly(True):
            # target_predict = torch.tensor(target_predict_input)
            target_true = torch.as_tensor(target_true, dtype=torch.long)
            # 先计算sum h, 这里之所以没有用mask是由于计算的时候需要mask的部分都为0,相当于已经mask了
            target_true = nn.functional.one_hot(target_true)
            target_true = target_true[:, :, :-1]
            sum_h = torch.sum(target_predict * target_true,[1, 2])

            # 再计算sum g
            pre_tag = target_true[:, :-1,:]
            fut_tag = target_true[:, 1:,:]
            # fut_tag_t = fut_tag[:, 1:]
            pre_tag_unsqueeze = torch.unsqueeze(pre_tag, 3)
            fut_tag_unsqueeze = torch.unsqueeze(fut_tag, 2)
            trans_tag = torch.matmul(pre_tag_unsqueeze, fut_tag_unsqueeze)
            g_mtx = torch.matmul(self.trans, trans_tag.float())
            sum_g = torch.sum(g_mtx, [1, 2, 3])
            # 再用动态规划计算分母log(Z(x))
            Z_t_col = target_predict[:, 0, :].unsqueeze(2)
            Z_t = Z_t_col.repeat(1, 1, Z_t_col.shape[1])
            mask_cmp = torch.tensor(lens).float()
            for i in range(1,target_true.shape[1]):
                # 第一种方式，直接计算，会出现数据大小溢出NAN
                # coe_mtx = torch.exp((target_predict[:,i,:].unsqueeze(1).repeat(1, Z_t.shape[2], 1) + self.trans).float())
                # Z_t = torch.matmul(Z_t, coe_mtx)
                # 第二种方式，使用logsumexp，避免数据溢出
                coe_mtx = target_predict[:,i,:].unsqueeze(1).repeat(1, Z_t.shape[2], 1) + self.trans
                zeros = torch.zeros(len(lens)).float()
                ones = torch.ones(len(lens)).float()
                mask = torch.where(mask_cmp < i, zeros, ones).unsqueeze(1).repeat(1, coe_mtx.shape[1]).unsqueeze(2).repeat(1, 1, coe_mtx.shape[2])
                add_mtx = coe_mtx * mask
                Z_t = Z_t + add_mtx
                Z_t_sum = torch.logsumexp(Z_t, 2, keepdim=True)
                Z_t_col = torch.where(mask_cmp.unsqueeze(1).repeat(1, coe_mtx.shape[1]).unsqueeze(2) < i, Z_t_col, Z_t_sum)
                # Z_t_col = Z_t_sum
                Z_t = Z_t_col.repeat(1, 1, Z_t.shape[1])
            sum_Z_t = torch.sum(Z_t, [1, 2])
            loss_batch = sum_Z_t - sum_h - sum_g
            # loss_batch = - sum_h - sum_g
            loss_mean = torch.mean(loss_batch.float())
            # print(self.trans)
            return loss_mean
    # def path_eval(self, targer_true, target_predict):





class CNN_CRF_model(nn.Module):
    def __init__(self, char_num, crf_model, tag_class, embedding_dim=128):
        super(CNN_CRF_model, self).__init__()
        self.embedding = nn.Embedding(char_num + 1, embedding_dim) # char_num+1的原因是多了pad这一项
        self.hidden_dim = 128
        self.conv1_1 = nn.Conv1d(in_channels=embedding_dim,
                               out_channels=self.hidden_dim,
                               kernel_size=3,
                               padding=1)
        self.relu_1 = nn.ReLU()
        self.tag_class = tag_class
        self.conv1_2 = nn.Conv1d(in_channels=self.hidden_dim,
                               out_channels=self.tag_class,
                               kernel_size=3,
                               padding=1)
        self.softmax = nn.Softmax(dim=-1)
        self.crf = crf_model

    def forward(self, input, lens):
        emb = self.embedding(input)
        # mask_mtx = torch.nn.functional.one_hot(torch.as_tensor(lens)-1)
        # mask_mtx = np.zeros((emb.shape[0], emb.shape[1], emb.shape[1]))
        # for i in range(len(lens)):
        #     emb[i, lens[i]:, :] = 0
        emb = emb.permute(0, 2, 1)


        x1 = self.conv1_1(emb)
        x2 = self.relu_1(x1)
        x3 = self.conv1_2(x2)
        x4 = x3.permute(0, 2, 1)
        x5 = self.softmax(x4)
        x6 = self.crf(x5)
        x7 = torch.zeros_like(x6)
        for i in range(len(lens)):
            x7[i, :lens[i], :] = x6[i, :lens[i], :]
        return x7