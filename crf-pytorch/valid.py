import torch


def viterbi(h, crf_trans, lens):
    '''输入h和crf训练出来的转移矩阵，利用vitebi算法解码出分词结果

    Args:
        h: h
        crf_trans: crf转移矩阵

    Returns:
        最大可能路径

    '''
    score = h[:,0,:]
    path = []
    max_path = []
    for i in range(1,h.shape[1]):
        score = (score + h[:,i,:]).unsqueeze(2).repeat(1, 1, crf_trans.shape[0]) + crf_trans
        score, index = torch.max(score, dim=1)
        path.append(index)
    path_mtx = torch.stack(path, dim=2)
    max_score, index = torch.max(score, dim=1)
    max_path_batch = []
    for k, item in enumerate(path_mtx):
        tmp_idx = index[k]
        max_path.append(tmp_idx.item())
        for j in range(lens[k]-1, 0, -1):
            tmp_idx = item[tmp_idx][j-1]
            max_path.append(tmp_idx.item())
        max_path.reverse()
        max_path_batch.append(max_path)
    return max_path_batch

