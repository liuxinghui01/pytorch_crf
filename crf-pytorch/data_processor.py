import re
import warnings

import numpy as np
from torch.utils.data import Dataset, DataLoader
'''
function:对数据进行处理，标签和id的映射关系字典为：{'B': 0, 'M': 1, 'E': 2, 'S': 3}，
按照此映射方式对每一个汉字进行标注。
'''
def batch_proc(batch_data):
    '''collate_fn函数，对每个batch中的数据进行对齐

    '''
    lens = []
    # 找出最长的序列长度
    for item in batch_data:
        lens.append(len(item[0]))
    max_len = max(lens)
    sentences = []
    targets = []
    #遍历batch_data，进行填充操作，另外，将lens返回，后面计算loss以及其他有关任务的时候有用
    for item in batch_data:
        sentence = np.append(item[0], np.zeros(max_len - len(item[0])))
        sentences.append(sentence)
        target = np.append(item[1], np.array([[4] * (max_len - len(item[1]))]))
        targets.append(target)

    return sentences, targets, lens


def target_id_part(word):
    ''' 将输入的单词转化成分词的标签

    Args:
        word: String. 输入单词。

    Returns:
        根据单词长度输出不同标签。
    '''
    assert len(word) > 0
    if len(word) == 1:
        return [3]
    elif len(word) == 2:
        return [0, 2]
    else:
        return [0] + [1]*(len(word)-2) + [2]

def ids2segment(sentence_id, id2char, max_path_batch):
    sentence = [[id2char.get(id, 'unk') for id in ids] for ids in sentence_id.tolist()]
    for i in range(len(max_path_batch)):
        pass

def getAccuracy(target_id, max_path_batch, lens):
    target = []
    predict = []
    for i in range(len(max_path_batch)):
        target.extend(target_id[i].tolist()[:lens[i]])
        predict.extend(max_path_batch[i][:lens[i]])
    cmp_ret = [1 if target[t] == predict[t] else 0 for t in range(len(target))]
    acc = sum(cmp_ret)/len(cmp_ret)
    return acc

def getDictFromFile(corpus_path=None, read_type='readlines', corpus_encoding_type='utf-8', min_count=0):
    """Construct dictionary bases on the input corpus.

    Args:
        corpus_path: String. The corpus file path.
        read_type: String, it could be 'readlines', 'read' and 'readline'. The type of reading corpus file.
        corpus_encoding_type: String. The encoding type of the corpus file.
        min_count: Int. Minimum count in the char dictionary.

    Returns:
        word2id: Dict. The dictionary of the words in the corpus file. The key is word and the value is the corresponding id.
        id2word: Dict. Inverse map of word2id.

    Raises:
        RuntimeError: If data could not be read correctly from ths corpus file.
    """
    try:
        if read_type == 'readlines':
            # 读取数据，readlines按\n换行符进行分割，返回list
            lines_all = open(corpus_path, encoding=corpus_encoding_type).readlines()
    except Exception:
        warnings.warn(
            'corpus file read Error.',
            stacklevel=2)
    char_dict = {}
    lines = [re.split(' +', line.strip()) for line in lines_all]
    lines = [[word for word in line if word] for line in lines]
    str_lines = ["".join(line) for line in lines]

    for str_line in str_lines:
        for char in str_line:
            char_dict[char] = char_dict.get(char, 0) + 1
    char_dict_filtered = {key: value for (key, value) in char_dict.items() if value > min_count}
    char2id = {char: id + 1 for id, (char, value) in enumerate(char_dict_filtered.items())}
    id2char = {id: char for (char, id) in char2id.items()}
    sentence_ids = [[char2id.get(char, 0) for char in str_line] for str_line in str_lines]
    # 训练数据标签数据生成
    target_ids = []
    for i, line in enumerate(lines):
        target_id_item = []
        for word in line:
            target_id_item.extend(target_id_part(word))
        target_ids.append(target_id_item)
        assert len(target_id_item) == len(sentence_ids[i])
    return char2id, id2char, sentence_ids, target_ids


class MyDataSet(Dataset):
    def __init__(self, sentence_ids, target_ids):
        assert len(sentence_ids)==len(target_ids)
        super(MyDataSet, self).__init__()
        self.sentence_ids = sentence_ids
        self.target_ids = target_ids

    def __getitem__(self, i):
        return np.array(self.sentence_ids[i]), np.array(self.target_ids[i])

    def __len__(self):
        return len(self.sentence_ids)


# corpus_path = './icwb2-data/training/msr_training.utf8'
# getDictFromFile(corpus_path=corpus_path)
