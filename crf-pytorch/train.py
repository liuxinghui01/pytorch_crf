import torch
from data_processor import batch_proc, ids2segment, getAccuracy
from data_processor import getDictFromFile
from data_processor import MyDataSet
from torch.utils.data import DataLoader
from model import CNN_CRF_model, CRF
from torch import nn, optim
from tqdm import tqdm
from valid import viterbi

def train(args):
    char2id, id2char, sentence_ids, target_ids = getDictFromFile(corpus_path=args.corpus_path)
    train_dataset = MyDataSet(sentence_ids=sentence_ids[:int(len(sentence_ids)*args.train_ratio)],
                              target_ids=target_ids[:int(len(target_ids)*args.train_ratio)])
    val_dataset = MyDataSet(sentence_ids=sentence_ids[int(len(sentence_ids)*args.train_ratio):int(len(sentence_ids)*(args.train_ratio+args.val_ratio))],
                            target_ids=target_ids[int(len(target_ids)*args.train_ratio):int(len(target_ids)*(args.train_ratio+args.val_ratio))])
    train_dataLoader = DataLoader(dataset=train_dataset,
                                  batch_size=16,
                                  collate_fn=batch_proc)
    val_dataLoader = DataLoader(dataset=val_dataset,
                                  batch_size=16,
                                  collate_fn=batch_proc)
    crf = CRF(tag_class=4)
    model = CNN_CRF_model(len(char2id),
                          crf_model=crf,
                          tag_class=4,
                          embedding_dim=128)

    print(model)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True)
    for epoch in range(5):
        model.train()
        loss_list = []
        with tqdm(train_dataLoader) as train_pbar:
            for i, (sentence_id, target_id, lens) in enumerate(train_pbar):
                model.train()
                sentence_id = torch.as_tensor(sentence_id, dtype=torch.long)
                out = model(sentence_id, lens)
                loss = crf.loss(out, target_id, lens)
                loss.requires_grad_(True)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_list.append(loss)
                if i % 50 == 0:
                    loss_mean = torch.mean(torch.as_tensor(loss_list, dtype=torch.float))
                    loss_list = []
                    with tqdm(val_dataLoader) as val_pbar:
                        for j, (sentence_id, target_id, lens) in enumerate(val_pbar):
                            model.eval()
                            sentence_id = torch.as_tensor(sentence_id, dtype=torch.long)
                            out = model(sentence_id, lens)
                            max_path_batch = viterbi(out, crf.trans, lens)
                            # seg_result = ids2segment(sentence_id, id2char, max_path_batch)
                            # seg_result_true = ids2segment(sentence_id, id2char, target_id.tolist())
                            accuracy = getAccuracy(target_id, max_path_batch, lens)
                            train_pbar.set_description(
                                "train: epoch = %s, i = %s, loss:%s, acc:%s" % (epoch, i, loss_mean.item(),accuracy))
                            break



