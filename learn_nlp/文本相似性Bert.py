import os
import time
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from hanziconv import HanziConv
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from transformers.optimization import AdamW
from sys import platform


"""
text_a                 text_b              label
在python中如何实现字典  如何用python构建字典  1
在python中如何实现字典  java很棒             0
"""
class DataPrecessForSentence(Dataset):
    def __init__(self, bert_tokenizer, LCQMC_file, pred=None, max_char_len=103):
        self.bert_tokenizer = bert_tokenizer
        self.max_seq_len = max_char_len
        self.seqs, self.seq_masks, self.seg_segments, self.labels = self.get_input(LCQMC_file, pred)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.seqs[idx], self.seq_masks[idx], self.seg_segments[idx], self.labels[idx]
    
    def get_input(self, file, pred=None):
        if pred:
            sentences1 = []
            sentences2 = []
            for i, j in enumerate(file):
                sentences1.append(j[0])
                sentences2.append(j[1])
            sentences1 = map(HanziConv.toSimplified, sentences1)
            sentences2 = map(HanziConv.toSimplified, sentences2)
            labels = [0] * len(file)
        else:
            df = pd.read_csv(file, sep='\t')
            sentences1 = map(HanziConv.toSimplified, df['text_a'].values)
            sentences2 = map(HanziConv.toSimplified, df['text_b'].values)
            labels = df['label'].values

        tokens_seq_1s = list(map(self.bert_tokenizer.tokenize, sentences1))
        tokens_seq_2s = list(map(self.bert_tokenizer.tokenize, sentences2))

        result = list(map(self.trunate_and_pad, tokens_seq_1s, tokens_seq_2s))

        seqs = [i[0] for i in result]
        seq_maks = [i[1] for i in result]
        seq_segments = [i[2] for i in result]
        return torch.Tensor(seqs).type(torch.long), torch.Tensor(seq_maks).type(torch.long), torch.Tensor(seq_segments).type(torch.long),torch.Tensor(labels).type(torch.long)
    

    def trunate_and_pad(self, tokens_seq_1, tokens_seq_2):
        if len(tokens_seq_1) > (self.max_seq_len - 3)//2:
            tokens_seq_1 = tokens_seq_1[:(self.max_seq_len-3)//2]
        if len(tokens_seq_2) > (self.max_seq_len - 3)//2:
            tokens_seq_2 = tokens_seq_2[:(self.max_seq_len-3)//2]

        seq = ['[CLS]'] + tokens_seq_1 + ['[SEP]'] + tokens_seq_2 + ['[SEP]']
        seg_segment = [0]*(len(tokens_seq_1) + 2) + [1] * len(tokens_seq_2) + 1

        seq = self.bert_tokenizer.convert_tokens_to_ids(seq)
        padding = [0]*(self.max_seq_len - len(seq))
        seg_mask = [1]*len(seq) + padding
        seg_segment = seg_segment + padding
        seq += padding

        assert len(seq) == self.max_seq_len
        assert len(seg_mask) == self.max_seq_len
        assert len(seg_segment) == self.max_seq_len
        return seq, seg_mask, seg_segment


def correct_predictions(output_probs, targets):
    _, out_classes = output_probs.max(dim=1)
    correct = (out_classes == targets).sum()
    return correct.item()


def train(model:nn.Module, dataloader, optimizer, max_gradient_norm):
    model.train()
    device = model.device
    epoch_start = time.time()
    batch_time_avg = 0.0
    running_loss = 0.0
    correct_preds = 0
    tqdm_batch_iterator = tqdm(dataloader)
    for batch_index, (batch_seqs, batch_seq_masks, batch_seq_segments, batch_labels) in enumerate(tqdm_batch_iterator):
        batch_start = time.time()
        seqs, masks, segments, labels = batch_seqs.to(device), batch_seq_masks.to(device), batch_seq_segments.to(device), batch_labels.to(device)

        optimizer.zero_grad()
        loss, logits, probs = model(seqs, masks, segments, labels)   
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)     
        optimizer.step()

        batch_time_avg += time.time() - batch_start
        running_loss += loss.item()
        correct_preds += correct_predictions(probs, labels)
        dec = 'Avg. batch proc. time:{:.4f}s, loss:{:.4f}'.format(batch_time_avg/(batch_index+1),
                                                                  running_loss/(batch_index+1))
        tqdm_batch_iterator.set_description(desc=dec)

    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct_preds / (len(dataloader.dataset))
    return epoch_time, epoch_loss, epoch_acc
    

def validate(model:nn.Module, dataloader:DataLoader):
    model.eval()
    device = model.device
    epoch_start = time.time()
    running_loss = 0.0
    running_acc = 0.0
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for (batch_seqs, batch_seq_masks, batch_seq_segments, batch_labels) in dataloader:
            seqs, masks, segments, labels = batch_seqs.to(device), batch_seq_masks.to(device), batch_seq_segments.to(device), batch_labels.to(device)

            loss, logits, probs = model(seqs, masks, segments, labels)
            running_loss += loss.item()
            running_acc += correct_predictions(probs, labels)
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(batch_labels)

        epoch_time = time.time() - epoch_start
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = running_acc / (len(dataloader.dataset))
        return epoch_time, epoch_loss, epoch_acc, roc_auc_score(all_labels, all_probs)




def predict(model, test_file, dataloader, device):
    model.eval()
    with torch.no_grad():
        result = []
        for (batch_seqs, batch_seq_masks, batch_seq_segments, batch_labels) in dataloader:
            seqs, masks, segments, labels = batch_seqs.to(device), batch_seq_masks.to(device), batch_seq_segments.to(device), batch_labels.to(device)
            _, _, probs = model(seqs, masks, segments, labels)
            result.append(probs)

    text_result = []
    for i, j in enumerate(test_file):
        text_result.append([j[0], j[1], '相似' if torch.argmax(result[i][0]) == 1 else '不相似'])
    return text_result



class BertModel(nn.Module):
    def __init__(self):
        super(BertModel, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained('pretrained_model', num_labels=2)
        self.device = torch.device('cpu')
        for param in self.bert.parameters():
            param.requires_grad_(True)

    def forward(self, batch_seqs, batch_seq_masks, batch_seg_segments,labels):
        loss, logits = self.bert(input_ids=batch_seqs, attention_mask=batch_seq_masks,
                                 token_type_ids=batch_seg_segments,labels=labels)
        probs = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probs
    

class BertModelTest(nn.Module):
    def __init__(self, model_path):
        super(BertModelTest, self).__init__()
        config = BertConfig.from_pretrained(model_path)
        self.bert = BertForSequenceClassification(config)
        self.device = torch.device('cpu')
    
    def forward(self, batch_seqs, batch_seq_masks, batch_seg_segments, labels):
        loss, logits = self.bert(input_ids=batch_seqs, attention_mask=batch_seq_masks,
                                 token_type_ids=batch_seg_segments,labels=labels)
        probs = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probs


def main_train(train_file, dev_file,target_dir, epochs=10,batch_size=32,lr=1e-5,patence=2,max_gradient_norm=1.0,checkpoint=None):
    bert_tokenizer = BertTokenizer.from_pretrained('pretrained_model/',do_lower_case=True)
    device = torch.device('cpu')
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    train_data = DataPrecessForSentence(bert_tokenizer, train_file)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

    dev_data = DataPrecessForSentence(bert_tokenizer, dev_file)
    dev_loader = DataLoader(dev_data, shuffle=True, batch_size=batch_size)

    model = BertModel().to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_groupped_parameters = [
        {
            'params':[p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],'weight_decay':0.01
        },
        {
            'params':[p for n, p in param_optimizer if any(nd in n for nd in no_decay)],'weight_decay':0.01
        }
    ]

    optimizer = AdamW(optimizer_groupped_parameters, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=0)

    best_score = 0.0
    start_epoch = 1
    epochs_count = []
    train_losses = []
    valid_losses = []
    if checkpoint:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        best_score = checkpoint['best_score']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epochs_count = checkpoint['epochs_count']
        train_losses = checkpoint['train_losses']
        valid_losses = checkpoint['valid_losses']

    _, valid_loss, valid_acc, auc = validate(model, dev_loader)

    patience_counter = 0
    for epoch in range(start_epoch, epochs+1):
        epochs_count.append(epoch)
        epoch_time, epoch_loss, epoch_acc = train(model, train_loader, optimizer, epoch, max_gradient_norm)
        train_losses.append(epoch_loss)
        print('Training time:{:.4f}s loss={:.4f}, acc:{:.4f}'.format(epoch_time, epoch_loss, epoch_acc))

        epoch_time, epoch_loss, epoch_acc, epoch_auc = validate(model, train_loader)
        valid_losses.append(epoch_loss)
        print('Valid time:{:.4f}s loss={:.4f}, acc:{:.4f}, auc:{:.4f}'.format(epoch_time, epoch_loss, epoch_acc, epoch_auc))

        scheduler.step(epoch_acc)
        if epoch_acc < best_score:
            patience_counter += 1
        else:
            best_score = epoch_acc
            patience_counter = 0
            torch.save({'epoch':epoch,
                        'model':model.state_dict(),
                        'best_score':best_score,
                        'epochs_count':epochs_count,
                        'train_losses':train_losses,
                        'valid_losses':valid_losses},
                        os.path.join(target_dir, 'best.pth.jar'))
            
        if patience_counter >= patence:
            break

main_train('lcqmc_data/train.csv', 'lcqmc_data/dev.csv','models')

def main_test(test_file, pretrained_file, batch_size=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bert_tokenizer = BertTokenizer.from_pretrained('pretrained_model/', do_lower_case=True)
    if platform =='linux' or platform == 'linux2':
        checkpoint = torch.load(pretrained_file)
    else:
        checkpoint = torch.load(pretrained_file, map_location=device)

    test_data = DataPrecessForSentence(bert_tokenizer,test_file, pred=True)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)

    model = BertModelTest('pretrained_model/').to(device)
    model.load_state_dict(checkpoint['model'])
    result = predict(model, test_file, test_loader, device)
    return result




text = [['微信号如何二次修改', '如何二次修改微信号'],
        ['红米刷什么系统好', '什么牌子的红米好吃']]

result = main_test(text, 'models/bert.pth.jar')
print(result)

'''
[['微信号如何二次修改', '如何二次修改微信号', '相似'],
 ['红米刷什么系统好', '什么牌子的红米好吃', '不相似']]

'''