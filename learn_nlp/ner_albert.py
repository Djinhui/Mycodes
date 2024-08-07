import codecs
import numpy as np
from tqdm import trange
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import Dataset, DataLoader, TensorDataset,RandomSampler, SequentialSampler
from transformers import BertTokenizer
from transformers import AdamW
from transformers import AlbertForTokenClassification
from transformers import get_linear_schedule_with_warmup


def get_input_data(model_path, file_path):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    input_data = codecs.open(file_path, 'r', 'utf-8')
    # 将标注子句拆分为字列表和对应的标注列表
    datas = []
    labels = []
    tags = set()
    for line in input_data.readlines():
        linedata = list()
        linelabel = list()
        line = line.split()
        numNotO = 0
        for word in line:
            word = word.split('/')
            linedata.append(word[0])
            linelabel.append(word[1])
            tags.add(word[1])
            if word[1] != 'O':
                numNotO += 1
        if numNotO!= 0: # 只保存标注不全为O的子句
            datas.append(linedata)
            labels.append(linelabel)
    input_data.close()
    tags = ['B_ns', 'M_ns', 'E_ns',  'B_nr', 'M_nr', 'E_nr', 'B_nt', 'M_nt', 'E_nt', 'O']
    tag2id = {tag: idx for idx, tag in enumerate(tags)}
    id2tag = {idx: tag for idx, tag in enumerate(tags)}


    return tokenizer, datas, labels, tag2id, id2tag


def pad_sequences(sequences, maxlen=None, dtype='int32', padding='post', truncating='post', value=0.):
    lengths = [len(s) for s in sequences]
    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)
    x = np.full((nb_samples, maxlen), value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError("Truncating type '%s' not understood" % truncating)

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError("Padding type '%s' not understood" % padding)
    return x


def get_input_ids(datas, tokenizer, labels, tag2id):
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(seq) for seq in datas], maxlen=60)
    tags = pad_sequences([[tag2id[tag] for tag in seq] for seq in labels], maxlen=60)
    masks = (input_ids!= 0).astype(np.float)
    return input_ids, tags, masks


def utils(input_ids, tags, masks):
    tr_inputs, val_inputs, tr_tags, val_tags, tr_masks, val_masks = train_test_split(input_ids, tags, masks, test_size=0.2, random_state=42)
    tr_dataset = TensorDataset(torch.tensor(tr_inputs), torch.tensor(tr_masks), torch.tensor(tr_tags))
    val_dataset = TensorDataset(torch.tensor(val_inputs), torch.tensor(val_masks), torch.tensor(val_tags))
    tr_sampler = RandomSampler(tr_dataset)
    val_sampler = SequentialSampler(val_dataset)

    tr_dl = DataLoader(tr_dataset, sampler=tr_sampler, batch_size=16)
    val_dl = DataLoader(val_dataset, sampler=val_sampler, batch_size=16)
    return tr_dl, val_dl


def create_model(model_path, id2tag, train_dataloader):
    model = AlbertForTokenClassification.from_pretrained(model_path, num_labels=len(id2tag), output_attentions=False, output_hidden_states=False)
    
    FULL_FINETUNING = True
    if FULL_FINETUNING:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters()) # 仅训练最顶层分类层
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

    optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5)
    epochs = 20
    max_grad_norm = 1.0
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model, optimizer, max_grad_norm, scheduler, device, epochs

def train(model, optimizer, max_grad_norm, scheduler, device, epochs, tr_dl, val_dl, id2tag, tokenizer):
    loss_values, validation_loss_values = [], []
    for epoch in range(epochs, desc='Epoch'):
        model.train()
        total_loss = 0
        for step, batch in enumerate(tr_dl):
            batch = tuple(t.to(device) for t in batch)
            input_ids, masks, tags = batch
            model.zero_grad()
            loss, logits = model(input_ids, attention_mask=masks, labels=tags, token_type_ids=None)
            loss.backward()
            total_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()

            if step % 100 == 0:
                print("epoch: %d, step: %d, loss: %f" % (epoch, step, total_loss/(step+1)))
        loss_values.append(total_loss / len(tr_dl))
        print("epoch: %d, loss: %f" % (epoch, total_loss/len(tr_dl)))

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        predictions, true_labels = [], []
        for batch in val_dl:
            batch = tuple(t.to(device) for t in batch)
            input_ids, masks, tags = batch
            with torch.no_grad():
                loss, logits = model(input_ids, attention_mask=masks, labels=tags, token_type_ids=None)
                eval_loss += loss.item()
                logits = logits.detach().cpu().numpy()
                label_ids = tags.to('cpu').numpy()
                masks = masks.to('cpu').numpy()
                eval_loss += loss.mean().item()
                preds = torch.argmax(logits, dim=2)
                predictions.append(preds.masked_select(masks.bool()))
                true_labels.append(true_labels.masked_select(masks.bool()))
        eval_loss = eval_loss / len(val_dl)
        validation_loss_values.append(eval_loss)  
        predictions = torch.cat(predictions)
        true_labels = torch.cat(true_labels)

        pred_tags = [id2tag[idx] for idx in predictions.tolist()]
        valid_tags = [id2tag[idx] for idx in true_labels.tolist()]
        eval_accuracy = accuracy_score(valid_tags, pred_tags)
        print("eval_loss: %f, eval_accuracy: %f" % (eval_loss, eval_accuracy))
        print(classification_report(valid_tags, pred_tags))

    torch.save(model.state_dict(),'model.pth')


def main():
    model_path = 'albert_chinese_xlarge'
    file_path = 'renmin4.txt'
    tokenizer, datas, labels, tag2id, id2tag = get_input_data(model_path, file_path)
    input_ids, tags, masks = get_input_ids(datas, tokenizer, labels, tag2id)
    tr_dl, val_dl = utils(input_ids, tags, masks)
    model, optimizer, max_grad_norm, scheduler, device, epochs = create_model(model_path, id2tag, tr_dl)
    train(model, optimizer, max_grad_norm, scheduler, device, epochs, tr_dl, val_dl, id2tag, tokenizer)


if __name__ == '__main__':
    main()