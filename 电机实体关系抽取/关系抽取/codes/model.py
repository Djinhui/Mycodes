# coding:utf-8
import torch
import torch.nn as nn
from transformers import BertModel,AdamW

class CasRel(nn.Module):
    def __init__(self,conf):
        super(CasRel, self).__init__()
        self.bert = BertModel.from_pretrained(conf.bert_path)
        self.sub_heads_linear = nn.Linear(conf.bert_dim, 1)
        self.sub_tails_linear = nn.Linear(conf.bert_dim, 1)
        self.obj_heads_linear = nn.Linear(conf.bert_dim,conf.num_rel)
        self.obj_tails_linear = nn.Linear(conf.bert_dim,conf.num_rel)

    def get_encoded_text(self,token_ids,mask):
        encoded_text = self.bert(token_ids,attention_mask=mask)[0]
        print('encoded_text-->',encoded_text.shape)
        return encoded_text

    def get_subs(self,encoded_text):
        pre_sub_heads = torch.sigmoid(self.sub_heads_linear(encoded_text))
        print('pre_sub_heads--->',pre_sub_heads.shape)
        pre_sub_tails = torch.sigmoid(self.sub_tails_linear(encoded_text))
        print('pre_sub_tails--->',pre_sub_tails.shape)
        return pre_sub_heads,pre_sub_tails

    def get_obj_for_specific_sub(self,sub_head2tail,sub_len,encoded_text):
        sub = torch.matmul(sub_head2tail,encoded_text)
        sub_len = sub_len.unsqueeze(1)
        sub = sub / sub_len
        encoded_text = encoded_text + sub
        pred_obj_heads = torch.sigmoid(self.obj_heads_linear(encoded_text))
        pred_obj_tails = torch.sigmoid(self.obj_tails_linear(encoded_text))

        return pred_obj_heads,pred_obj_tails


    def forward(self,input_ids,mask,sub_head2tail,sub_len):
        encoded_text = self.get_encoded_text(input_ids,mask)
        pred_sub_heads,pred_sub_tails = self.get_subs(encoded_text)
        print('pred_sub_heads--->', pred_sub_heads.shape)
        print('pred_sub_tails--->',pred_sub_tails.shape)
        sub_head2tail = sub_head2tail.unsqueeze(1)
        print('sub_head2tail--->',sub_head2tail.shape)
        pred_obj_heads,pred_obj_tails = self.get_obj_for_specific_sub(sub_head2tail,sub_len,encoded_text)
        print('pred_obj_heads',pred_obj_heads.shape)
        print('pred_obj_tails',pred_obj_tails.shape)
        result_dict = {
            'pred_sub_heads':pred_sub_heads,
            'pred_sub_tails':pred_sub_tails,
            'pred_obj_heads':pred_obj_heads,
            'pred_obj_tails':pred_obj_tails,
            'mask':mask
        }
        return result_dict

    def compute_loss(self,
                     pred_sub_heads,pred_sub_tails,
                     pred_obj_heads,pred_obj_tails,
                     mask,
                     sub_heads,sub_tails,
                     obj_heads,obj_tails):
        rel_count = obj_heads.shape[-1]
        rel_mask = mask.unsqueeze(-1).repeat(1,1,rel_count)
        loss_1 = self.loss(pred_sub_heads,sub_heads,mask)
        loss_2 = self.loss(pred_sub_tails,sub_tails,mask)
        loss_3 = self.loss(pred_obj_heads,obj_heads,rel_mask)
        loss_4 = self.loss(pred_obj_tails,obj_tails,rel_mask)
        return loss_1 + loss_2 + loss_3 + loss_4

    def loss(self,pred,gold,mask):
        pred = pred.squeeze(-1)
        loss = nn.BCELoss(reduction='none')(pred,gold)
        if loss.shape != mask.shape:
            mask = mask.unsqueeze(-1)
        loss = torch.sum(loss * mask) / torch.sum(mask)
        return loss

def load_model(conf):
    device = conf.device
    model = CasRel(conf)
    model.to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias','LayerNorm.bias','LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n,p in param_optimizer if not any(nd in n for nd in no_decay)],'weight_decay':0.01},
        {'params': [p for n,p in param_optimizer if any(nd in n for nd in no_decay)],'weight_decay':0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,lr=conf.learning_rate,eps=10e-8)
    sheduler = None
    return model,optimizer,sheduler,device


