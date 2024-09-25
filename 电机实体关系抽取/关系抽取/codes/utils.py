# coding:utf-8
import torch
from config import *
conf = Config()


def extract_sub(pred_sub_heads,pred_sub_tails):
    subs = []
    heads = torch.arange(0,len(pred_sub_heads),device=conf.device)[pred_sub_heads == 1]
    tails = torch.arange(0,len(pred_sub_tails),device=conf.device)[pred_sub_tails == 1]

    for head,tail in zip(heads,tails):
        if tail >= head:
            subs.append((head.item(),tail.item()))
    return subs

def extract_obj_and_rel(obj_heads,obj_tails):
    obj_heads = obj_heads.T
    obj_tails = obj_tails.T
    rel_count = obj_heads.shape[0]
    obj_and_rels = []
    for rel_index in range(rel_count):
        obj_head = obj_heads[rel_index]
        obj_tail = obj_tails[rel_index]
        objs = extract_sub(obj_head,obj_tail)
        if objs:
            for obj in objs:
                start_index,end_index = obj
                obj_and_rels.append((rel_index,start_index,end_index))
    return obj_and_rels


def convert_score_to_zero_one(tensor):
    tensor[tensor >= 0.5] = 1
    tensor[tensor < 0.5] = 0
    return tensor





