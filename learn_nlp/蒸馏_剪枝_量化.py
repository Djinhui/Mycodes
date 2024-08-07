# 《精通Transformer》CH08

from sentence_transformers import SentenceTransformer 
from datasets import load_metric, load_dataset 
import math
import tensorflow as tf
import pandas as pd
import torch

stsb_metric = load_metric('glue', 'stsb') 
stsb = load_dataset('glue', 'stsb') 

mrpc_metric = load_metric('glue', 'mrpc') 
mrpc = load_dataset('glue','mrpc')

# 蒸馏模型
distilroberta = SentenceTransformer('stsb-distilroberta-base-v2')

def roberta_sts_benchmark(batch): 
    sts_encode1 = tf.nn.l2_normalize(distilroberta.encode(batch['sentence1']),axis=1) 
    sts_encode2 = tf.nn.l2_normalize(distilroberta.encode(batch['sentence2']),axis=1) 
    cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2),axis=1) 
    clip_cosine_similarities = tf.clip_by_value(cosine_similarities,-1.0,1.0)
    scores = 1.0 - tf.acos(clip_cosine_similarities) / math.pi 
    return scores

references = stsb['validation'][:]['label'] 
distilroberta_results = roberta_sts_benchmark(stsb['validation']) 


# 剪枝
from torch.nn.utils import prune
pruner = prune.L1Unstructured(amount=0.2)

state_dicts = distilroberta.state_dict()
for key in state_dicts:
    if 'weight' in key:
        state_dicts[key] = pruner(state_dicts[key])

distilroberta.load_state_dict(state_dicts)
distilroberta_results_p = roberta_sts_benchmark(stsb['validation'])

pd.DataFrame({ 
  "DistillRoberta":stsb_metric.compute(predictions=distilroberta_results, references=references),
  "DistillRobertaPruned":stsb_metric.compute(predictions=distilroberta_results_p, references=references)
}) 

# 量化
distilroberta = torch.quantization.quantize_dynamic(
    model=distilroberta,
    qconfig_spec = {torch.nn.Linear: torch.quantization.default_dynamic_qconfig}, dtype=torch.qint8)

distilroberta_results_pq = roberta_sts_benchmark(stsb['validation']) 

pd.DataFrame({ 
  "DistillRoberta":stsb_metric.compute(predictions=distilroberta_results, references=references), 
  "DistillRobertaPruned":stsb_metric.compute(predictions=distilroberta_results_p, references=references), 
  "DistillRobertaPrunedQINT8":stsb_metric.compute(predictions=distilroberta_results_pq, references=references) 
})

'''
	   DistillRoberta	DistillRobertaPruned	DistillRobertaPrunedQINT8
pearson  	0.888461	0.849915	            0.826784
spearmanr	0.889246	0.849125	            0.824857
'''