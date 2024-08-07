# 《精通transformer》 CH8
# 1. Longformer

import torch
from transformers import LongformerTokenizer, LongformerModel

tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
model = LongformerModel.from_pretrained('allenai/longformer-base-4096')

sequence = 'hello '*4093
inputs = tokenizer(sequence, return_tensors='pt')
print('input shape: ', inputs['input_ids'].shape) # (1, 4096)

outputs = model(**inputs)

# default attention window size is 512
# Window size refers to the size of an attention window around each token.
from transformers import LongformerConfig, PyTorchBenchmark, PyTorchBenchmarkArguments
config_longformer=LongformerConfig.from_pretrained(
    "allenai/longformer-base-4096")
config_longformer_window4=LongformerConfig.from_pretrained(
    "allenai/longformer-base-4096", 
    attention_window=4)

sequence_lengths=[128,256,512,1024,2048,4096]
models=["config_longformer","config_longformer_window4"]
configs=[eval(m) for m in models]

benchmark_args = PyTorchBenchmarkArguments(
    sequence_lengths= sequence_lengths, 
    batch_sizes=[1], 
    models= models)
benchmark = PyTorchBenchmark(
    configs=configs, 
    args=benchmark_args)
results = benchmark.run()

import matplotlib.pyplot as plt 

def plotMe(results,title="Time"):
    plt.figure(figsize=(8,8))
    fmts= ["rs--","go--","b+-","c-o"]
    q=results.memory_inference_result
    if title=="Time": 
        q=results.time_inference_result
    models=list(q.keys())
    seq=list(q[models[0]]['result'][1].keys())
    models_perf=[list(q[m]['result'][1].values()) for m in models] 
    plt.xlabel('Sequence Length') 
    plt.ylabel(title) 
    plt.title('Inference Result') 
    for perf,fmt in zip(models_perf,fmts):
        plt.plot(seq, perf,fmt)
    plt.legend(models)  
    plt.show() 

plotMe(results,"Memory")


# 2. BigBird
from transformers import BigBirdConfig

# Default Bird  with num_random_blocks=3, block_size=64
sparseBird = BigBirdConfig.from_pretrained("google/bigbird-roberta-base")
# Fuyll attention Bird:
fullBird = BigBirdConfig.from_pretrained(
    "google/bigbird-roberta-base", 
    attention_type="original_full")

sequence_lengths=[256,512,1024,2048, 3072, 4096]
models=["sparseBird","fullBird"]
configs=[eval(m) for m in models]
benchmark_args = PyTorchBenchmarkArguments(
    sequence_lengths=sequence_lengths,
    batch_sizes=[1],
    models=models)
benchmark = PyTorchBenchmark(
    configs=configs, 
    args=benchmark_args)
results = benchmark.run()

plotMe(results)
plotMe(results,"Memory")

# 3. Reformer
from transformers import ReformerConfig
fullReformer = ReformerConfig.from_pretrained("google/reformer-enwik8",
                                               lsh_attn_chunk_length=16384, 
                                              local_attn_chunk_length=16384)
sparseReformer = ReformerConfig.from_pretrained("google/reformer-enwik8")

sequence_lengths=[256, 512, 1024, 2048, 4096, 8192, 12000]
models=["fullReformer","sparseReformer"]
configs=[eval(e) for e in models]

benchmark_args = PyTorchBenchmarkArguments(
    sequence_lengths=sequence_lengths,
    batch_sizes=[1],
    models=models)
benchmark = PyTorchBenchmark(
    configs=configs, 
    args=benchmark_args)
results = benchmark.run()

plotMe(results)
plotMe(results,"Memory")