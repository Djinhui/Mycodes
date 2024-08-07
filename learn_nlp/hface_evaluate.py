from datasets import list_metrics
from datasets import load_metric


metric = load_metric('glue','mrpc')
print(metric.inputs_description)

metric.compute(predictions=[0,1],references=[0,1])
