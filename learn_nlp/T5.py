# # 《精通Transformer》CH04

import pandas as pd
df = pd.read_csv("TR2EN.txt",sep="\t").astype(str)
print(df)
'''
	EN	TR
0	Hi.	Merhaba.
1	Hi.	Selam.
2	Run!	Kaç!
3	Run!	Koş!
4	Run.	Kaç!
...	...	...
473030	A carbon footprint is the amount of carbon dio...	Bir karbon ayakizi bizim faaliyetlerimizin bir...
473031	At a moment when our economy is growing, our b...	Ekonomimizin büyüdüğü bir anda bizim işletmele...
473032	Using high heat settings while ironing synthet...	Sentetik kumaşları ütülerken yüksek ısı ayarla...
473033	If you want to sound like a native speaker, yo...	Eğer bir yerli gibi konuşmak istiyorsan, banço...
473034	If someone who doesn't know your background sa...	Senin geçmiş deneyimini bilmeyen biri senin bi...
'''

data = []
for item in df[:10500].iterrows():
    data.append(["translate english to turkish", item[1].EN, item[1].TR])

df = pd.DataFrame(data, columns=["prefix", "input_text", "target_text"])
train_df = df[:50]
eval_df = df[50:100]
print(train_df)
'''
	prefix	                  input_text	target_text
0	translate english to turkish	Hi.	     Merhaba.
1	translate english to turkish	Hi.	     Selam.
2	translate english to turkish	Run!	 Kaç!
3	translate english to turkish	Run!	 Koş!
4	translate english to turkish	Run.	 Kaç!
'''

import logging
import pandas as pd
from simpletransformers.t5 import T5Model, T5Args

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

model_args = T5Args()
model_args.max_seq_length = 96
model_args.train_batch_size = 20
model_args.eval_batch_size = 20
model_args.num_train_epochs = 1
model_args.evaluate_during_training = True
model_args.evaluate_during_training_steps = 30000
model_args.use_multiprocessing = False
model_args.fp16 = False
model_args.save_steps = -1
model_args.save_eval_checkpoints = False
model_args.no_cache = True
model_args.reprocess_input_data = True
model_args.overwrite_output_dir = True
model_args.preprocess_inputs = False
model_args.num_return_sequences = 1
model_args.wandb_project = "MT5 English-Turkish Translation"

model = T5Model("mt5", "google/mt5-small", args=model_args, use_cuda=False)
model.train_model(train_df, eval_data=eval_df)