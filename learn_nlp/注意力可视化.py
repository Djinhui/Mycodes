# 《精通transformer》 CH11

# Visualization with BertViz
'''
!pip install bertviz
!pip install transformers
!pip install ipywidgets
'''
## Head View
from bertviz import head_view
from transformers import BertTokenizer, BertModel

def get_bert_attentions(model_path, sentence_a, sentence_b):
    model = BertModel.from_pretrained(model_path, output_attentions=True)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    inputs = tokenizer.encode_plus(sentence_a, sentence_b, return_tensors='pt', add_special_tokens=True)
    token_type_ids = inputs['token_type_ids']
    input_ids = inputs['input_ids']
    attention = model(input_ids, token_type_ids=token_type_ids)[-1]
    input_id_list = input_ids[0].tolist() 
    tokens = tokenizer.convert_ids_to_tokens(input_id_list)
    return attention, tokens

model_path = 'bert-base-cased'
sentence_a = "The cat is very sad."
sentence_b = "Because it could not find food to eat."
attention, tokens=get_bert_attentions(model_path, sentence_a, sentence_b)
head_view(attention, tokens)


## Model View
from bertviz import model_view
from transformers import BertTokenizer, BertModel

def show_model_view(model, tokenizer, sentence_a, sentence_b=None, hide_delimiter_attn=False, display_mode="light"):
    inputs = tokenizer.encode_plus(sentence_a, sentence_b, return_tensors='pt', add_special_tokens=True)
    input_ids = inputs['input_ids']
    if sentence_b:
        token_type_ids = inputs['token_type_ids']
        attention = model(input_ids, token_type_ids=token_type_ids)[-1]
        sentence_b_start = token_type_ids[0].tolist().index(1)
    else:
        attention = model(input_ids)[-1]
        sentence_b_start = None
    input_id_list = input_ids[0].tolist() # Batch index 0
    tokens = tokenizer.convert_ids_to_tokens(input_id_list)  
    if hide_delimiter_attn:
        for i, t in enumerate(tokens):
            if t in ("[SEP]", "[CLS]"):
                for layer_attn in attention:
                    layer_attn[0, :, i, :] = 0
                    layer_attn[0, :, :, i] = 0
    model_view(attention, tokens, sentence_b_start, display_mode=display_mode)

model_path='bert-base-german-cased'
model = BertModel.from_pretrained(model_path, output_attentions=True)
tokenizer = BertTokenizer.from_pretrained(model_path)
show_model_view(model, tokenizer, sentence_a, sentence_b, hide_delimiter_attn=False, display_mode="light")


# Neuron View
from bertviz.transformers_neuron_view import BertModel, BertTokenizer
from bertviz.neuron_view import show
model_path='bert-base-german-cased'
sentence_a = "Die Katze ist sehr traurig."
sentence_b = "Weil sie zu viel gegessen hat"
model = BertModel.from_pretrained(model_path, output_attentions=True)
tokenizer = BertTokenizer.from_pretrained(model_path)
model_type = 'bert'
show(model, model_type, tokenizer, sentence_a, sentence_b, layer=8, head=11, display_mode="light")

#let us check  <8,11>  that is for pronoun-antecedent relation,  <2,6> is for nect token pattern