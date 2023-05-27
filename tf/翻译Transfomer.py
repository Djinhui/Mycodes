# https://tensorflow.google.cn/text/tutorials/transformer
import logging
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_text


# 加载数据
examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                               with_info=True,
                               as_supervised=True)

train_examples, val_examples = examples['train'], examples['validation']

for pt_examples, en_examples in train_examples.batch(3).take(1):
    print('> Examples in Portuguese')
    for pt in pt_examples.numpy():
        print(pt.decode('utf-8'))
    print()

    print('> Examples in English')
    for en in en_examples.numpy():
        print(en.decode('utf-8'))

'''
> Examples in Portuguese:
e quando melhoramos a procura , tiramos a única vantagem da impressão , que é a serendipidade .
mas e se estes fatores fossem ativos ?
mas eles não tinham a curiosidade de me testar .

> Examples in English:
and when you improve searchability , you actually take away the one advantage of print , which is serendipity .
but what if it were active ?
but they did n't test for curiosity .

'''

# Tokenize the text:each element is represented as a token or token ID 

model_name = 'ted_hrlr_translate_pt_en_converter'  # subword tokenizer
tf.keras.utils.get_file(
    f'{model_name}.zip',
    f'https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip',
    cache_dir='.', cache_subdir='', extract=True
)

tokenizers = tf.saved_model.load(model_name)

[item for item in dir(tokenizers.en) if not item.startswith('_')]

print('> This is a batch of strings:')
for en in en_examples.numpy():
    print(en.decode('utf-8'))

'''
The tokenize method converts a batch of strings to a padded-batch of token IDs. 
This method splits punctuation, lowercases and unicode-normalizes the input before tokenizing.
'''
# 注意： the tokenized text includes '[START]' and '[END]' tokens.
encoded = tokenizers.en.tokenize(en_examples)
print('> This is a padded-batch of token IDs:')
for row in encoded.to_list():
    print(row)

''' 以2开始, 以3结束
> This is a padded-batch of token IDs:
[2, 72, 117, 79, 1259, 1491, 2362, 13, 79, 150, 184, 311, 71, 103, 2308, 74, 2679, 13, 148, 80, 55, 4840, 1434, 2423, 540, 15, 3]
[2, 87, 90, 107, 76, 129, 1852, 30, 3]
[2, 87, 83, 149, 50, 9, 56, 664, 85, 2512, 15, 3]
'''

round_trip = tokenizers.en.detokenize(encoded)

print('> This is human-readable text:')
for line in round_trip.numpy():
    print(line.decode('utf-8'))


print('> This is the text split into tokens:')
tokens = tokenizers.en.lookup(encoded)
tokens

# 句长分布
lengths = []
for pt_examples, en_examples in train_examples.batch(1024):
    pt_tokens = tokenizers.pt.tokenize(pt_examples)
    lengths.append(pt_tokens.row_lengths())

    en_tokens = tokenizers.en.tokenize(en_examples)
    lengths.append(en_tokens.row_lengths())
    print('.', end='', flush=True)


all_lengths = np.concatenate(lengths)
plt.hist(all_lengths, np.linspace(0, 500,101))
plt.ylim(plt.ylim())
max_length = max(all_lengths)
plt.plot([max_length, max_length], plt.ylim())
plt.title(f'Maximum tokens per example:{max_length}') # 最长320，集中在100以下

MAX_TOKENS = 128
def prepare_batch(pt, en):
    pt = tokenizers.pt.tokenize(pt) # 输出是不规则的， RaggedTensor
    pt = pt[:, :MAX_TOKENS] # Trim to MAX_TOKENS
    pt = pt.to_tensor() # Convert to 0-padded dense Tensor

    en = tokenizers.en.tokenize(en)
    en = en[:, :(MAX_TOKENS)+1]
    en_inputs = en[:, :-1].to_tensor() # Drop the [END] tokens
    en_labels = en[:, 1:].to_tensor() # Drop the [START] tokens, shifted by one step 'teacher forcing'
    return (pt, en_inputs), en_labels


BUFFER_SIZE = 20000
BATCH_SIZE = 64

def make_batches(ds):
    return (
        ds.
        shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
        .map(prepare_batch, tf.data.AUTOTUNE)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )


# Create training and validation set batches.
train_batches = make_batches(train_examples)
val_batches = make_batches(val_examples)

for (pt, en), en_labels in train_batches.take(1):
    break

print(pt.shape) # (64, 86)
print(en.shape) # (64, 81)
print(en_labels.shape) # (64, 81)

print(en[0][:10])
print(en_labels[0][:10])

'''
tf.Tensor([   2  476 2569 2626 6010   52 2564 1915  188   15], shape=(10,), dtype=int64)
tf.Tensor([ 476 2569 2626 6010   52 2564 1915  188   15    3], shape=(10,), dtype=int64)
'''

# 架构
# 1. Attention sublayer
# input embedding & positional encoding
# input embedding-> layers.Embedding

def positional_encoding(length, depth):
    depth = depth / 2
    
    positions = np.arange(length)[:, np.newaxis] # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] # (1, depth)
    angle_rates = 1 / (10000**depths) # (1, depth)
    angle_rads = positions * angle_rates # (pos,depth)

    pos_encoding = np.concatenate([np.sinc(angle_rads), np.column_stack(angle_rads)], axis=-1)
    return tf.cast(pos_encoding, dtype=tf.float32)

pos_encoding = positional_encoding(length=2048, depth=512)

# Check the shape.
print(pos_encoding.shape) # (2048, 512)

# Plot the dimensions.
plt.pcolormesh(pos_encoding.numpy().T, cmap='RdBu')
plt.ylabel('Depth')
plt.xlabel('Position')
plt.colorbar()
plt.show()


pos_encoding/=tf.norm(pos_encoding, axis=1, keepdims=True)
p = pos_encoding[1000]
dots = tf.einsum('pd,d -> p', pos_encoding, p)
plt.subplot(2,1,1)
plt.plot(dots)
plt.ylim([0,1])
plt.plot([950, 950, float('nan'), 1050, 1050],
         [0,1,float('nan'),0,1], color='k', label='Zoom')
plt.legend()
plt.subplot(2,1,2)
plt.plot(dots)
plt.xlim([950, 1050])
plt.ylim([0,1])

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True)
        self.pos_encoding = positional_encoding(length=2048, depth=d_model)

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)
    
    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x
    

'''
注意， 原始论文uses a single tokenizer and weight matrix for both the source and target languages. 
'''
embed_pt = PositionalEmbedding(vocab_size=tokenizers.pt.get_vocab_size(), d_model=512)
embed_en = PositionalEmbedding(vocab_size=tokenizers.en.get_vocab_size(), d_model=512)

pt_emb = embed_pt(pt)
en_emb = embed_en(en)

en_emb._keras_mask # shape=(64, 81)

# 

class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add() #  use the Add layer to ensure that Keras masks are propagated (the + operator does not)


class CrossAttention(BaseAttention):
    def call(self,x, context):
        attn_output, attn_scores = self.mha(query=x, key=context, value=context, return_attention_scores=True)

        self.last_attn_scores = attn_scores
        
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x
    
sample_ca = CrossAttention(num_heads=2, key_dim=512)

print(pt_emb.shape)  # (64, 86, 512)
print(en_emb.shape)  # (64, 81, 512) as query
print(sample_ca(en_emb, pt_emb).shape) (64, 81, 512)


class GlobalSelfAttention(BaseAttention): # MultiHeadAttention
    def call(self, x):
        attn_output = self.mha(query=x, key=x, value=x)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x
    
sample_gsa = GlobalSelfAttention(num_heads=2, key_dim=512)

print(pt_emb.shape) # (64, 86, 512)
print(sample_gsa(pt_emb).shape) # (64, 86, 512)


class CausalSelfAttention(BaseAttention): # Masked MultiHeadAttention
    def call(self, x):
        # The causal mask ensures that each location only has access to the locations that come before it
        attn_output = self.mha(query=x, value=x, key=x, use_causal_mask=True) 
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x
    

sample_csa = CausalSelfAttention(num_heads=2, key_dim=512)

print(en_emb.shape) # (64, 81, 512)
print(sample_csa(en_emb).shape) # (64, 81, 512)

'''
The output for early sequence elements doesn't depend on later elements, so it shouldn't matter if you trim elements before or after applying the layer
'''
out1 = sample_csa(embed_en(en[:, :3])) 
out2 = sample_csa(embed_en(en))[:, :3]

tf.reduce_max(abs(out1 - out2)).numpy() # 4.7683716e-07

# 2. Feed-Forwrad network sublayer

class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x
    

sample_ffn = FeedForward(512, 2048)

print(en_emb.shape) # (64, 81, 512)
print(sample_ffn(en_emb).shape) # (64, 81, 512)


# Encoder = Embedding + N * EncoderLayer(GlobalSelfAttention+FeedForward)
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout=0.1):
        super().__init__()
        self.self_attention = GlobalSelfAttention(num_head=num_heads, key_dim=d_model, dropout=dropout)
        self.ffn = FeedForward(d_model, dff)

    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x
    

sample_encoder_layer = EncoderLayer(d_model=512, num_heads=8, dff=2048)

print(pt_emb.shape) # (64, 86, 512)
print(sample_encoder_layer(pt_emb).shape) # (64, 86, 512)

class Encoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size, d_model=d_model)
        self.enc_layers = [EncoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, dropout=dropout) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x):
        # x is token-IDs shape(batch, seq_len)
        x = self.pos_embedding(x) # (batch, seq_len, d_model)
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x)

        return x # (batch_size, seq_len, d_model)
    
# Instantiate the encoder.
sample_encoder = Encoder(num_layers=4,
                         d_model=512,
                         num_heads=8,
                         dff=2048,
                         vocab_size=8500)

sample_encoder_output = sample_encoder(pt, training=False)

# Print the shape.
print(pt.shape) # (64, 86)
print(sample_encoder_output.shape)  # Shape `(batch_size, input_seq_len, d_model)`. (64, 86, 512)


# Decoder = Embedding + N * DecoderLayer(CausalSelfAttention+CrossAttention+FeedForward)
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.causal_self_attention = CausalSelfAttention(num_heads=num_heads, key_dim=d_model, dropout=dropout)
        self.cross_attention = CrossAttention(num_heads=num_heads, key_dim=d_model, dropout=dropout)
        self.ffn = FeedForward(d_model, dff)

    def call(self, x, context):
        x = self.causal_self_attention(x=x)
        x = self.cross_attention(x=x, context=context)
        self.last_attn_scores = self.cross_attention.last_attn_scores
        x = self.ffn(x)
        return x 
    

sample_decoder_layer = DecoderLayer(d_model=512, num_heads=8, dff=2048)

sample_decoder_layer_output = sample_decoder_layer(
    x=en_emb, context=pt_emb)

print(en_emb.shape) # (64, 81, 512)
print(pt_emb.shape) # (64, 86, 512)
print(sample_decoder_layer_output.shape)  # `(batch_size, seq_len, d_model)` (64, 81, 512)

class Decoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size, dropout=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size, d_model=d_model)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.dec_layers = [
            DecoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, dropout=dropout)
            for _ in range(num_layers)
        ]

        self.last_attn_scores = None

    def call(self, x, context):
        # `x` is token-IDs shape (batch, target_seq_len)
        x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, context)

        self.last_attn_scores = self.dec_layers[-1].last_attn_scores

        return x #(batsh_size, target_seq_len, d_model)

# Instantiate the decoder.
sample_decoder = Decoder(num_layers=4,
                         d_model=512,
                         num_heads=8,
                         dff=2048,
                         vocab_size=8000)

output = sample_decoder(
    x=en,
    context=pt_emb)

# Print the shapes.
print(en.shape) # (64, 81)
print(pt_emb.shape) # (64, 86, 512)
print(output.shape) # (64, 81, 512)

sample_decoder.last_attn_scores.shape  # (batch, heads, target_seq, input_seq) TensorShape([64, 8, 81, 86])


# Transfomer
# 注意在原始论文中the weight matrix between the embedding layer and the final linear layer are shared
class Transformer(tf.keras.Model):
    def __init__(self, *, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, dropout=0.1):
        super().__init__()

        self.encoder = Encoder(num_layers=num_layers, d_model=d_model, num_heads=num_heads,
                               dff=dff, vocab_size=input_vocab_size, dropout=dropout)
        self.decoder = Decoder(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff, 
                               ocab_size=target_vocab_size, dropout=dropout)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call (self, inputs):
        # To use a Keras model with `.fit` you must pass all your inputs in the first argument.
        context, x = inputs
        context = self.encoder(context) # (batch_size, context_len, d_model)
        x = self.decoder(x, context) # (batch_size, target_len, d_model)

        logits = self.final_layer(x) # # (batch_size, target_len, target_vocab_size)
        try: # Drop the keras mask, so it doesn't scale the losses/metrics.
            del logits._keras_mask
        except AttributeError:
            pass

        return logits
    

num_layers = 4 # 6
d_model = 128 # 512
dff = 512 # 2048
num_heads = 8
dropout = 0.1


transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=tokenizers.pt.get_vocab_size().numpy(),
    target_vocab_size=tokenizers.en.get_vocab_size().numpy(),
    dropout_rate=dropout)

output = transformer((pt, en))

print(en.shape) # (64, 81)
print(pt.shape) # (64, 86)
print(output.shape) # (64, 81, 7010)

attn_scores = transformer.decoder.dec_layers[-1].last_attn_scores
print(attn_scores.shape)  # (batch, heads, target_seq, input_seq) (64, 8, 81, 86)

transformer.summary() # Total params: 10,184,162

# ====================训练=======================
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.sqrt(step)
        arg2 = step*(self.warmup_steps**-1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
    

learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

plt.plot(learning_rate(tf.range(40000, dtype=tf.float32)))
plt.ylabel('Learning Rate')
plt.xlabel('Train Step')


def masked_loss(label, pred):
    mask = label != 0
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    loss = loss_object(label, pred)
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
    return loss

def masked_accuracy(label, pred):
    pred = tf.argmax(pred, axis=2)
    label = tf.cast(label, dtype=pred.dtype)
    match = label == pred

    mask = label != 0
    match = match & mask

    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(match) / tf.reduce_sum(mask)

transformer.compile(loss=masked_loss, optimizer=optimizer, metrics=[masked_accuracy])
transformer.fit(train_batches, epochs=20, validation_data=val_batches)



# =======================推断=========================
'''
The following steps are used for inference:

Encode the input sentence using the Portuguese tokenizer (tokenizers.pt). This is the encoder input.
The decoder input is initialized to the [START] token.
Calculate the padding masks and the look ahead masks.
The decoder then outputs the predictions by looking at the encoder output and its own output (self-attention).
Concatenate the predicted token to the decoder input and pass it to the decoder.
In this approach, the decoder predicts the next token based on the previous tokens it predicted.
'''

class Translator(tf.Module):
    def __init__(self, tokenizers, transformer):
        self.tokenizers = tokenizers
        self.transformer = transformer

    def __call__(self, sentence, max_length=MAX_TOKENS):
        # The input sentence is Portuguese, hence adding the `[START]` and `[END]` tokens.
        assert isinstance(sentence, tf.Tensor)
        if len(sentence.shape) == 0:
            sentence = sentence[tf.newaxis]

        sentence = self.tokenizers.pt.tokenize(sentence).to_tensor()
        encoder_input = sentence

        # As the output language is English, initialize the output with the English `[START]` token.
        start_end = self.tokenizers.en.tokenize([''])[0]
        start = start_end[0][tf.newaxis]
        end = start_end[1][tf.newaxis]
        
        # `tf.TensorArray` is required here (instead of a Python list), so that the dynamic-loop can be traced by `tf.function`.
        output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        output_array = output_array.write(0, start)

        for i in tf.range(max_length):
            output = tf.transpose(output_array.stack())
            predictions = self.transformer([encoder_input, output], training=False)

            # select the last token from the 'seq_len' dimension
            predictions = predictions[:, -1, :] # (batch_size, 1, vocab_size)
            predicted_id = tf.argmax(predictions, axis=-1)

            # Concatenate the `predicted_id` to the output which is given to the decoder as its input.
            output_array = output_array.write(i+1, predicted_id[0])

            if predicted_id == end:
                break

        output = tf.transpose(output_array.stack())
        # The output shape is `(1, tokens)`.
        text = tokenizers.en.detokenize(output)[0]
        tokens = tokenizers.en.lookup(output)[0]

        # `tf.function` prevents us from using the attention_weights that were
        # calculated on the last iteration of the loop.
        # So, recalculate them outside the loop.
        self.transformer([encoder_input, output[:, :-1]], training=False)
        attention_weights = self.transformer.decoder.last_attn_scores
        return text, tokens, attention_weights
    

translator = Translator(tokenizers, transformer)

def print_translation(sentence, tokens, ground_truth):
    print(f'{"Input:":15s}: {sentence}')
    print(f'{"Prediction":15s}: {tokens.numpy().decode("utf-8")}')
    print(f'{"Ground truth":15s}: {ground_truth}')

sentence = 'este é o primeiro livro que eu fiz.'
ground_truth = "this is the first book i've ever done."

translated_text, translated_tokens, attention_weights = translator(
    tf.constant(sentence))
print_translation(sentence, translated_text, ground_truth)


def plot_attention_head(in_tokens, translated_tokens, attention):
    # The model didn't generate `<START>` in the output. Skip it.
    translated_tokens = translated_tokens[1:]

    ax = plt.gca()
    ax.matshow(attention)
    ax.set_xticks(range(len(in_tokens)))
    ax.set_yticks(range(len(translated_tokens)))

    labels = [label.decode('utf-8') for label in in_tokens.numpy()]
    ax.set_xticklabels(labels, rotation=90)

    labels = [label.decode('utf-8') for label in translated_tokens.numpy()]
    ax.set_yticklabels(labels)

head = 0
# Shape: `(batch=1, num_heads, seq_len_q, seq_len_k)`.
attention_heads = tf.squeeze(attention_weights, 0)
attention = attention_heads[head]
attention.shape # TensorShape([9, 11])


in_tokens = tf.convert_to_tensor([sentence])
in_tokens = tokenizers.pt.tokenize(in_tokens).to_tensor()
in_tokens = tokenizers.pt.lookup(in_tokens)[0]
in_tokens # 分词token
'''
<tf.Tensor: shape=(11,), dtype=string, numpy=
array([b'[START]', b'este', b'e', b'o', b'primeiro', b'livro', b'que',
       b'eu', b'fiz', b'.', b'[END]'], dtype=object)>
'''

translated_tokens
'''
<tf.Tensor: shape=(10,), dtype=string, numpy=
array([b'[START]', b'this', b'is', b'the', b'first', b'book', b'i',
       b'did', b'.', b'[END]'], dtype=object)>
'''

plot_attention_head(in_tokens, translated_tokens, attention)

def plot_attention_weights(sentence, translated_tokens, attention_heads):
    in_tokens = tf.convert_to_tensor([sentence])
    in_tokens = tokenizers.pt.tokenize(in_tokens).to_tensor()
    in_tokens = tokenizers.pt.lookup(in_tokens)[0]

    fig = plt.figure(figsize=(16, 8))

    for h, head in enumerate(attention_heads):
        ax = fig.add_subplot(2, 4, h+1)

        plot_attention_head(in_tokens, translated_tokens, head)

        ax.set_xlabel(f'Head {h+1}')

    plt.tight_layout()
    plt.show()

plot_attention_weights(sentence, translated_tokens, attention_weights[0])


sentence = 'Eu li sobre triceratops na enciclopédia.'
ground_truth = 'I read about triceratops in the encyclopedia.'

translated_text, translated_tokens, attention_weights = translator(tf.constant(sentence))
print_translation(sentence, translated_text, ground_truth)

plot_attention_weights(sentence, translated_tokens, attention_weights[0])

# =========================保存模型==========================
class ExportTranslator(tf.Module):
    def __init__(self, translator):
        self.translator = translator

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def __call__(self, sentence):
        (result, tokens, attention_weights) = self.translator(sentence, max_length=MAX_TOKENS)

        return result
    

translator = ExportTranslator(translator)
translator('este é o primeiro livro que eu fiz.').numpy()

tf.saved_model.save(translator, export_dir='translator')
reloaded = tf.saved_model.load('translator')
reloaded('este é o primeiro livro que eu fiz.').numpy()