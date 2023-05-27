# https://tensorflow.google.cn/text/tutorials/text_classification_rnn
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt


def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel('Epochs')
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])


dataset, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']
# train_dataset.element_spec

for example, label in train_dataset.take(1):
    print('text:', example.numpy())
    print('label:', label.numpy())

'''
text:  b"This was an absolutely terrible movie. Don't be lured in by Christopher Walken or Michael Ironside. Both are great actors, but this must simply be their worst role in history. Even their great acting could not redeem this movie's ridiculous storyline. This movie is an early nineties US propaganda piece. The most pathetic scenes were those when the Columbian rebels were making their cases for revolutions. Maria Conchita Alonso appeared phony, and her pseudo-love affair with Walken was nothing but a pathetic emotional plug in a movie that was devoid of any real meaning. I am disappointed that there are movies like this, ruining actor's like Christopher Walken's good name. I could barely sit through it."
label:  0
'''

BUFFERSIZE = 10000
BATCHSIZE = 64

train_dataset = train_dataset.shuffle(BUFFERSIZE).batch(BATCHSIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCHSIZE).prefetch(tf.data.AUTOTUNE)

for example, label in train_dataset.take(1):
    print('texts:', example.numpy()[:3])
    print('labels:', label.numpy()[:3])

'''
texts:  
[b"As a huge fan of horror films, especially J-horror and also gore i thought Nekeddo bur\xc3\xa2ddo sounded pretty good. I researched the plot, read reviews, and even looked at some photos to make sure it seemed like a good gory and scary movie to watch before downloading it. So excited it had finished and ready to be scared and recoiling in horror at the amazing gore i was expecting i was terribly disappointed. The plot was ridiculous and didn't even make sense and left too much unexplained, the gore was hilarious rather then horrifying, and what was with the cartoon style sound effects ? The acting was probably the only thing mildly scary about it. I did not understand the cactus idea and the way the mothers husband disappeared in the middle of the sea after following a flashing light, they left both pretty unexplained, or perhaps i missed it as my mind couldn't understand what i was actually seeing. I appreciate the way it was supposed to be; shocking and a few scenes (the strange cannibalism and own mother kissing?)certainly were, i just think they went a little bit far and not even in a horrifying way, they made it to unconvincing which made it more believable to be a comedy rather than a horror in my opinion. However it is a very entertaining film and got a lot of laughs out of me and a couple of friends, but sadly we were expecting horror not comedy so its worth a watch for the entertainment value, but don't be expecting a dark, deeply scary and horrifying film; you'll just be disappointed. If this was a horror comedy/spoof i'd probably rate it about a nine, the climax being the weird scene when the husband climbed inside his wife's stomach and closed up her wounds, but as a horror sadly i gave it a one."
 b'"What happens when you give a homeless man \\(100,000?" As if by asking that question they are somehow morally absolved of what is eventually going to happen. The creators of "Reversal of Fortune" try to get their voyeuristic giggles while disguising their antics as some kind of responsible social experiment.<br /><br />They take Ted, a homeless man in Pasadena, and give him \\)100,000 to see if he will turn his life around. Then, with only the most cursory guidance and counseling, they let him go on his merry way.<br /><br />What are they trying to say? "Money can\'t buy you happiness?" "The homeless are homeless because they deserve to be?" Or how about, "Lift a man up - it\'s more fun to watch him fall from a greater altitude." They took a man with nothing to lose, gave him something to lose, and then watched him dump it all down the drain. That\'s supposed to be entertainment? They dress this sow up with some gloomy music and dramatic camera shots, but in the end it has all the moral high ground of car crash videos - only this time they engineered the car crashes and asked, "What happens when you take down a stop sign?"'
 b"'Mojo' is a story of fifties London, a world of budding rock stars, violence and forced homosexuality. 'Mojo' uses a technique for shooting the 1950s often seen in films that stresses the physical differences to our own time but also represents dialogue in a highly exaggerated fashion (owing much to the way that speech was represented in films made in that period); I have no idea if people actually spoke like this outside of the movies, but no films made today and set in contemporary times use such stylised language. It's as if the stilted discourse of 1950s screenwriters serves a common shorthand for a past that seems, in consequence, a very distant country indeed; and therefore stresses the particular, rather than the universal, in the story. 'Mojo' features a strong performance from Ian Hart and annoying ones from Aiden Gillan and Ewan Bremner, the latter still struggling to build a post-'Trainspotting' career; but feels like a period piece, a modern film incomprehensibly structured in an outdated idiom. Rather dull, actually."]

labels:  [0 0 0]
'''

VOCAB_SIZE = 1000
encoder = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE)
encoder.adapt(train_dataset.map(lambda text, label:text))

vocab = np.array(encoder.get_vocabulary())
vocab[:20] # After the padding and unknown tokens they're sorted by frequency
'''
array(['', '[UNK]', 'the', 'and', 'a', 'of', 'to', 'is', 'in', 'it', 'i',
       'this', 'that', 'br', 'was', 'as', 'for', 'with', 'movie', 'but'],
      dtype='<U14')
'''
# The tensors of indices are 0-padded to the `longest sequence in the batch`` (unless you set a fixed `output_sequence_length`)
encoded_example = encoder(example)[:3].numpy()
encoded_example

# Pad:0 [UNK]:1
'''
array([[ 15,   4, 629, ...,   0,   0,   0],
       [ 49, 557,  51, ...,   0,   0,   0],
       [  1,   7,   4, ...,   0,   0,   0]])
'''

'''
With the default settings, the process is not completely reversible. There are reasons for that:

The default value for preprocessing.TextVectorization's standardize argument is "lower_and_strip_punctuation".
The limited vocabulary size and lack of character-based fallback results in some unknown tokens.

'''
for n in range(1):
    print("Original: ", example[n].numpy())
    print("Round-trip: ", " ".join(vocab[encoded_example[n]]))
    print()

'''
Original:  b"As a huge fan of horror films, especially J-horror and also gore i thought Nekeddo bur\xc3\xa2ddo sounded pretty good. I researched the plot, read reviews, and even looked at some photos to make sure it seemed like a good gory and scary movie to watch before downloading it. So excited it had finished and ready to be scared and recoiling in horror at the amazing gore i was expecting i was terribly disappointed. The plot was ridiculous and didn't even make sense and left too much unexplained, the gore was hilarious rather then horrifying, and what was with the cartoon style sound effects ? The acting was probably the only thing mildly scary about it. I did not understand the cactus idea and the way the mothers husband disappeared in the middle of the sea after following a flashing light, they left both pretty unexplained, or perhaps i missed it as my mind couldn't understand what i was actually seeing. I appreciate the way it was supposed to be; shocking and a few scenes (the strange cannibalism and own mother kissing?)certainly were, i just think they went a little bit far and not even in a horrifying way, they made it to unconvincing which made it more believable to be a comedy rather than a horror in my opinion. However it is a very entertaining film and got a lot of laughs out of me and a couple of friends, but sadly we were expecting horror not comedy so its worth a watch for the entertainment value, but don't be expecting a dark, deeply scary and horrifying film; you'll just be disappointed. If this was a horror comedy/spoof i'd probably rate it about a nine, the climax being the weird scene when the husband climbed inside his wife's stomach and closed up her wounds, but as a horror sadly i gave it a one."
Round-trip:  as a huge fan of horror films especially [UNK] and also gore i thought [UNK] [UNK] [UNK] pretty good i [UNK] the plot read reviews and even looked at some [UNK] to make sure it seemed like a good [UNK] and scary movie to watch before [UNK] it so [UNK] it had [UNK] and [UNK] to be [UNK] and [UNK] in horror at the amazing gore i was expecting i was [UNK] disappointed the plot was ridiculous and didnt even make sense and left too much [UNK] the gore was hilarious rather then [UNK] and what was with the [UNK] style sound effects the acting was probably the only thing [UNK] scary about it i did not understand the [UNK] idea and the way the [UNK] husband [UNK] in the middle of the [UNK] after [UNK] a [UNK] light they left both pretty [UNK] or perhaps i [UNK] it as my mind couldnt understand what i was actually seeing i [UNK] the way it was supposed to be [UNK] and a few scenes the strange [UNK] and own mother [UNK] were i just think they went a little bit far and not even in a [UNK] way they made it to [UNK] which made it more believable to be a comedy rather than a horror in my opinion however it is a very entertaining film and got a lot of laughs out of me and a couple of friends but [UNK] we were expecting horror not comedy so its worth a watch for the entertainment [UNK] but dont be expecting a dark [UNK] scary and [UNK] film youll just be disappointed if this was a horror [UNK] id probably [UNK] it about a [UNK] the [UNK] being the weird scene when the husband [UNK] inside his [UNK] [UNK] and [UNK] up her [UNK] but as a horror [UNK] i gave it a one                                                                                                                                                                                                                                                                                     
'''

model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(input_dim=len(encoder.get_vocabulary()),
                              output_dim=64,
                              mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

# The embedding layer uses masking to handle the varying sequence-lengths. All the layers after the Embedding support masking
print([layer.supports_masking for layer in model.layers]) # [False, True, True, True, True]

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4), metrics=['accuracy'])
history = model.fit(train_dataset, epochs=10, validation_data=test_dataset, validation_steps=30)


test_loss, test_acc = model.evaluate(test_dataset)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plot_graphs(history, 'accuracy')
plt.ylim(None, 1)
plt.subplot(1, 2, 2)
plot_graphs(history, 'loss')
plt.ylim(0, None)

sample_text = ('The movie was cool. The animation and the graphics '
               'were out of this world. I would recommend this movie.')
predictions = model.predict(np.array([sample_text]))



model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(len(encoder.get_vocabulary()), 64, mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1)
])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

history = model.fit(train_dataset, epochs=10,
                    validation_data=test_dataset,
                    validation_steps=30)

test_loss, test_acc = model.evaluate(test_dataset)

print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)

# predict on a sample text without padding.

sample_text = ('The movie was not good. The animation and the graphics '
               'were terrible. I would not recommend this movie.')
predictions = model.predict(np.array([sample_text]))
print(predictions)

plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plot_graphs(history, 'accuracy')
plt.subplot(1, 2, 2)
plot_graphs(history, 'loss')