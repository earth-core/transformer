#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 14:38:07 2024

@author: machine
"""
from tensorflow.keras.layers import TextVectorization
import numpy as np 
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
import pathlib
import unicodedata
import random
import re
import sys
import json


class PositionalEmbedding(tf.keras.layers.Layer):
    """Postional Embedding layer. Assume tokenized input, transform into 
    embedding and returns positinal encoded output"""
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        '''
        
        Parameters
        ----------
        sequence_length : input sequence length 
        
        vocab_size :  No of unique tokens,vocabulary size
            
        embed_dim :  embedding vector size\
            
        '''
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        #Token Embedding Layer: Convert integer token to embed_dim-dimension 
        #floating vector
        self.token_embeddings = tf.keras.layers.Embedding(
            input_dim=vocab_size+1, output_dim=embed_dim,
                                                          mask_zero=True)
        #Positional Encoding Layer
        matrix = pos_enc_matrix(L=sequence_length, d=embed_dim)
        self.position_embeddings = tf.constant(matrix, dtype="float32")
        
    def call(self, inputs):
        """Input tokens - converted to embedding and super-imposed with
        positional encoding"""
        embedded_tokens = self.token_embeddings(inputs)
        return embedded_tokens + self.position_embeddings
    
    def compute_mask(self, *args, **kwargs):
        return self.token_embeddings.compute_mask(*args, **kwargs)
    
    def get_config(self):
        #To save and load model which uses custom layers 
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim 
            })
        return config

def pos_enc_matrix(L, d, n=10):
    ''' create positional encoding matrix 
     args :
           L: Input Dimension(length)
           d: Output Dimension(depth), even only 
           n: Constant for sinusoidal functions

    returns:
            numpy matrix of floats of dimension L-by-d. At element (k,2i)
            the value is sin(k/n^(2i/d)) while at element (k,2i+1) the 
            value is cos(k/n^(2i/d))
    '''
    assert d % 2 == 0 
    d2 = d//2
    P = np.zeros((L, d))
    k = np.arange(L).reshape(-1, 1)     # L-by-1  column vector
    i = np.arange(d2).reshape(1, -1)    # 1-by-d2 row vector
    denom = np.power(n, -i/d2)          # n**(-2*i/d)
    args = k * denom                    # (L,d2) matrix
    P[:, ::2] = np.sin(args)
    P[:, 1::2] = np.cos(args)
    return P

def self_attention(input_shape, num_heads, model_dim, prefix="self-att",
                   mask=False,
                   **kwargs):
    """
    Self-Attention layer of encoder and decoder, takes input from 
    PositionalEmbedding  

    Parameters
    ----------
    input_shape : 
    prefix : (string)Prefix added to layer names
        
    mask : whether to use causal mask false on encoder and true on decoder. 
    when true, a mask will be applied such that each location only has access 
    to positions before it.
    The default is False.
        
        Returns
        -------
        Model
    """
    inputs = tf.keras.layers.Input(shape=input_shape, dtype="float32", 
                                   name=f"{prefix}_in1")
    attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads,
                                                   key_dim=model_dim,
                                                   name=f"{prefix}_attn1",
                                                   **kwargs)
    norm = tf.keras.layers.LayerNormalization(name=f"{prefix}_norm1")
    add = tf.keras.layers.Add(name=f"{prefix}_add1")
    
    attout = attention(query=inputs, value=inputs, key=inputs,
                       use_causal_mask=mask)
    outputs = norm(add([inputs, attout]))
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, 
                           name=f"{prefix}_att")
    
    return model

def cross_attention(input_shape, context_shape, num_heads, model_dim,
                    prefix="cross-att", **kwargs):
    """
    Cross-Attention Layers at transformer decoder.Assumes it's input is the 
    output from the positional encoding layer at decoder and context is final 
    output from encoder.

    Parameters
    ----------
    input_shape : Input Shape
    
    context_shape : Context Shape
    
    prefix :(String) Prefix added to layer names. 
    
    Returns
    -------
    Model

    """
    context = tf.keras.layers.Input(shape=context_shape, dtype="float32",
                                    name=f"{prefix}_cntxt")#ctxt2
    inputs = tf.keras.layers.Input(shape=input_shape, dtype="float32", 
                                   name=f"{prefix}_in")
    attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, 
                                                   key_dim=model_dim,
                                                   name=f"{prefix}_MultiAttn")
    norm = tf.keras.layers.LayerNormalization(name=f"{prefix}_norm")
    add = tf.keras.layers.Add(name=f"{prefix}_add")
    
    attout = attention(query=inputs, value=context, key=context)
    outputs = norm(add([attout, inputs]))
    
    model = tf.keras.Model(inputs=[inputs, context], outputs=outputs, 
                           name=f"{prefix}")
    
    return model

def Feed_Forward(input_shape, model_dim, FF_dim, dropout=0.1, prefix="FF"):
    """
    Feed-Forward layers at transformer encoder and decoder. Assumes it's input
    is the input from attention layer with add & norm. The output is output of 
    encoder or decoder block.

    Parameters
    ----------
    input_shape : 
        
    model_dim : (int) Output dimension of feed-forward layer, which is also 
    output of encoder/decoder block.
    
    FF_dim : (int) internal dimension of feed-forward layer 
    
    dropout : The default is 0.1.
    
    prefix : The default is "FF".

    Returns
    -------
    Model

    """
    inputs = tf.keras.layers.Input(shape=input_shape, name=f"{prefix}_in", 
                                   dtype="float32")
    dense1 = tf.keras.layers.Dense(FF_dim, name=f"{prefix}_FF1", 
                                   activation="relu")
    dense2 = tf.keras.layers.Dense(model_dim, name = f"{prefix}_FF2")
    drop = tf.keras.layers.Dropout(dropout, name=f"{prefix}_drop")
    add = tf.keras.layers.Add(name=f"{prefix}_add")
    norm = tf.keras.layers.LayerNormalization(name=f"{prefix}_norm")
    
    FFout = drop(dense2(dense1(inputs)))
    outputs = norm(add([inputs, FFout]))
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"{prefix}")
    
    return model

def encoder(input_shape, model_dim, FF_dim, num_heads, dropout=0.1,
            prefix="Enc", **kwargs):
    """
    One Encoder unit. The input and output are in the same shape so we can 
    chain multiple encoder unit to a longer/wider encoder    

    Parameters
    ----------
    input_shape : 
        
    model_dim : Model output dimension same as model input
        
    FF_dim : 
        
    dropout : The default is 0.1.
    
    prefix : The default is "Enc".
  

    Returns
    -------
    Model
    
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=input_shape, dtype='float32', 
                              name=f"{prefix}_in0"),
        self_attention(input_shape, prefix=prefix, num_heads=num_heads,
                       model_dim=model_dim,
                       mask=False, **kwargs), 
        Feed_Forward(input_shape, model_dim, FF_dim, dropout)],
     name=prefix)
    return model

def decoder(input_shape, model_dim, FF_dim, num_heads, dropout=0.1,
            prefix="Dec", **kwargs):
    """
    One Decoder unit. Inputs and outputs are of same dimension so we can chain 
    multiple to make one large decoder. The Context vector is also assumed to 
    be of same shape.
    

    Parameters
    ----------
    input_shape : 
        
    model_dim : Model output dimension same as model input 
        
    FF_dim : 
        
    dropout :  The default is 0.1.
    
    prefix :  The default is "Dec".
    

    Returns
    -------
    None.

    """
    inputs = tf.keras.layers.Input(shape=input_shape, dtype="float32", 
                                    name=f"{prefix}_in")
    context = tf.keras.layers.Input(shape=input_shape, dtype="float32", 
                                     name=f"{prefix}_cntxt")
    crossmodel = cross_attention(input_shape, input_shape, num_heads=num_heads,
                                 model_dim=model_dim,prefix=prefix, **kwargs)
    ff_model = Feed_Forward(input_shape, model_dim, FF_dim, dropout, 
                            prefix=f"{prefix}_FF")
    ATTmodel = self_attention(input_shape=input_shape, num_heads=num_heads,
                              model_dim=model_dim,
                              prefix=f"{prefix}_S-ATT", mask=True)
    x = ATTmodel(inputs)
    x = crossmodel([x, context])
    output = ff_model(x)
    model = tf.keras.Model(inputs=[inputs, context], outputs=output,  
                           name=prefix )
    return model

def transformer(num_layers, num_heads, seq_len, model_dim, FF_dim, 
                vocab_size_src, vocab_size_target, dropout=0.1, 
                name="transformer"):
    """
    Transformer Model 

    Parameters
    ----------
    num_layers : TYPE
        DESCRIPTION.
    num_heads : TYPE
        DESCRIPTION.
    seq_len : TYPE
        DESCRIPTION.
    model_dim : TYPE
        DESCRIPTION.
    ff_dim : TYPE
        DESCRIPTION.
    vocab_size_src : TYPE
        DESCRIPTION.
    vocab_size_target : TYPE
        DESCRIPTION.
    dropout : TYPE, optional
        DESCRIPTION. The default is 0.1.
    name : TYPE, optional
        DESCRIPTION. The default is "transformer".
        
        

    Returns
    -------
    None.

    """
    embed_shape = (seq_len, model_dim)
    input_enc = tf.keras.layers.Input(shape=(seq_len,), dtype="int32", 
                                      name="Encoder_Inputs")
    input_dec = tf.keras.layers.Input(shape=(seq_len,), dtype="int32",
                                      name="Decoder_Inputs")
    embed_enc = PositionalEmbedding(sequence_length=seq_len, 
                                    vocab_size=vocab_size_src
                                    , embed_dim=model_dim, name="Embed_Enc")
    embed_dec = PositionalEmbedding(sequence_length=seq_len, 
                                    vocab_size=vocab_size_target,
                                    embed_dim=model_dim, name="Embed_Dec")
    encoders = [encoder(input_shape=embed_shape, model_dim=model_dim,
                        FF_dim=FF_dim, num_heads=num_heads, prefix=f"Enc{i}") 
                for i in range(num_layers)]
    decoders = [decoder(input_shape=embed_shape, model_dim=model_dim,
                        FF_dim=FF_dim, num_heads=num_heads, prefix=f"Deco{i}") 
                 for i in range(num_layers)]
    final = tf.keras.layers.Dense(vocab_size_target, name='Linear')
    
    x1 = embed_enc(input_enc)
    x2 = embed_dec(input_dec)
    for layer in encoders:
        x1 = layer(x1)
    for layer in decoders:
        x2 = layer([x2,x1])
    output = final(x2)
    
    try:
        del output._keras_mask
    except AttributeError:
        pass
    
    model = tf.keras.Model(inputs=[input_enc, input_dec], outputs=output,
                           name=name)
    return model
    
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    "Custom learning rate for Adam optimizer"
    def __init__(self, key_dim, warmup_steps=4000):
        super().__init__()
        self.key_dim = key_dim
        self.warmup_steps = warmup_steps
        self.d = tf.cast(self.key_dim, tf.float32)
 
    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d) * tf.math.minimum(arg1, arg2)
 
    def get_config(self):
        # to make save and load a model using custom layer possible0
        config = {
            "key_dim": self.key_dim,
            "warmup_steps": self.warmup_steps,
        }
        return config
    
            
def masked_loss(label, pred):
    mask = label != 0
 
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    loss = loss_object(label, pred)
 
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
    return loss
 
 
def masked_accuracy(label, pred):
    pred = tf.argmax(pred, axis=2)
    label = tf.cast(label, pred.dtype)
    match = label == pred
 
    mask = label != 0
 
    match = match & mask
 
    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(match)/tf.reduce_sum(mask)

def normalize(line):
    line = unicodedata.normalize("NFKC", line.strip().lower())
    line = re.sub(r"^([^ \w])(?!\s)",r"\1",line)
    line = re.sub(r"(\s[^ \w])(?!\s)",r"\1",line)
    line = re.sub(r"(?!\s)([^ \w])$",r"\1",line)
    line = re.sub(r"(?!\s)([^ \w]\s)",r"\1",line)
    src , target = line.split("\t")
    target ="[start]" + target + "[end]"
    return src,target


def format_dataset(src, target):
    src = src_vectorizer(src)
    target = target_vectorizer(target)
    source = {"Encoder_Inputs": src,
              "Decoder_Inputs": target[:, :-1]}
    target = target[:, 1:]
    return (source,target)

def make_dataset(pairs, batch_size=64):
    """Create TensorFlow Dataset for the sentence pairs"""
    # aggregate sentences using zip(*pairs)
    src_texts, target_texts = zip(*pairs)
    # convert them into list, and then create tensors
    dataset = tf.data.Dataset.from_tensor_slices((list(src_texts), list(target_texts)))
    return dataset.shuffle(2048) \
                  .batch(batch_size).map(format_dataset) \
                  .prefetch(16).cache()

file_path = sys.argv[1]
text_file = pathlib.Path(file_path)
textSrc = sys.argv[2]
textTarget = sys.argv[3]

with open(text_file) as fp:
    text_pairs = [normalize(line) for line in fp]

with open(f"{textSrc}-{textTarget}Pairs.pickle", "wb") as fp:
    pickle.dump(text_pairs,fp)

with open(f"{textSrc}-{textTarget}Pairs.pickle","rb") as fp:
    text_pairs = pickle.load(fp)


src_tokens, target_tokens = set(), set()
src_maxlen, target_maxlen = 0, 0

for src,target in text_pairs:
    src_tok, target_tok = src.split(), target.split()
    src_maxlen = max(src_maxlen, len(src_tok))
    target_maxlen = max(target_maxlen, len(target_tok))
    src_tokens.update(src_tok)
    target_tokens.update(target_tok)

print(f"Total {textSrc} tokens: {len(src_tokens)}")
print(f"Total {textTarget} tokens: {len(target_tokens)}")
print(f"Max {textSrc} length: {src_maxlen}")
print(f"Max {textTarget} length: {target_maxlen}")
print(f"{len(text_pairs)} - total pairs")



'''
EDITT


# histogram of sentence length in tokens
en_lengths = [len(eng.split()) for eng, fra in text_pairs]
fr_lengths = [len(fra.split()) for eng, fra in text_pairs]
 
plt.hist(en_lengths, label="en", color="red", alpha=0.33)
plt.hist(fr_lengths, label="fr", color="blue", alpha=0.33)
plt.yscale("log")     # sentence length fits Benford"s law
plt.ylim(plt.ylim())  # make y-axis consistent for both plots
plt.plot([max(en_lengths), max(en_lengths)], plt.ylim(), color="red")
plt.plot([max(fr_lengths), max(fr_lengths)], plt.ylim(), color="blue")
plt.legend()
plt.title("Examples count vs Token length")
plt.show()
'''

vocab_size_src = 22000
vocab_size_target = 42000
seq_length = 36

src_vectorizer = TextVectorization(
    max_tokens = vocab_size_src,
    standardize=None,
    split="whitespace",
    output_mode="int",
    output_sequence_length=seq_length)
target_vectorizer = TextVectorization(
    max_tokens = vocab_size_target,
    split="whitespace",
    output_mode="int",
    output_sequence_length=seq_length+1) #outputsequencelen =+1

#Splitting Data 
#train-test-validation split
random.shuffle(text_pairs)
n_val = int(0.15*len(text_pairs))
n_train = len(text_pairs) - 2*n_val
train_pairs = text_pairs[:n_train]
val_pairs = text_pairs[n_train:n_train+n_val]
test_pairs = text_pairs[n_train+n_val:]


# train the vectorization layer using traning dataset
train_src_texts = [pair[0] for pair in train_pairs]
train_target_texts = [pair[1] for pair in train_pairs]
src_vectorizer.adapt(train_src_texts)
target_vectorizer.adapt(train_target_texts)


train_ds = make_dataset(train_pairs)
val_ds = make_dataset(val_pairs)
test_ds = make_dataset(test_pairs)

seq_len = seq_length
num_layers = 4
num_heads = 8
model_dim = 128
FF_dim = 512
dropout = 0.1

model = transformer(num_layers, num_heads, seq_len, model_dim, FF_dim,
                    vocab_size_src, vocab_size_target, dropout)
lr = CustomSchedule(model_dim)
optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
model.compile(loss=masked_loss, optimizer=optimizer, metrics=[masked_accuracy])
model.summary(show_trainable=True)
tf.keras.utils.plot_model(model=model,to_file='Transformer_Model.png', 
                          show_shapes=True, show_dtype=True, 
                          show_layer_activations=True)

my_callbacks = [tf.keras.callbacks.ModelCheckpoint(
    filepath='callback/ModelCheckpoint',
    save_weights_only=False,verbose=1),
                tf.keras.callbacks.BackupAndRestore(
                    backup_dir='callback/BackupandRestore',save_freq="epoch"),
                tf.keras.callbacks.TensorBoard(log_dir='callback/logs'),
                tf.keras.callbacks.CSVLogger(
                    filename='callback/CSVLogger/logger.csv'),
                tf.keras.callbacks.EarlyStopping(monitor="loss")]

epochs = 20
history = model.fit(train_ds, epochs=epochs, validation_data=val_ds,
                    callbacks=my_callbacks)
model.save_weights(f'weights/{textSrc}_{textTarget}_model_weights')
history_dict = history.history
json.dump(history_dict,open(f'history_model_{textSrc}-{textTarget}','w'))
model.save(f"{textSrc}-{textTarget}-transformer1.keras")
print("Model Saved")

# Plot the loss and accuracy history
fig, axs = plt.subplots(2, figsize=(6, 8), sharex=True)
fig.suptitle('Traininig history')
x = list(range(1, epochs+1))
axs[0].plot(x, history.history["loss"], alpha=0.5, label="loss")
axs[0].plot(x, history.history["val_loss"], alpha=0.5, label="val_loss")
axs[0].set_ylabel("Loss")
axs[0].legend(loc="upper right")
axs[1].plot(x, history.history["masked_accuracy"], alpha=0.5, label="acc")
axs[1].plot(x,history.history["val_masked_accuracy"],alpha=0.5,label="val_acc")
axs[1].set_ylabel("Accuracy")
axs[1].set_xlabel("epoch")
axs[1].legend(loc="lower right")
plt.show()





