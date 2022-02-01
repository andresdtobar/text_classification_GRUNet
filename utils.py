import numpy as np
import pandas as pd
import spacy
from spellchecker import SpellChecker
import json
import hunspell
import wordninja
import re

# Keras for Recurrent Network model
from keras.layers import Dense, Input, Lambda, Dropout, Activation, GRU, Bidirectional
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.initializers import Constant
import keras
import tensorflow as tf

import matplotlib.pyplot as plt
import itertools
from collections import Counter

from sklearn.model_selection import train_test_split

import config

wordninja.DEFAULT_LANGUAGE_MODEL = wordninja.LanguageModel('dict/my_lang.txt.gz')
nlp = spacy.load('es_core_news_md')


def train_model(text, y):

    n_classes = 2

    #Identify vocabulary
    tokens = {word for sentence in text for word in sentence.split(' ')} - {''}

    # Update vocabulary with new words found in previuos excecutions 
    new_words = np.loadtxt('dict/new_words.txt', dtype=str, usecols=0)
    vocab = sorted(set(list(tokens) + new_words.tolist()))

    # Save updated vocabulary
    np.savetxt('dict/vocab.txt', vocab, fmt='%s', delimiter=',')

    # Clean new_words file
    file = open('dict/new_words.txt', 'w')
    file.close()

    # Build the embedding matrix
    embedding_matrix = np.array([nlp(word).vector for word in vocab])
    num_tokens, embedding_dim = embedding_matrix.shape

    X_indices = sentences_to_indices(text, vocab, config._MAX_TEXT_LENGHT)

    classes = dict(Counter(y))
    pos = classes[1]
    neg = classes[0]

    model = TextClassification(config._MAX_TEXT_LENGHT, embedding_matrix, n_classes, num_tokens, embedding_dim)

    metrics = [
            'accuracy', 'Recall', 'Precision',
            keras.metrics.AUC(name='auc'),
            keras.metrics.AUC(name='prc', curve='PR')
            ]

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=metrics)
    total = len(y)
    weight_for_0 = (1 / neg) * (total / 2.0)
    weight_for_1 = (1 / pos) * (total / 2.0)

    class_weight = {0: weight_for_0, 1: weight_for_1}

    #Train model
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_recall', 
        patience=5,
        mode='max',
        restore_best_weights=True
        )

    # Split dataset into train and val sets 
    X_train_indices, X_val_indices, y_train, y_val = train_test_split(X_indices, y, test_size=.1, random_state=42)

    model.fit(X_train_indices, y_train, batch_size=128, epochs=100, validation_data = (X_val_indices, y_val), 
        callbacks = [early_stop], class_weight=class_weight)

    model.save('models/bigru_model_last.h5')

def predict_text(text_test):

    # Read vocabulary 
    vocab = np.loadtxt('dict/vocab.txt', dtype=str, usecols=0)
    X_test_indices = sentences_to_indices(text_test, vocab, config._MAX_TEXT_LENGHT)
    
    model = keras.models.load_model('models/bigru_model_last.h5')

    rg = [.3, .7]
    h_test = 2*np.ones((len(text_test),1))
    pos = model.predict(X_test_indices) >= rg[1]
    h_test[pos] = 1
    neg = model.predict(X_test_indices) <= rg[0]
    h_test[neg] = 0
    # print(f"in-confiden-interval labeled samples: {(len(h_test[pos]) + len(h_test[neg]))/len(text_test)}")
    # print(f"Not labeled samples: {-len(h_test[pos]) - len(h_test[neg]) + len(text_test)}")
    return h_test

def process_text(df, print_results=False):
    # Erase rows with null values
    df = df.loc[~df['ObjetoProceso'].isnull()]

    # Erase rows where text length is less than the MINUMUM_LENGHT_TEXT
    df = df.loc[df['ObjetoProceso'].apply(len) > config._MIN_TEXT_LENGHT]

    # Separate label column and make lower case all text
    text = df['ObjetoProceso']
    text = np.array([text.lower() for text in text])

    # Spell checker and in-context correction
    fixed_text = correct_words(text, print_results)
    y = df['Clasificacion'].astype(int)

    return fixed_text, y

def process_text1(df, print_results=False):
    # Erase rows with null values
    df = df.loc[~df['DetalleObjetoAContratar'].isnull()]

    # Erase rows where text length is less than the MINUMUM_LENGHT_TEXT
    df = df.loc[df['DetalleObjetoAContratar'].apply(len) > config._MIN_TEXT_LENGHT]

    # Separate label column and make lower case all text
    text = df['DetalleObjetoAContratar']
    text = np.array([text.lower() for text in text])

    # Spell checker and in-context correction
    fixed_text = correct_words(text, print_results)
    y = df['Clasificacion'].astype(int)

    return fixed_text, y

def clean_text(doc_text):
    # Remove symbols
    for ch in '!"-#$%&()--.*+,-/:;<=>?@[\\]^_`{|}~\t\n':
        doc_text = doc_text.replace(ch, '')
    
    #Remove numbers
    doc_text = re.sub(r'[^\D]', '', doc_text)
    return ' '+doc_text+' '

def average_word_embeddings(text):
    X = []
    for row in text:
        doc = nlp(str(row))
        word_mean = np.zeros((doc[0].vector.size))
        for word in doc:
            word_mean += word.vector/len(doc)
        X.append(word_mean)
    return np.array(X)


def sentence_embeddings(text, max_len, embedding_dim):
    m = len(text)
    X = np.zeros((m,max_len,embedding_dim), dtype='float32')

    for i, row in enumerate(text):
        doc = nlp(str(row))
        if len(doc) > max_len:
            doc = doc[:max_len]
        for j, word in enumerate(doc):
            X[i,j,:] = word.vector
    return X

def sentence_sparse_embeddings(text, max_len, embedding_dim):
    m = len(text)
    dense_shape = (m,max_len,embedding_dim)
    indices = [[0,0,0]]
    values = [0.0]    
    st_x = tf.SparseTensor(indices, values, dense_shape)

    inds = np.zeros((embedding_dim, 3), dtype='int32')
    inds[:,2] = list(range(embedding_dim))
    for i, row in enumerate(text):
        doc = nlp(str(row))        
        inds[:,0] = i
        values, indices = [], []
        for j, word in enumerate(doc):            
            inds[:,1] = j            
            indices += inds.tolist()
            values += list(word.vector.astype('float32'))
        st_a = tf.SparseTensor(indices, values, dense_shape)
        st_x = tf.sparse.add(st_x, st_a)
        print(f'{i} of {m} docs')

    return st_x

def sentences_to_indices(X_train, vocab, max_len):
    word_to_index = dict()
    for index, word in enumerate(vocab):
        word_to_index[word] = index 
    
    #return [[word_to_index[word] for word in text.split(' ') if word != ''] for text in X_train]

    m = len(X_train)
    X_indices = np.zeros((m, max_len))
    unknown_words = []
    for i in range(m):                               # loop over examples
        
        # Convert the ith training sentence in lower case and split is into words. You should get a list of words.
        sentence_words = X_train[i].lower().split(' ')
        if len(sentence_words) > max_len:
            sentence_words = sentence_words[:max_len]
        
        j = 0        
        for w in sentence_words:
            if w != '':
                if w in set(word_to_index.keys()):
                    X_indices[i, j] = word_to_index[w]
                    j = j + 1
                else:
                    #print(f'La palabra "{w}" no está en el vocabulario y no se podrá generar su representación vectorial')
                    unknown_words.append(w)
    
    np.savetxt('dict/new_words.txt', unknown_words, fmt='%s', delimiter=',')
    return X_indices

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y

def correct_words_old(text):
    #Load instance of SpellChecker class
    spell = SpellChecker(language=None, case_sensitive=False, distance=1)
    
    # Load in-context dictionary 
    spell.word_frequency.load_dictionary('dict/custom_dict.json')
    
    # Correct missspelled words
    text_all_corrected = []
    for x in text:
        x = x.lower()
        x = clean_text(str(x))
        misspelled = list(spell.unknown(x.split(' ')))
        for word in misspelled:
            if word != '':            
                x = x.replace(word, spell.correction(word))
    
        text_all_corrected.append(x)
    
    return text_all_corrected

def correct_words(text, verbose=False):
    #Load instance of SpellChecker class
    spell = SpellChecker(language=None, case_sensitive=False, distance=2)
    external_dict = hunspell.HunSpell('dict/es_CO.dic', 'dict/es_CO.aff')

    # Load in-context dictionary 
    spell.word_frequency.load_dictionary('dict/custom_dict.json')    

    # Correct missspelled words
    text_all_corrected = []
    for x in text:
        x = x.lower()
        x = clean_text(str(x))
        misspelled = list(spell.unknown(x.split(' ')))
        for word in misspelled:
            if word != '':                
                known_word = external_dict.spell(word)       
                if known_word:                    
                    spell.word_frequency.add(word)
                    if verbose:
                        print(f'Palabra añadida: {word}')
                else:
                    #Here should be the option to te user add manually the word if he consider it
                    sep_word = wordninja.split(word)
                    if len(sep_word) == 2 and len(word) > 4:
                        if len(sep_word[1]) != 1 and word != 'demas':
                            x = x.replace(word, ' '.join(sep_word))
                            if verbose:
                                print(f'Palabras separadas: {word}: {" ".join(sep_word)}')
                    else:
                        new_word = spell.correction(word)
                        x = x.replace(' '+word+' ', ' '+new_word+' ')
                        if verbose:
                            if external_dict.spell(new_word) or (spell.known([new_word]) == {new_word}):
                                print(f'Palabra corregida: {word}: {new_word}')
                            else:
                                print(f'Palabra desconocida: {word}')
                    
    
        text_all_corrected.append(x)
    spell.export('dict/custom_dict.json', gzipped=False)   
    
    return text_all_corrected

# def viterbi_segment(text):
#     probs, lasts = [1.0], [0]
#     for i in range(1, len(text) + 1):
#         prob_k, k = max((probs[j] * word_prob(text[j:i]), j)
#                         for j in range(max(0, i - max_word_length), i))
#         probs.append(prob_k)
#         lasts.append(k)
#     words = []
#     i = len(text)
#     while 0 < i:
#         words.append(text[lasts[i]:i])
#         i = lasts[i]
#     words.reverse()
#     return words, probs[-1]

def TextClassificationV1(maxLen, n_classes, embedding_dim, output_bias=None):
    """
    Function creating the recurrent text classification model's graph.
    
    Arguments:
    
    Returns:
    model -- a model instance in Keras
    """
    if n_classes == 2:
        n_classes = 1

    if output_bias is not None:
        output_bias = Constant(output_bias)
    
    # Define sentence_indices as the input of the graph
    # It should be of shape input_shape and dtype 'int32' (as it contains indices).
    sentence_embedding = Input((maxLen, embedding_dim), dtype='float32')   
    
    # Propagate the embeddings through an GRU layer with 128-dimensional hidden state
    # Be careful, the returned output should be a batch of sequences.
    X = Bidirectional(GRU(128, return_sequences=True))(sentence_embedding)
    # Add dropout with a probability of 0.2
    X = Dropout(0.2)(X)
    # Propagate X trough another GRU layer with 128-dimensional hidden state
    # Be careful, the returned output should be a single hidden state, not a batch of sequences.
    X = Bidirectional(GRU(128, return_sequences=False))(X)
    # Add dropout with a probability of 0.2
    X = Dropout(0.2)(X)
    # Propagate X through a Dense layer with softmax activation to get back a batch of 5-dimensional vectors.
    X = Dense(n_classes, activation='sigmoid', bias_initializer=output_bias)(X)
    
    # Create Model instance which converts sentence_indices into X.
    model = Model(inputs=sentence_embedding, outputs=X)
    
    return model

def TextClassification(maxLen, embedding_matrix, n_classes, num_tokens, embedding_dim, output_bias=None):
    """
    Function creating the recurrent text classification model's graph.
    
    Arguments:
    
    Returns:
    model -- a model instance in Keras
    """
    if n_classes == 2:
        n_classes = 1

    if output_bias is not None:
        output_bias = Constant(output_bias)
    
    # Define sentence_indices as the input of the graph
    # It should be of shape input_shape and dtype 'int32' (as it contains indices).
    sentence_indices = Input((maxLen,), dtype='int32')
    
    #Load the embedding matrix as the weights matrix for the embedding layer and set trainable to False
    Embedding_layer = Embedding(
        num_tokens,
        embedding_dim,
        embeddings_initializer=Constant(embedding_matrix),
        mask_zero=True,
        trainable=False)
    
    # Propagate sentence_indices through your embedding layer, you get back the embeddings
    embeddings = Embedding_layer(sentence_indices)    
    
    # Propagate the embeddings through an GRU layer with 128-dimensional hidden state
    # Be careful, the returned output should be a batch of sequences.
    X = Bidirectional(GRU(128, return_sequences=True))(embeddings)
    # Add dropout with a probability of 0.2
    X = Dropout(0.2)(X)
    # Propagate X trough another GRU layer with 128-dimensional hidden state
    # Be careful, the returned output should be a single hidden state, not a batch of sequences.
    X = Bidirectional(GRU(128, return_sequences=False))(X)
    # Add dropout with a probability of 0.2
    X = Dropout(0.2)(X)
    # Propagate X through a Dense layer with softmax activation to get back a batch of 5-dimensional vectors.
    X = Dense(n_classes, activation='sigmoid', bias_initializer=output_bias)(X)
    
    # Create Model instance which converts sentence_indices into X.
    model = Model(inputs=sentence_indices, outputs=X)
    
    return model

def plot_confusion_matrix(cm, classes, title='Confusion matrix'):
    
    cm_norm = np.round(cm.astype(float) / cm.sum(axis=0)*100, 2)

    title_fontsize = 20
    ann_fontsize = 14
    axis_fontsize = 16
    labels_fontsize = 18
    
    plt.rcParams.update({'font.size': ann_fontsize})
    plt.rcParams.update({'font.family':'sans-serif'})
    plt.figure(figsize=(5,5))
    plt.imshow(cm_norm, interpolation='nearest', cmap='Reds')
    #plt.title(title, fontsize=title_fontsize)
    tick_marks = np.arange(len(classes))
    N_neg, N_pos = cm.sum(axis=0)
    plt.xticks(tick_marks, [f'{classes[0]}\n_______\n{np.round(N_neg,1)}', f'{classes[1]}\n_______\n{np.round(N_pos,1)}'], rotation=0, fontsize=axis_fontsize, weight='bold')
    plt.yticks(tick_marks, classes, fontsize=axis_fontsize, weight='bold')
    #plt.colorbar()
    thresh = cm_norm.max() / 1.8

    group_names = np.array([['True Neg','False Neg'],['False Pos','True Pos']])
    n,m = cm.shape
    for i, j in itertools.product(range(n), range(m)):
        plt.text(j, i, f'{group_names[i,j]} \n{np.round(cm[i,j], 1)} \n{cm_norm[i,j]}%', horizontalalignment="center", color="white" if cm_norm[i, j] > thresh else "black")

    plt.tight_layout()    
    plt.grid(False)
    plt.ylabel('Prediction', fontsize=labels_fontsize)
    plt.xlabel('True Value', fontsize=labels_fontsize)