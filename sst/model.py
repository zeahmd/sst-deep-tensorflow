from sst.utils import loadFastTextModel, buildEmbeddingMatrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, BatchNormalization, Dense, Dropout
from tensorflow.keras.layers import GRU, Bidirectional, SimpleRNN, Conv1D, GlobalMaxPool1D
import tensorflow as tf
from loguru import logger
import os


def LSTM_Model(weights, vocab_size, embedding_size, max_sen_len, num_classes):
    return Sequential([
        Embedding(vocab_size, embedding_size, weights=[weights], trainable=False,
        input_shape=(max_sen_len,)),
        LSTM(32, return_sequences=True),
        BatchNormalization(),
        Dropout(0.1),
        LSTM(64, return_sequences=True),
        BatchNormalization(),
        Dropout(0.1),
        LSTM(128),
        BatchNormalization(),
        Dropout(0.2),
        Dense(10, activation='relu', name='relu_dense'),
        BatchNormalization(),
        Dense(num_classes, activation='softmax', name='softmax_dense')
    ])

def GRU_Model(weights, vocab_size, embedding_size, max_sen_len, num_classes):
    initializer = tf.keras.initializers.GlorotUniform()
    return Sequential([
        Embedding(vocab_size, embedding_size, weights=[weights], trainable=False,
        input_shape=(max_sen_len,)),
        GRU(32, return_sequences=True, kernel_initializer=initializer, recurrent_initializer=initializer),
        BatchNormalization(),
        Dropout(0.1),
        GRU(64, return_sequences=True, kernel_initializer=initializer, recurrent_initializer=initializer),
        BatchNormalization(),
        Dropout(0.1),
        GRU(128),
        BatchNormalization(),
        Dropout(0.2),
        Dense(10, activation='relu', kernel_initializer=initializer, name='relu_dense'),
        BatchNormalization(),
        Dense(num_classes, activation='softmax', name='softmax_dense')
    ])

def RNN_Model(weights, vocab_size, embedding_size, max_sen_len, num_classes):
    initializer = tf.keras.initializers.GlorotUniform()
    return Sequential([
        Embedding(vocab_size, embedding_size, weights=[weights], trainable=False,
        input_shape=(max_sen_len,)),
        SimpleRNN(32, return_sequences=True, kernel_initializer=initializer, recurrent_initializer=initializer),
        BatchNormalization(),
        Dropout(0.1),
        SimpleRNN(64, return_sequences=True, kernel_initializer=initializer, recurrent_initializer=initializer),
        BatchNormalization(),
        Dropout(0.1),
        SimpleRNN(128),
        BatchNormalization(),
        Dropout(0.2),
        Dense(10, activation='relu', kernel_initializer=initializer, name='relu_dense'),
        BatchNormalization(),
        Dense(num_classes, activation='softmax', name='softmax_dense')
    ])

def Conv1d_Model(weights, vocab_size, embedding_size, max_sen_len, num_classes):
    return Sequential([
        Embedding(vocab_size, embedding_size, weights=[weights], trainable=False,
        input_shape=(max_sen_len,)),
        Conv1D(128, 3, strides=1, padding='SAME', activation='relu'),
        BatchNormalization(),
        Dropout(0.1),
        Conv1D(256, 3, strides=1, padding='SAME', activation='relu'),
        BatchNormalization(),
        Dropout(0.1),
        GlobalMaxPool1D(),
        BatchNormalization(),
        Dropout(0.2),
        Dense(10, activation='relu', name='relu_dense'),
        BatchNormalization(),
        Dense(num_classes, activation='softmax', name='softmax_dense')
    ])

def BiLSTM_Model(weights, vocab_size, embedding_size, max_sen_len, num_classes):
    return Sequential([
        Embedding(vocab_size, embedding_size, weights=[weights], trainable=False,
        input_shape=(max_sen_len,)),
        Bidirectional(LSTM(32, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.1),
        Bidirectional(LSTM(64, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.1),
        Bidirectional(LSTM(128)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(10, activation='relu', name='relu_dense'),
        BatchNormalization(),
        Dense(num_classes, activation='softmax', name='softmax_dense')
    ])

def buildModel(name='lstm', word_index={}, vocab_size=16222, max_sen_len=56, num_classes=5):
    EMBEDDING_SIZE = 300

    embeddings_index = loadFastTextModel(path='sst/embedding/crawl-300d-2M.vec')
    embedding_matrix = buildEmbeddingMatrix(word_index, vocab_size, EMBEDDING_SIZE, embeddings_index)

    if name is 'lstm':
        return LSTM_Model(embedding_matrix, vocab_size, EMBEDDING_SIZE, max_sen_len, num_classes)
    elif name is 'gru':
        return GRU_Model(embedding_matrix, vocab_size, EMBEDDING_SIZE, max_sen_len, num_classes)
    elif name is 'rnn':
        return RNN_Model(embedding_matrix, vocab_size, EMBEDDING_SIZE, max_sen_len, num_classes)
    elif name is 'bilstm':
        return BiLSTM_Model(embedding_matrix, vocab_size, EMBEDDING_SIZE, max_sen_len, num_classes)
    elif name is 'conv1d':
        return Conv1d_Model(embedding_matrix, vocab_size, EMBEDDING_SIZE, max_sen_len, num_classes)
    else:
        logger.error(f"Invalid model name {name}")
        os._exit(0)

