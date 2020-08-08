import numpy as np
from loguru import logger
from tqdm import tqdm
import os

def get_binary_label(sentiment):
    if sentiment <= 1:
        return 0
    else:
        return 1

def loadFastTextModel(path=''):
    logger.info("Loading FastText Model!")
    embeddings_index = dict()

    try:
        with open(path, 'r') as f:
            with tqdm(total=1999996, desc='loading FastText') as pbar:
                for line in f:
                    values = line.strip().split(' ')
                    word = values[0]
                    coefs = np.asarray(values[1:], dtype='float32')
                    embeddings_index[word] = coefs
                    pbar.update(1)

        return embeddings_index
    except FileNotFoundError:
        logger.error("Embedding file not in path!")
        os._exit(0)


def buildEmbeddingMatrix(word_index, vocab_size, embedding_size, embeddings_index):
    logger.info("Building Embedding Matrix!")
    embedding_matrix = np.zeros((vocab_size, embedding_size))

    for word, i in word_index.items():
        if i >= vocab_size:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix