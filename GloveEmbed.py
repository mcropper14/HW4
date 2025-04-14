"""
Get pre-train embedding from GLOVE
"""

import numpy as np
import torch


## download the pretrain embedding
def _read_glove_embedding(glove_file):
    '''Fun:read embedding from pre-train by glove
    '''
    ## map word to embedding
    word2embedding = dict()
    with open(glove_file,"r") as f:
        for num,line in enumerate(f):
            values = line.split()
            word = values[0]
            emb = np.array(values[1:], dtype='float32')
            word2embedding[word] = emb
            
    return word2embedding


def _get_embedding(glove_file,tokens2index,embed_dim=200):
    """Fun:get the embedding matrix for our vocabulary
    """
    ##load glove embedding to embedding matrix
    word2embedding = _read_glove_embedding(glove_file)
    embedding_matrix = np.zeros((len(tokens2index), embed_dim))
    for word, i in tokens2index.items():
        embedding_vector = word2embedding.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    
    return torch.FloatTensor(embedding_matrix)
    
