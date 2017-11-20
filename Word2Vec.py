import logging
import os
import sys
import multiprocessing
import numpy as np
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim import models
from gensim.models.keyedvectors import KeyedVectors


class Word2Vec_Model():
    def __init__(self, training_data=None, word2vec_model_path=None):

        self.training_data = training_data
        self.word2vec_model_path = word2vec_model_path


    def train_word2vec(self, inp, outp1, outp2):
        model = Word2Vec(LineSentence(inp), size=500, window=10, min_count=10, workers=multiprocessing.cpu_count()-1)
        model.wv.save_word2vec_format(outp1, binary=False)
        model.save(outp2)

    def load_word2vec(self):
        print("##### loading word2vec model #####")
        model = KeyedVectors.load_word2vec_format(self.word2vec_model_path, binary=False)
        return model

    def get_classes_text_embedding(self, all_text_embedding, classes):
        classes_text_embedding = []
        for class_label in classes:
            if "_" in class_label:
                embedding = [0.0 for x in range(all_text_embedding.vector_size)]
                for word in class_label.split("_"):
                    embedding += all_text_embedding[word]
                classes_text_embedding.append(embedding)
            else: 
                classes_text_embedding.append(all_text_embedding[class_label])
        return np.array(classes_text_embedding)

    def nearest_neighbor_embeddings(self, input_embedding, all_text_embedding, num_nearest_neighbor):
        return all_text_embedding.most_similar(positive=[input_embedding], topn=num_nearest_neighbor)
        


 
    # word2vec_model_path = "./Data/wiki.en.vec"
    # #train_word2vec(inp, outp1, outp2)
    # model = load_word2vec(word2vec_model_path)
    # create_embedding_lookup_table(model, classes)



    
