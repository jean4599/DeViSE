import logging
import os
import sys
import multiprocessing
import numpy as np
from sklearn.neighbors import NearestNeighbors
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim import models
from gensim.models.keyedvectors import KeyedVectors


class Word2Vec_Model():
    def __init__(self, training_data=None, word2vec_model_path=None, nearest_neighbor=None):

        self.training_data = training_data
        self.word2vec_model_path = word2vec_model_path
        self.nearest_neighbor = nearest_neighbor


    def train_word2vec(self, inp, outp1, outp2):
        model = Word2Vec(LineSentence(inp), size=500, window=10, min_count=10, workers=multiprocessing.cpu_count()-1)
        model.wv.save_word2vec_format(outp1, binary=False)
        model.save(outp2)

    def load_word2vec(self):
        print("##### loading word2vec model #####")
        model = KeyedVectors.load_word2vec_format(self.word2vec_model_path, binary=False)
        return model

    def load_light_word2vec(self):
        print ("##### Loading light-weight Glove Word2Vec Model #####")
        model = {}
        with open(self.word2vec_model_path, encoding='utf-8') as data_file:
            for line in data_file:
                splitLine = line.split()
                word = splitLine[0]
                embedding = np.array([float(val) for val in splitLine[1:]], dtype=np.float32)
                model[word] = embedding
            print ("Done.", len(model), " words loaded!")
        return model

    def text_embedding_lookup(self, all_text_embedding, embedding_dim, label):
        # labels_embedding = np.zeros((labels.shape[0], all_text_embedding.vector_size))
        # for i, label in enumerate (labels):
        #     labels_embedding[i] = classes[label]
        embedding = [0.0 for x in range(embedding_dim)]
        if "_" in label:
            num_word = 0
            for word in label.split("_"):
                num_word += 1
                embedding += all_text_embedding[word]
            embedding /= num_word
        else: 
            embedding = all_text_embedding[label]
        return embedding

    def get_classes_text_embedding(self, all_text_embedding, embedding_dim, classes):
        classes_text_embedding = []
        for class_label in classes:
            if "_" in class_label:
                embedding = [0.0 for x in range(embedding_dim)]
                for word in class_label.split("_"):
                    embedding += all_text_embedding[word]
                classes_text_embedding.append(embedding)
            else: 
                classes_text_embedding.append(all_text_embedding[class_label])
        return np.array(classes_text_embedding, dtype=np.float32)

    #def nearest_neighbor_embeddings(self, input_embedding, all_text_embedding, num_nearest_neighbor):

        # return all_text_embedding.most_similar(positive=[input_embedding], topn=num_nearest_neighbor)
    
    def train_nearest_neighbor(self, all_text_embedding, num_nearest_neighbor=1):
        print("NearestNeighbors training start...")
        embeddings = list(all_text_embedding.values())
        embeddings = np.array(embeddings, dtype=np.float32) # Get embeddings from all_text_embedding dictionary
        print('In train_nearest_neighbor: embeddings shape:', embeddings.shape)
        nbrs = NearestNeighbors(num_nearest_neighbor, algorithm='ball_tree').fit(embeddings)
        self.nearest_neighbor = nbrs
        distances, indices = self.nearest_neighbor.kneighbors([all_text_embedding['dog']])
        print(distances)
        print(indices)
        return 'NearestNeighbors training finished!'

    # Action: Find [num_nearest_neighbor] nearest neighbor(s) of [input_embedding] in [all_text_embedding]
    # Member: all_text_embedding--a dinctionary with entries: {'label': text embedding}
    # Member: input_embedding--a embedding 
    def get_nearest_neighbor_labels(self, embeddings, labels):
        distances, indices = self.nearest_neighbor.kneighbors(embeddings)
        # indices: an array with shape [# of visual_embeddings, 5], each element is the index of a neibor
        # distances: same shape of indeces, each element is the distance between visual_embeddings[i] and corresponding neighbor
        print('In W2V NN')
        print('Labels shape: ', labels.shape, 'label 0:', labels[0])
        
        nn_labels = []
        for idx, index in enumerate(indices):
            nn_labels.append(labels[index])

        return np.array(nn_labels, dtype=np.str)


 
    # word2vec_model_path = "./Data/wiki.en.vec"
    # #train_word2vec(inp, outp1, outp2)
    # model = load_word2vec(word2vec_model_path)
    # create_embedding_lookup_table(model, classes)



    
