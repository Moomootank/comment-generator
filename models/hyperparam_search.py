# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 09:25:59 2018

@author: Moomootank
"""

import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from comment_generator_config import CommentGeneratorConfig
from generator_network import GeneratorNetwork

def load_obj(url):
    with open(url, 'rb') as file:
        return pickle.load(file)
    
#=====Functions that set up the parameters of the model=====
def create_comment_generator_dict():
    #Function that returns a dictionary to customize the CommentGeneratorObject with
    #Normally, would feed arguments into this function
    #However, since there are so many...just modify this function if you need to change model params
    
    params_dict = {}
    
    #params_dict['embed_size'] = 55 
    params_dict['num_classes'] = 55 #Number of characters for the model 
    params_dict['batch_size'] = 1000
    
    url_root = r"../data/data_for_gen_model/max_len_350/"
    embeddings_url = url_root + r"char_embeddings_df.pickle"
    params_dict['embeddings']  = load_obj(embeddings_url) #Just need to have one copy of all the embeddings
    
    vocab_idx_url =  url_root + r"char_idx_dict.pickle"
    params_dict['vocab_idx_dict'] = load_obj(vocab_idx_url)
    
    idx_vocab_dict_url = url_root + r"idx_char_dict.pickle"
    params_dict['idx_vocab_dict'] = load_obj(idx_vocab_dict_url)
    
    params_dict['ending_char'] = "~" #Char that means sequence has ended
    params_dict['max_time'] = 350
        
    params_dict['num_epochs'] = 10
    params_dict['learning_rate'] = 1e-4
    params_dict['num_layers'] = 2
    params_dict['log_location']  = r"/max_len_350_saved/log_location" #Where you want to save the intermediate models
    params_dict['chkpt'] = r"/max_len_350_saved/log_location" #Where you want to save checkpoints
    
    return params_dict

#=====Functions that train the model=====

def train_model(train_indices, train_labels, other_indices, other_labels, model, params):
    '''
    Function that trains a tensorflow model with the desired parameters, then checks loss on validation/test set
    
    '''
    print ("Optimizing params:", params)
    graph = tf.Graph()
    with graph.as_default():        
        #model.define_fixed_hyperparams(200,3, 1694, 659, 600, 1e-4, 35, 2, embedding_matrix)
        #n_features, n_classes, batch, other_batch, n_epochs, lr, max_l, num_layers, embeddings
        model.define_network_hyperparams(**params)
        #unfold params into the model
        
        model.initialize_ops()
        variables_init = tf.global_variables_initializer()
        #sess = tf.Session() # Not using "with tf.Session() as sess" as that would close the session outside of the indent      
        model.session.run(variables_init)
        saver = tf.train.Saver()
        
        losses = model.fit(train_indices, train_labels, None, None, saver)
        sns.tsplot(losses)
        plt.show()
        plt.clf()
        print()
        
        #saver.restore(sess, "training_logs/checkpoints/current_best.ckpt") #Restore the best model 

        return graph, model

if __name__ == "__main__":
    params_dict = create_comment_generator_dict()
    config_obj = CommentGeneratorConfig(params_dict)
    
    session = tf.Session()
    comment_generator = GeneratorNetwork(session, config_obj)
    
    nndict = {'n_hidden_units': [140, 180], 
              'n_dropout' : [0.3, 0.4]}
    
    input_data_url = "../data/data_for_gen_model/max_len_350/comment_indices_350.pickle"
    train_indices = load_obj(input_data_url)
    
    train_labels = train_indices[:, 1:]
    last_col = np.full((train_labels.shape[0],1), 54)
    train_labels = np.concatenate([train_labels, last_col], axis = 1)
    graph, model = train_model(train_indices, train_labels, np.zeros(350), np.zeros(350), comment_generator, nndict)
    