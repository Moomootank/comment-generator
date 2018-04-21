# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 05:51:01 2018

@author: Moomootank
"""
import tensorflow as tf
import pickle
import numpy as np
import pandas as pd
from seq_seq_network import Seq2SeqGenerator
from comment_generator_config import CommentGeneratorConfig

def load_obj(location):
    with open(location,'rb') as file:
        return pickle.load(file)

#=====
        
def create_seqseq_dict(url_root):
    #Function that returns a dictionary to customize the CommentGeneratorObject with
    #Normally, would feed arguments into this function
    #However, since there are so many...just modify this function if you need to change model params

    params_dict = {}

    params_dict['num_classes'] = 57 #Number of characters for the model. 54 + end, start, sep tokens
    params_dict['num_batches'] = 600
    
    embeddings_url = url_root + r"char_embeddings_df.pickle"
    params_dict['embeddings'] = load_obj(embeddings_url).values 
    #Just need to have one copy of all the embeddings. values, because want numpy array
    
    vocab_idx_url = url_root + r"char_idx_dict.pickle"
    params_dict['vocab_idx_dict'] = load_obj(vocab_idx_url)
    
    idx_vocab_dict_url = url_root + r"idx_char_dict.pickle"
    params_dict['idx_vocab_dict'] = load_obj(idx_vocab_dict_url)    
    params_dict["starting_char"] = '*'
    params_dict["num_enc_layers"] = 1
    params_dict['ending_char'] = "~" #Char that means sequence has ended
    params_dict['max_time'] = 300
        
    params_dict['num_epochs'] = 11
    params_dict['learning_rate'] = 1e-4
    params_dict['num_layers'] = 1
    params_dict['log_location'] = r"max_len_350_saved/log_location" #Where you want to save the tensorboard log
    params_dict['chkpt'] = r"max_len_350_saved/log_location" #Where you want to save checkpoints
    
    return params_dict

#=====Functions that train the model=====
    
def train_seqseq_model(file_loc,
                config_obj, params, buffer_size):
    '''
    Function that trains a tensorflow model with the desired parameters, then checks loss on validation/test set
    
    '''
    print("Optimizing params:", params)
    graph = tf.Graph()
    with graph.as_default():        
        session = tf.Session()
        
        
        model = Seq2SeqGenerator(session, config_obj)
        model.define_network_hyperparams(params)
        #unfold params into the model
        
        model.writer = tf.summary.FileWriter(model.log_location, graph=graph)
                
        losses = model.fit(file_loc, buffer_size)

        #saver.save(session, r"D:\Data Science\Projects\comment_generator\models\test_save.ckpt")
        #saver.restore(sess, "training_logs/checkpoints/current_best.ckpt") #Restore the best model 
        text = model.generate_inferences("No one gets left behind. I am committed to helping ALL Americans achieve the American dream, including our farmers, small businesses, and great veterans!|Thank you Mr President", None)
        print (text)
        model.writer.add_graph(graph)
        return graph, session, model, losses
    
#=====Test function to read data=====
        
def decode(serialized_example):
  # NOTE: You might get an error here, because it seems unlikely that the features
  # called 'coord2d' and 'coord3d', and produced using `ndarray.flatten()`, will
  # have a scalar shape. You might need to change the shape passed to
  # `tf.FixedLenFeature()`.
  features = tf.parse_single_example(
      serialized_example,
      features={'post': tf.FixedLenFeature([1, 300], tf.int64),
                'comment': tf.FixedLenFeature([1, 300], tf.int64),
                'label': tf.FixedLenFeature([1, 300], tf.int64)})

  # NOTE: No need to cast these features, as they are already `tf.float32` values.
  return features['post'], features['comment'], features['label']


if __name__ == "__main__":
    url_root = r"../data/data_for_seqseq/dual_enc_max_300/"

    posts_idx = load_obj(url_root + r"post_indices.pickle")
    c_idx = load_obj(url_root + r"tbg_indices.pickle")
    c_labels = load_obj(url_root + r"tbg_labels.pickle")
    param_dict = create_seqseq_dict(url_root)
    config_obj = CommentGeneratorConfig(param_dict)
    

    params = {'encoder_n_hidden_units': [350],
              'decoder_n_hidden_units': [350],
              'n_encoder_dropout': [0.2],
              'n_decoder_dropout': [0.2]}
    
    train_filename = r"../data/data_for_seqseq/dual_enc_max_300/train_data.tfrecords"    
    '''
    graph, session, model, losses = train_seqseq_model(train_filename,
                config_obj, params, 570965)
    
    with graph.as_default():
        #text = model.generate_inferences("No one gets left behind. I am committed to helping ALL Americans achieve the American dream, including our farmers, small businesses, and great veterans!|Thank", None)
        tvars = tf.trainable_variables()
        tvars_vals = session.run(tvars)
        for var, val in zip(tvars, tvars_vals):
            print(var.name, val.shape)
            print  ("------")
    '''
    
    