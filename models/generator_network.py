# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 06:13:31 2018

@author: Moomootank
"""

import tensorflow as tf

class GeneratorNetwork():
    #=====Functions that set up the parameters of the network=====
    def __init__(self, sess, config):
        self.num_features = config.embed_size 
        self.num_classes = config.num_classes
        self.num_batches = config.batch_size #Number of batches you want
        self.pretrained_embeddings = config.embeddings #Just need to have one copy of all the embeddings

        self.num_epochs =  config.num_epochs
        self.learning_rate = config.learning_rate
        self.num_layers = config.num_layers 
        self.log_location = config.log_location #Where you want to save the intermediate models
        
        self.session = sess #tensorflow session
    
    def define_network_hyperparams(self, n_hidden_units, n_dropout, n_input_dropout):
        #Placed in another function instead of init to allow for easier
        #parameter optimization
        self.num_hidden_units = n_hidden_units
        self.num_dropout = n_dropout
        self.num_input_dropout = n_input_dropout
    
    #=====Functions that set up the basic structure of the network, such as data=====
    def add_placeholders(self):
        with tf.name_scope("Data"):
            self.input_placeholder = tf.placeholder(dtype= tf.float32, shape = (None, self.num_features))
            self.labels_placeholder = tf.placeholder(dtype= tf.int32, shape = (None))
            self.input_len_placeholder = tf.placeholder(dtype = tf.int16, shape = (None))
            #Need dropout to be in placeholder format to make it easier to turn it off during prediction
            self.dropout_placeholder = tf.placeholder(dtype = tf.float32, shape = (self.num_layers)) 
            self.input_dropout_placeholder = tf.placeholder(dtype = tf.float32, shape = (self.num_layers))
               
    def create_feed_dict(self, inputs_batch, labels_batch, n_dropout, input_len):
        feed_dict = {self.input_placeholder: inputs_batch, self.labels_placeholder: labels_batch, 
                     self.dropout_placeholder: n_dropout,
                     self.input_len_placeholder: input_len }
        return feed_dict
    
    def add_cells(self):
        assert len(self.num_hidden_units)==len(self.num_dropout)
        assert len(self.num_hidden_units)==self.num_layers
        
        forward_cells = []
        init = tf.contrib.layers.xavier_initializer(uniform = True, dtype= tf.float32)
        
        for i in range(self.num_layers):
            without_dropout = tf.contrib.rnn.LSTMCell(self.num_hidden_units[i], activation = tf.relu, initializer = init)
            one_layer = tf.contrib.rnn.DropoutWrapper(without_dropout, 
                                                      output_keep_prob = 1 - self.dropout_placeholder[i])
            forward_cells.append(one_layer)
    
        multi_forward_cell = tf.contrib.rnn.MultiRNNCell(forward_cells)
        return multi_forward_cell
    
    def add_embeddings(self):
        #Deal with this later
        with tf.name_scope("Data"):
            embedding = tf.nn.embedding_lookup(params = self.pretrained_embeddings, ids = self.input_placeholder)
        return tf.cast(embedding, dtype = tf.int16) #Int, as it is just returning a one-hot
    
    #=====Functions that run the model=====
    def prediction_op(self):
        init = tf.contrib.layers.xavier_initializer(uniform = True, dtype= tf.float32)
        x = self.add_embeddings()
        
        cells = self.add_cells()
        outputs, state = tf.nn.dynamic_rnn(cells, inputs = x, sequence_length = self.input_len_placeholder, dtype = tf.float32)
            
        '''
        outputs: Tensor shaped: [batch_size, max_time, cell.output_size]
        cell.output_size == self.num_hidden_units[-1] (hopefully lol)
        '''
        #class_weights: going to be the hidden size of the last layer, and num_classes = cell.output_size
        class_weights = tf.get_variable("class_weights", shape = (self.num_hidden_units[-1], self.num_classes))
        class_bias = tf.get_variable("class_bias", initializer = init, shape = (self.num_classes))
        
        reshaped_output = tf.reshape(outputs, shape = [-1, self.num_hidden_units[-1]])
        predictions = tf.matmul(reshaped_output, class_weights) + class_bias #[batch_size * max_time, self.num_classes]
        return predictions
        
    def loss_op(self, predictions):
        with tf.name_scope("loss_ops"):
            labels_reshaped = tf.reshape(self.labels_placeholder, shape = [-1])
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self.labels_reshaped, logits = predictions)
            loss = tf.reduce_mean(loss)
        return loss
    
    def training_op(self, loss):
        with tf.name_scope("optimizer"):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            train_op = optimizer.minimize(loss)      
        return train_op
    
    def initialize_ops(self):
        self.add_placeholders() # Add the placeholders
        self.pred = self.prediction_op() #Add the prediction op
        self.loss = self.loss_op(self.pred)
        self.train = self.training_op(self.loss)