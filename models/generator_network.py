# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 06:13:31 2018

@author: Moomootank
"""

import tensorflow as tf
import numpy as np
import time
import math

class GeneratorNetwork():
    #=====Functions that set up the parameters of the network=====
    def __init__(self, sess, config):
        #self.num_features = config.embed_size 
        self.num_classes = config.num_classes
        self.num_batches = config.num_batches #Number of batches you want
        
        self.pretrained_embeddings = config.embeddings #Just need to have one copy of all the embeddings
        self.session = sess #tensorflow session
        self.vocab_to_idx = config.vocab_idx_dict
        self.idx_to_vocab = config.idx_vocab_dict
        self.ending_char = config.ending_char #Char that means sequence has ended
        self.max_time = config.max_time
        
        self.num_epochs =  config.num_epochs
        self.learning_rate = config.learning_rate
        self.num_layers = config.num_layers 
        self.log_location = config.log_location #Where you want to save the intermediate models
        self.checkpoint_location = config.chkpt #Where you want to save checkpoints
        
    def define_network_hyperparams(self, n_hidden_units, n_dropout):
        #Placed in another function instead of init to allow for easier
        #parameter optimization
        self.num_hidden_units = n_hidden_units
        self.num_dropout = n_dropout
        #self.num_input_dropout = n_input_dropout
    
    #=====Functions that set up the basic structure of the network, such as data=====
    def add_placeholders(self):
        with tf.name_scope("Data"):
            self.input_placeholder = tf.placeholder(dtype= tf.int32, shape = (None, None))
            self.labels_placeholder = tf.placeholder(dtype= tf.int32, shape = (None))
            self.input_len_placeholder = tf.placeholder(dtype = tf.int32, shape = (None))
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
            without_dropout = tf.contrib.rnn.LSTMCell(self.num_hidden_units[i], activation = tf.nn.tanh, initializer = init)
            one_layer = tf.contrib.rnn.DropoutWrapper(without_dropout, 
                                                      output_keep_prob = 1 - self.dropout_placeholder[i])
            forward_cells.append(one_layer)
    
        multi_forward_cell = tf.contrib.rnn.MultiRNNCell(forward_cells)
        return multi_forward_cell
    
    def add_embeddings(self):
        #Deal with this later
        with tf.name_scope("Data"):
            embedding = tf.nn.embedding_lookup(params = self.pretrained_embeddings, ids = self.input_placeholder)
        return tf.cast(embedding, dtype = tf.float32) #Int, as it is just returning a one-hot
    
    #=====Functions that make up the heart of the model=====
    def prediction_op(self):
        init = tf.contrib.layers.xavier_initializer(uniform = True, dtype= tf.float32)
        x = self.add_embeddings()
        
        cells = self.add_cells()
        outputs, states = tf.nn.dynamic_rnn(cells, inputs = x, sequence_length = self.input_len_placeholder
                                            , dtype = tf.float32)
            
        '''
        outputs: Tensor shaped: [batch_size, max_time, cell.output_size]
        cell.output_size == self.num_hidden_units[-1] (hopefully lol)
        For each row, after the seq_len for it, the output is just 0
        Softmax of all 0: each value is just 1/num_classes
        '''
        #class_weights: going to be the hidden size of the last layer, and num_classes = cell.output_size
        class_weights = tf.get_variable("class_weights", initializer = init, shape = (self.num_hidden_units[-1], self.num_classes))
        class_bias = tf.get_variable("class_bias", initializer = init, shape = (self.num_classes))
        
        reshaped_output = tf.reshape(outputs, shape = [-1, self.num_hidden_units[-1]])
        predictions = tf.matmul(reshaped_output, class_weights) + class_bias #[batch_size * max_time, self.num_classes]
        return predictions
        
    def loss_op(self, predictions):
        with tf.name_scope("loss_ops"):
            labels_reshaped = tf.reshape(self.labels_placeholder, shape = [-1]) #1d array of batch_Size* max_time
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels_reshaped, logits = predictions)
            #loss should be tensor of batch_size*max_time now, with each entry being the cross_entropy loss
            
            length_mask = tf.sequence_mask(self.input_len_placeholder, maxlen = self.max_time, dtype = tf.float32)
            #this should generate boolean array of [batch_size, max_time]
            length_mask = tf.reshape(length_mask, shape = [-1]) #1d array of batch_Size* max_time
            
            masked_loss = tf.multiply(loss, length_mask) 
            #element wise multiplication; loss of superfluous elements becomes 0
            masked_loss = tf.reduce_sum(masked_loss) #Summing loss all up; should not include superfluous loss now 
            correct_length = tf.reduce_sum(length_mask) #number of actual elements
            masked_loss = masked_loss/correct_length
           
            return masked_loss
    
    def training_op(self, loss):
        with tf.name_scope("optimizer"):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon = 1e-6)
            train_op = optimizer.minimize(loss)
            
        return train_op
    
    def initialize_ops(self):
        self.add_placeholders() # Add the placeholders
        self.pred = self.prediction_op() #Add the prediction op
        self.loss = self.loss_op(self.pred) 
        self.train = self.training_op(self.loss)
        
    #=====These functions execute the model=====
    
    #===== The following functions execute the model =====
    
    def get_minibatches(self, data, labels, lens, num_batches):
        '''
        Helper function that returns a list of tuples of batch_size sized inputs and labels
        Return n_samples/batch size number of batches
        Args:
            data: dataframe with columns of independent variables
            x_cols: list of length n_features of independent column names
            labels: name of label column       
            b_size: size of each batch
            
        Returns: List of tuples where each tuple is (input_batch, label_batch)
        '''
            
        data_length = len(data)
        assert data_length==len(labels)
        
        p = np.random.permutation(data_length)
        reshuffled_data = data[p]
        reshuffled_labels = labels[p] #These create copies of the array, so the original copies are untouched
        reshuffled_lengths = lens[p]
        batches = []         
        b_size = math.ceil(len(data)/num_batches) 

        for i in range(num_batches):
            start = i*b_size
            if start>=data_length:
                print ("Start exceeds max_length!")
                break
            end = start + b_size
            if end>=data_length:
                end = data_length
            
            data_sample = reshuffled_data[start:end] # Sample of b size
            labels_sample = reshuffled_labels[start: end]
            lens_sample = reshuffled_lengths[start: end]
            batches.append((data_sample, labels_sample, lens_sample))
                
        return batches
    
    def track_time(self, start, previous, average_loss, epoch):
        current_time = time.time()
        duration = current_time - start
        duration_min = math.floor(duration/60)
        duration_sec = duration % 60
        since_last = current_time - previous
        since_last_min = math.floor(since_last/60)
        since_last_sec = since_last % 60 
        print ("Epoch number {e} completed. Time taken since start: {start_min} min {start_sec} s."
               .format(e = epoch, start_min = duration_min, start_sec = duration_sec))
        print ("Time taken since last checkpoint: {last_min} min {last_sec} s."
               .format(last_min = since_last_min, last_sec = since_last_sec ))
        print ("Average loss this epoch is:" , average_loss)
        print ()
        
        return current_time
        
    def run_epoch(self, session, data, labels, lens):
        n_minibatches, total_loss = 0, 0
        for input_batch, labels_batch, len_batch in self.get_minibatches(data, labels, lens, self.num_batches):
            feed = self.create_feed_dict(input_batch, labels_batch, self.num_dropout, len_batch)
            _ , batch_loss = session.run([self.train, self.loss], feed_dict = feed) #self.loss will not run two times. This just fetches the value
            #print ("Batch loss is:", batch_loss)
            n_minibatches += 1
            total_loss += batch_loss
            
            if batch_loss>10:
                for v in tf.trainable_variables():
                    print (v.name)
                    print (session.run(v))
                    print (input_batch)
                    print (batch_loss)
                    print ("=========")
            elif math.isnan(batch_loss):
                for v in tf.trainable_variables():
                    print (batch_loss)
                    print (v.name)
                    print (session.run(v))
                    print (input_batch.shape)
                    print (labels_batch.shape)
                    print (len_batch.shape)
                    print ("Minibatch number:", n_minibatches)
                    print ("=========")
                    return 0
                    
                    
        print ("Total loss is:", total_loss)
        epoch_average_loss = total_loss/n_minibatches
        return epoch_average_loss
    
    def fit(self, data, labels, lens, other_data, other_labels, other_lens, saver):
        #data, labels: training data and labels
        #other_data, other_labels: the validation/test data and labels
        
        losses = []
        start = time.time() # Start time of the entire model
        previous = start
        
        best_loss = 100 # Initial val score
        val_losses = [best_loss]
        for epoch in range(self.num_epochs):
            average_loss = self.run_epoch(self.session, data, labels, lens)
            losses.append(average_loss) #Tracking loss per epoch
            '''
            if epoch%10==0 or epoch==self.num_epochs-1:
                #This block checks to see when the best epoch is
                val_loss = self.predict_other(self.session, other_data, other_labels)
                val_losses.append(val_loss)
                print ("New val_loss", val_loss)
                if val_loss < best_loss:
                    best_loss = val_loss    
                    saver.save(self.session, self.log_location)
                    print ("New best model saved. Epoch:", epoch)
            '''
            if epoch % 1 ==0 or epoch==self.num_epochs-1: #-1 due to characteristics of range
            #This if block just prints out the progress and time taken to reach a certain stage
                previous = self.track_time(start, previous, average_loss, epoch) #Set the new "last checkpoint" to this one
                
        return losses, val_losses #Can try to plot how much the loss has gone down
        
    #=====These functions generate a new sequence!=====
    def generate_character_sequence(self, prompt, best_model_location, max_chars, top_char_num):
        assert type(prompt)==str, "Prompt is not a string!"
        prompt = prompt.lower()
        #Not taking in list of strings, as the different lengths of each would make matrix mult difficult anyway
        #with self.session as new_sess:
        #Idk, might be better to use new session for predictions rather than the checkpointed one
        #saver = tf.train.Saver() #Idk, do we need to initialize variables again? Use the same saver?
        #saver.restore(self.session, best_model_location)
        
        my_prompt = [self.vocab_to_idx[char] for char in prompt]
        chars = my_prompt
        end_index = self.vocab_to_idx[self.ending_char]
        
        while True:
            '''
            Procedure:
                1: Take current chars, run the prediction op. This gives you num_char predictions
                2: You only care about the LAST prediction, since that is what you will use to generate the next ochar
                3: Take the last prediction, softmax it, find the most likely char
                4: Append it to the list of chars
                5: Repeat steps 1-4 until you generate a "terminator" char, then break (or until certain limit is reached)
            '''
            if (len(chars)>=max_chars) or (chars[-1] == end_index):
                break
            
            #Just feed something random as labels batch             
            feed = self.create_feed_dict(np.array([chars]), np.array(chars), [0]*self.num_layers, len(chars)) 
            predictions = self.session.run(self.pred, feed_dict = feed)
            next_char_preds = predictions[-1]
            
            #pred_char_index = self.session.run(tf.argmax(tf.nn.softmax(next_char_preds)))
            values, indices = tf.nn.top_k(next_char_preds, k = top_char_num)
            prob_distribution = tf.nn.softmax(values)
            
            prob_distribution, indices = self.session.run([prob_distribution, indices])
            print ("Prob dist is", prob_distribution)
            print ("Indices are:", indices)
            pred_char_index = np.random.choice(indices, 
                                                p = prob_distribution)
            print ("Pred char index is:", pred_char_index)
            chars.append(pred_char_index)

        words = [self.idx_to_vocab[index] for index in chars]
        generated_comment = "".join(words)
        print (generated_comment)
        return generated_comment
    
    #=====These functions are used to validate the hyperparameters=====