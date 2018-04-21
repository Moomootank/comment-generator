# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 06:26:56 2018

@author: Moomootank
"""

import tensorflow as tf
import numpy as np
import time
import math

class Seq2SeqGenerator():
    def __init__(self, sess, config):
        #self.num_features = config.embed_size 
        self.num_classes = config.num_classes
        self.num_batches = config.num_batches #Number of batches you want
        
        self.pretrained_embeddings = config.embeddings #Just need to have one copy of all the embeddings
        self.session = sess #tensorflow session
        self.vocab_to_idx = config.vocab_idx_dict
        self.idx_to_vocab = config.idx_vocab_dict
        self.starting_char = config.starting_char
        self.ending_char = config.ending_char #Char that means sequence has ended
        self.max_time = config.max_time
        
        self.num_epochs =  config.num_epochs
        self.learning_rate = config.learning_rate
        self.num_layers = config.num_layers #Number of decoder layers
        self.num_enc_layers = config.num_enc_layers #Number of enc layers
        self.log_location = config.log_location #Where you want to save the intermediate models
        self.checkpoint_location = config.chkpt #Where you want to save checkpoints
        
        
    def define_network_hyperparams(self, arg_dict):
        #Placed in another function instead of init to allow for easier
        #parameter optimization
        self.num_enc_units = arg_dict['encoder_n_hidden_units']
        self.num_dec_units = arg_dict['decoder_n_hidden_units']
        self.enc_dropout = arg_dict['n_encoder_dropout']
        self.dec_dropout = arg_dict['n_decoder_dropout'] #These are input dropout only

        assert len(self.num_enc_units)==len(self.enc_dropout)
        assert len(self.num_enc_units)==self.num_enc_layers
        assert len(self.num_dec_units)==len(self.dec_dropout)
        assert len(self.num_dec_units)==self.num_layers
        
    #=====Functions that set up the basic structure of the network, such as data=====
    def add_placeholders(self):
        with tf.name_scope("Data"):
            #Placeholders for posts
            self.post_input_ph = tf.placeholder(dtype= tf.int32, shape = (None, None), 
                                                name = "post_input_ph")
            
            #Placeholders for comments: the part to be generated
            self.comments_input_ph = tf.placeholder(dtype= tf.int32, shape = (None, None),
                                                    name = "comments_input_ph")
            self.comments_labels_ph = tf.placeholder(dtype= tf.int32, shape = (None, None),
                                                     name = "comments_labels_ph")
            
            #Need dropout to be in placeholder format to make it easier to turn it off during prediction
            self.encoder_dropout_ph = tf.placeholder(dtype = tf.float32, shape = (self.num_layers),
                                                     name = "encoder_dropout_ph") 
            self.decoder_dropout_ph = tf.placeholder(dtype = tf.float32, shape = (self.num_layers),
                                                     name = "decoder_dropout_ph")
               
    def create_feed_dict(self, data_dict):
        feed_dict = {self.post_input_ph: data_dict['post_indices'], 
                     self.comments_input_ph: data_dict['comments_indices'],
                     self.comments_labels_ph: data_dict['comments_labels'],
                     self.encoder_dropout_ph: data_dict['encoder_dropout'],
                     self.decoder_dropout_ph: data_dict['decoder_dropout']
                     }
        self.feed_dict = feed_dict
        return feed_dict
    
    def add_cells(self, num_hidden_units, dropout_placeholder): 
        with tf.name_scope("create_cells"):
            forward_cells = []
            init = tf.contrib.layers.xavier_initializer(uniform = True, dtype= tf.float32)
            
            for i in range(self.num_layers):
                without_dropout = tf.contrib.rnn.LSTMCell(num_hidden_units[i], activation = tf.nn.tanh, 
                                                          initializer = init, reuse = tf.AUTO_REUSE)
                one_layer = tf.contrib.rnn.DropoutWrapper(without_dropout, 
                                                          output_keep_prob = 1 - dropout_placeholder[i])
                forward_cells.append(one_layer)
        
            multi_forward_cell = tf.contrib.rnn.MultiRNNCell(forward_cells)
            return multi_forward_cell
    
    def add_embeddings(self, input_ids):
        #Creates embeddings from input indices for each character
        with tf.name_scope("add_embeddings"):
            embedding = tf.nn.embedding_lookup(params = self.pretrained_embeddings, ids = input_ids)
            return tf.cast(embedding, dtype = tf.float32) #Int, as it is just returning a one-hot
    
    def obtain_text_length(self, tensor_array):
        #Function for obtaining the lengths of the tensors dynamically
        with tf.name_scope("obtain_text_length"):
            ending_index = self.vocab_to_idx[self.ending_char]
            mask = tf.to_int32(tf.not_equal(tensor_array, ending_index)) #1 if not equal to ending char
            lengths = tf.reduce_sum(mask, axis = 1)
            return lengths
    
    #=====Functions that make up the heart of the model=====
    def encoder_post_op(self):
        with tf.variable_scope("encoder_post_ops"):
            #Simple op that returns state of rnn for use in decoder
            post_embeds = self.add_embeddings(self.post_input_ph) #Array of one hot vectors
            
            cells = self.add_cells(self.num_enc_units, self.encoder_dropout_ph)
            post_lens = self.obtain_text_length(self.post_input_ph)
            outputs, states = tf.nn.dynamic_rnn(cells, inputs = post_embeds, 
                                                sequence_length = post_lens
                                                ,dtype = tf.float32)
                
            '''
            outputs: Tensor shaped: [batch_size, max_time, cell.output_size]
            cell.output_size == self.num_hidden_units[-1] (hopefully lol)
            For each row, after the seq_len for it, the output is just 0
            Softmax of all 0: each value is just 1/num_classes
            '''        
            return states

    
    def decoder_op(self, enc_final_state, training = True):
        #The op for decoding
        with tf.variable_scope("decoder_ops"):
            dec_inputs = self.add_embeddings(self.comments_input_ph)
            decoder_cells = self.add_cells(self.num_dec_units, self.decoder_dropout_ph)
            
            #Helpers define how the decoder read the data.
            #A train helper would just feed the training inputs in at each time step for instance
            #Whereas another helper might feed the previous time-step predictions in instead
            if training:
                comment_lens = self.obtain_text_length(self.comments_input_ph)
                helper = tf.contrib.seq2seq.TrainingHelper(inputs = dec_inputs, 
                                                           sequence_length = comment_lens)
            else:

                starting_idx = self.vocab_to_idx[self.starting_char]
                ending_idx = self.vocab_to_idx[self.ending_char]
                #Need to cast to float, or tensorflow will derp
                #Also needs to fill >1 for some reason, or some bug will trigger
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(tf.cast(self.pretrained_embeddings, tf.float32),
                                                                  tf.fill([2], starting_idx),
                                                                  ending_idx)

            #Think about how inference would work
            
            init = tf.contrib.layers.xavier_initializer(uniform = True, dtype= tf.float32)                                                                       
            decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cells, helper, enc_final_state)
            #For some strange reason would literally not process anything beyond len=300
            #Like if you feed dec_input with len=301 it will just truncate outputs to len=300
            #idk why. can't fix it, but can work around it so w/e
            outputs, states, seq_lens = tf.contrib.seq2seq.dynamic_decode(decoder, 
                                                                          maximum_iterations = self.max_time)
            
            #outputs should be tensor of: [batch_size, self.max_time, num_decoder_units]

        
            logits = tf.contrib.layers.fully_connected(inputs = outputs[0], 
                                                       num_outputs = self.num_classes, 
                                                       weights_initializer = init,
                                                       reuse = tf.AUTO_REUSE, scope = "decoder_ops")
            
            if training:
                return (logits, comment_lens)
            
            else:
                return (logits, logits) #second one is literally useless
    
    def loss_op(self, decoder_output):
        with tf.name_scope("loss_ops"):
            #logits should be batch_size x max_comment_time x num_classes
            logits = decoder_output[0]
            logit_lengths = decoder_output[1]
            
            length_mask = tf.sequence_mask(logit_lengths, maxlen = self.max_time, dtype = tf.float32)
            
            #tf.contrib.seq2seq.sequence_loss already uses nn_ops.sparse_softmax_cross_entropy_with_logits
            loss = tf.contrib.seq2seq.sequence_loss(logits, self.comments_labels_ph, 
                                                    weights = length_mask,
                                                    average_across_timesteps = True,
                                                    average_across_batch = True)
            
            tf.summary.histogram("loss", loss)
            return loss
    
    def train_op(self, loss):
        #I want to try using higher level api actually
        with tf.name_scope("optimizer"):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon = 1e-6)
            train_op = optimizer.minimize(loss)
            
        return train_op
    
    def initialize_train_pipeline(self):
        self.add_placeholders() # Add the placeholders
        self.enc_states = self.encoder_post_op() 
        self.dec_output = self.decoder_op(self.enc_states, True)
        self.loss = self.loss_op(self.dec_output) 
        self.train = self.train_op(self.loss)
        

    #=====These functions execute the model=====
    def get_minibatches(self, data_list, num_batches):
        '''
        Helper function that returns a list of tuples of batch_size sized inputs and labels
        Return n_samples/batch size number of batches
        Args:
            data_list: list of matrices. Should be 5: posts_data, posts_len, comments_data, len, labels
            Every entry in data_list should have the same length 
        
        Returns: List of tuples where each tuple is (input_batch, label_batch)
        '''
            
        data_length = len(data_list[0])
        p = np.random.permutation(data_length)
        reshuffled_data = [frame[p] for frame in data_list]
        
        batches = []         
        b_size = math.ceil(data_length/num_batches) 

        for i in range(num_batches):
            start = i*b_size
            if start>=data_length:
                print ("Start exceeds max_length!")
                break
            end = start + b_size
            if end>=data_length:
                end = data_length
            
            samples = [reshuffled[start:end] for reshuffled in reshuffled_data]
            batches.append(samples)
                
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
        
    def run_epoch(self, session, data_list, writer, epoch_num):
        n_minibatches, total_loss = 0, 0
        
        
        for samples in self.get_minibatches(data_list, self.num_batches):
            samples_dict = {'post_indices': samples[0], 
                            'comments_indices': samples[1],
                            'comments_labels': samples[2],
                            'encoder_dropout': self.enc_dropout,
                            'decoder_dropout': self.dec_dropout}
            

            feed = self.create_feed_dict(samples_dict)
            merged = tf.summary.merge_all()
            _ , batch_loss, summary = session.run([self.train, self.loss, merged], feed_dict = feed) #self.loss will not run two times. This just fetches the value
            '''
            logits = session.run(self.dec_output, feed_dict = feed) #self.loss will not run two times. This just fetches the value
            print ("logits:", logits[0].shape)
            print ("outputs:", logits[2].shape)
            print ("dec_inputs:", logits[3].shape)
            print ("enc_final_state", logits[4][0].h.shape)
            
            '''
            n_minibatches += 1
            total_loss += batch_loss
            writer.add_summary(summary, epoch_num)
            
        print ("Total loss this epoch is:", total_loss)
        epoch_average_loss = total_loss/n_minibatches
        return epoch_average_loss      
        
    def fit(self, data_list, saver):
        #data_dict: dictionary that holds parameters
        #other_data, other_labels: the validation/test data and labels
        
        losses = []
        start = time.time() # Start time of the entire model
        previous = start
        
        best_loss = 100 # Initial val score
        val_losses = [best_loss]
        
        
        for epoch in range(self.num_epochs):
            
            average_loss = self.run_epoch(self.session, data_list, self.writer, epoch)
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
            if epoch%1==0 or epoch==self.num_epochs-1: #-1 due to characteristics of range
                #This if block just prints out the progress and time taken to reach a certain stage
                previous = self.track_time(start, previous, average_loss, epoch) #Set the new "last checkpoint" to this one
                
        return losses, val_losses #Can try to plot how much the loss has gone down
    
    #=====Functions that generate inferences=====
    
    def generate_inferences(self, post, best_model_location):
        '''
        Function for generating inferences.
        prompt: The prompt for generating a comment
        best_model_location: if you saved the model
        '''
        #saver = tf.train.Saver() #Idk, do we need to initialize variables again? Use the same saver?
        #saver.restore(self.session, best_model_location)
        with tf.name_scope("inference_ops"):
            post_idxs = [self.vocab_to_idx[char] for char in post.lower()]
            end_index = self.vocab_to_idx[self.ending_char]
            
            #Need to have a prompt of >1, or some strange bug will trigger
            data_dict = {'post_indices' : np.array([post_idxs, post_idxs]), 
                         'comments_indices': np.array([[end_index]]), #doesn't matter
                         'comments_labels': np.array([[end_index]]), #doesn't matter
                         'encoder_dropout': [0]*self.num_layers,
                         'decoder_dropout': [0]*self.num_layers}
            
            feed = self.create_feed_dict(data_dict)
            encoder_state = self.encoder_post_op()
            logits, comment_lens = self.session.run(self.decoder_op(encoder_state, False), 
                                                    feed_dict = feed)
            
            softmaxed = tf.nn.softmax(logits[0])
            print (softmaxed)
            most_likely_chars = tf.argmax(softmaxed, axis = 1)
            wanted = self.session.run(most_likely_chars) #Since we duplicated prompt idxes to avoid the weird bug, just get one of them
            
            generated_string = [self.idx_to_vocab[idx] for idx in wanted]
            
            return (logits, "".join(generated_string))