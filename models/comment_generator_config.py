# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 09:16:58 2018

@author: Moomootank
"""

class CommentGeneratorConfig():
    def __init__(self, params_dict):
        #Params dict is a dictionary with all the necessary parameters
        #self.embed_size = params_dict['embed_size']
        self.num_classes = params_dict['num_classes']
        self.num_batches  = params_dict['num_batches']
        
        self.embeddings = params_dict['embeddings'] #Just need to have one copy of all the embeddings
        self.vocab_idx_dict = params_dict['vocab_idx_dict']
        self.idx_vocab_dict = params_dict['idx_vocab_dict']
        self.ending_char = params_dict['ending_char']#Char that means sequence has ended
        self.max_time = params_dict['max_time']
        
        self.num_epochs = params_dict['num_epochs']
        self.learning_rate = params_dict['learning_rate']
        self.num_layers = params_dict['num_layers']
        self.log_location = params_dict['log_location'] #Where you want to save the intermediate models
        self.chkpt = params_dict['chkpt'] #Where you want to save checkpoints
        
        #Additional seqseq params
        keys = list(params_dict.keys())
        if "starting_char" in keys:
            self.starting_char = params_dict["starting_char"]
        if "num_enc_layers" in keys:
            self.num_enc_layers = params_dict["num_enc_layers"]