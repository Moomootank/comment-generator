# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 09:50:52 2018

@author: schia

Cleans data for feeding into the generator
In reality, this would be best done by creating a class that cleans data
However, as much of this data cleaning process was experimental for me,
I decided to write the code in as scripting format to make exploration easier.5
"""
import numpy as np
import pandas as pd
import pickle
import copy

#=====Functions that tease out attributes of the data=====

def isEnglish(s):
    #Identify anything that isn't latin characters
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    except AttributeError:
        return False
    else:
        return True

def find_all_unique_characters(df, column):
    """
    df: dataframe of comments data
    column: column name of column in dataframe 
    """
    all_text = "".join(df[column].values) #This will be a string
    character_counts = pd.Series(list(all_text)).value_counts()
    
    return character_counts

def save_obj(obj, location):
    with open(location, 'wb') as file:
        pickle.dump(obj, file)

#=====Functions that change attributes of the data=====

def remove_non_english_chars(s):
    return s.encode(encoding = 'utf-8').decode('ascii', 'ignore')

def replace_rare_characters(df, column, to_replace, replace_with, new_col_name):
    '''
    Function that replaces fairly rare characters in the text, decreasing feature space
    to_replace: characters to NOT replace. replace_with: what to replace them with. 
    Chose the negative as it avoids a lot of funny errors
    '''
    
    replace_this = "[^{pat}]".format(pat = to_replace)
    #print(replace_this)
    df[new_col_name] = df[column].str.replace(replace_this, replace_with)
    
    return df

def clean_and_obtain_chars(df, column, replace_with, new_col_name, wanted_characters = None):
    '''
    Wrapper function that replaces fairly rare characters in the text.
    Also returns dictionary of wanted (common) characters
    Arguments are the same as replace_rare_characters
    '''
    character_counts = find_all_unique_characters(df, column)
    if wanted_characters is None:
        wanted_characters = character_counts.loc[character_counts > 20000] #Completely arbitrary
    
    wanted_characters_keys = "".join(wanted_characters.sort_index().keys())
    #print (wanted_characters)
        
    clean_comments = replace_rare_characters(df, column,
                            wanted_characters_keys, replace_with, new_col_name)
    
    wanted_chars = find_all_unique_characters(clean_comments, new_col_name)
    
    return clean_comments, wanted_chars

#=====Functions that help to generate the character embeddings=====
    
def create_mapping_dict(wanted_chars):
    '''
    Creates mapping dicts of characters to indices and indices to characters
    Also adds in an index for the end token
    wanted_chars: Series of wanted characters IN DESIRED INDEX ORDER
    '''
    char_list = list(wanted_chars.keys())
    char_idx_dict = {char_list[i]:i for i in range(len(char_list))}    
    idx_char_dict = {v:k for k,v in char_idx_dict.items()}
    
    return char_idx_dict, idx_char_dict
    
def create_char_array(wanted_chars):
    '''
    Creates one hot array of characters
    wanted_chars: Dictionary of key=char, value = frequency generated from clean_and_obtain_chars
    '''
    char_array = pd.get_dummies(pd.Series(wanted_chars.keys()))
    return char_array[wanted_chars.keys()] #Make sure the ordering is stable

def add_end_char_tail(indexed, SEQ_LEN, END_TOKEN, char_dict):
    #Adds string of end chars till the end
    string_length = len(indexed)
    if string_length<SEQ_LEN:
        difference = SEQ_LEN - string_length
        end_list = [char_dict[END_TOKEN] for i in range(difference)]
        indexed.extend(end_list)
        assert len(indexed)==SEQ_LEN, "Index array is not of the required length"
    
    return indexed, string_length

def map_chars(string, char_dict, SEQ_LEN, END_TOKEN, extend = False):
    '''
    Maps characters to their respective indices
    string: string to map.
    char_dict: character to index dict
    seq_len: maximum sequence length that we want
    '''
    EFF_SEQ_LEN = SEQ_LEN - 1 #-1 as last place is for end token
    
    if len(string)>EFF_SEQ_LEN:
        desired_str = string[:EFF_SEQ_LEN] #Just ignore everything after this
    else:
        desired_str = string
        
    indexed = [char_dict[char] for char in desired_str]
    #indexed.append(char_dict[END_TOKEN]) #appends the end_token
    
    #Append end tokens until we get SEQ_LEN. Makes storing easier
    string_length = len(indexed)
    if extend:
        indexed, string_length = add_end_char_tail(indexed, SEQ_LEN, END_TOKEN, char_dict)
        
    return indexed, string_length

def map_characters_to_indices(df, column_to_map, char_dict, SEQ_LEN, END_TOKEN, extend = False):
    '''
    Function that creates np matrix of character index mappings
    df: comment_df
    column_to_map: column of text that you want to do mappings from
    char_dict: char to index dict
    '''
    #For tomorrow. Basically just truncate the sequences by seq_len. then create array of index mappings
    
    items = df[column_to_map].apply(lambda x: map_chars(x, char_dict, SEQ_LEN, END_TOKEN, extend))
    #Items is a series. the values are an array where idx 0 = the indexed str, 1 = the length
    indices = [i[0] for i in items.values]
    lengths = [i[1] for i in items.values]
    
    return np.array(indices), np.array(lengths)

#=====Additional functions for preparing seq2seq data=====
def add_additional_characters(sep_char, start_char, char_idx_dict, idx_char_dict):
    POST_COM_SEPERATOR_IDX = len(char_idx_dict)
    START_CHAR_IDX = POST_COM_SEPERATOR_IDX + 1
    
    cid = copy.deepcopy(char_idx_dict)
    icd = copy.deepcopy(idx_char_dict)
    
    cid[sep_char] = POST_COM_SEPERATOR_IDX
    icd[POST_COM_SEPERATOR_IDX] = sep_char   
    cid[start_char] = START_CHAR_IDX
    icd[START_CHAR_IDX] = start_char
    
    return cid, icd

def create_seqseq_indices(post_idxs, comment_idxs, post_lengths, comment_lengths, param_dict, cutoff):
    '''
    Converts post text into idx squence, appending cutoff worth of comments at the end
    '''
    
    num_rows = len(post_idxs)
    assert num_rows==len(comment_idxs), "Arrays must be of equal lengths!"
    SEQ_LEN = param_dict["seq_len"]
    char_dict = param_dict["char_idx_dict"]
    END_TOKEN = param_dict["end_token"]
    SEP_TOKEN = param_dict["sep_token"]
    comment_cutoff = param_dict["comment_cutoff"]
    indices = []
    lengths = []
    for i in range(num_rows):
        post = post_idxs[i]
        comment = comment_idxs[i][:cutoff]
        post_len = post_lengths[i]
        comment_len = comment_lengths[i]
        if comment_len<comment_cutoff:
            #since the entire comment will be shifted to encoder
            comment_len = comment_len
        else:
            comment_len = comment_cutoff
        
        post.append(char_dict[SEP_TOKEN]) #Append sep token
        post.extend(comment) #Get the first 10 of comments
        indexed, WRONG_LEN = add_end_char_tail(post, SEQ_LEN, END_TOKEN, char_dict)
        #wrong_len as :10 of the comment could include a bunch of end_chars
        correct_len = post_len + 1 + comment_len #Post_len + sep_char + up to 10 characters
        indices.append(indexed)
        lengths.append(correct_len)
        
    return np.array(indices), np.array(lengths)

def post_process_seqseq_data(post_indices, post_lens, comment_indices, comment_lens, cutoff,
                             start_char_idx):
    '''
    Function for modifying the data for seqseq to make sure lens are correct
    '''
    len_mask = comment_lens>cutoff #Just completely remove those below certain length
    wanted_comments = comment_indices[len_mask]
    wanted_comments = wanted_comments[:, cutoff:] #just want from that point onwards 
    print (wanted_comments.shape)
    wanted_comments = np.insert(wanted_comments, 0 , start_char_idx, axis = 1)
    print (wanted_comments.shape)
    wanted_comments = wanted_comments[:, :-1]
    print (wanted_comments.shape)
    
    wanted_comment_lens = comment_lens[len_mask]
    wanted_comment_lens = wanted_comment_lens - cutoff #since portion of it went to indices
    
    wanted_post_indices = post_indices[len_mask] #remove corresponding 
    wanted_post_lens = post_lens[len_mask] #Lengths should already be correct from create_seqseq_indices
    
    return wanted_post_indices, wanted_post_lens, wanted_comments, wanted_comment_lens
    
        
if __name__ == "__main__":
    #=====Load the files=====
    raw_posts_url = r"raw_data/posts_2017_dt.csv"
    raw_posts = pd.read_csv(raw_posts_url, index_col = 0, encoding = "windows-1252")
    
    raw_comments_url = r"raw_data/comments_2017_dt.csv"
    raw_comments = pd.read_csv(raw_comments_url, index_col = 0, encoding = "windows-1252")
    raw_comments['comment_text'] = raw_comments['comment_text'].str.lower() #Removes caps for now
    
    #=====Replace rare-characters in the text with nothing=====
    is_english = raw_comments['comment_text'].apply(lambda x: isEnglish(x))
    raw_comments_subset = raw_comments.loc[is_english] #Removes entries with non-standard characters
    clean_comments, wanted_chars = clean_and_obtain_chars(raw_comments_subset, "comment_text",
                                                          "", "cleaned_text")
    #=====Create embedding matrices=====
    END_TOKEN = "~"
    wanted_chars[END_TOKEN] = 0 #Add an end token into the list
    char_idx_dict, idx_char_dict = create_mapping_dict(wanted_chars)
    char_array = create_char_array(wanted_chars)
    
    SEQ_LEN = 350 #maximum sequence length 
    indices, str_lengths = map_characters_to_indices(clean_comments, "cleaned_text", char_idx_dict,
                                                        SEQ_LEN, END_TOKEN, True)
    
    #save_obj(char_idx_dict, r"data_for_gen_model/max_len_350/char_idx_dict.pickle")
    #save_obj(idx_char_dict, r"data_for_gen_model/max_len_350/idx_char_dict.pickle")
    #save_obj(char_array, r"data_for_gen_model/max_len_350/char_embeddings_df.pickle")
    #save_obj(indices, r"data_for_gen_model/max_len_350/comment_indices_350.pickle")
    #save_obj(str_lengths, r"data_for_gen_model/max_len_350/str_lengths_350.pickle")
    
    #---------------------------------------------------------------------
    #=====Creating data for seq2seq model=====
    #Apply same transformations to post data
    raw_posts['post_text'] = raw_posts['post_text'].str.lower()
    raw_posts['post_text'] = raw_posts['post_text'].apply(lambda x: remove_non_english_chars(x))
    
    english_posts = raw_posts['post_text'].apply(lambda post: isEnglish(post))
    raw_posts_subset = raw_posts.loc[english_posts]
    #chars_in_posts is used just as a reference. Not really part of code
    clean_posts, chars_in_posts = clean_and_obtain_chars(raw_posts_subset, "post_text",
                                                          "", "cleaned_posts", wanted_chars)
    
    #Add additional tokens for seqseq data
    SEP_TOKEN = "|" 
    START_CHAR = "*"
    seqseq_char_idx_dict, seqseq_idx_char_dict = add_additional_characters(SEP_TOKEN, 
                                                                           START_CHAR, 
                                                                           char_idx_dict, 
                                                                           idx_char_dict)
    #Create the seqseq indices
    POST_SEQ_LEN = 339
    matched_posts = clean_posts.loc[clean_comments['post_id'].values] 

    post_indices, post_lengths = map_characters_to_indices(matched_posts, "cleaned_posts", 
                                                          seqseq_char_idx_dict, POST_SEQ_LEN, 
                                                          END_TOKEN, extend = False)
    
    param_dict = {"seq_len": SEQ_LEN,
                  "char_idx_dict": seqseq_char_idx_dict,
                  "end_token": END_TOKEN,
                  "sep_token": SEP_TOKEN,
                  "comment_cutoff": 10
            }
    
    seqseq_indices, seqseq_lens = create_seqseq_indices(post_indices, indices, 
                                                          post_lengths, str_lengths, param_dict,
                                                          param_dict['comment_cutoff'])
    
    p_idx, p_lens, c_idx, c_lens = post_process_seqseq_data(seqseq_indices, 
                                                            seqseq_lens, indices,
                                                            str_lengths, param_dict['comment_cutoff'],
                                                            seqseq_char_idx_dict[START_CHAR])
    
    #Don't need lengths; just have it to debug
    c_labels = c_idx[:, 1:]
    c_labels = np.insert(c_labels, c_labels.shape[1], seqseq_char_idx_dict[END_TOKEN], axis = 1 )
    
    seqseq_char_array = create_char_array(pd.Series(seqseq_char_idx_dict).sort_values())
    save_obj(p_idx, r"data_for_seqseq/max_len_350/post_indices_350.pickle")
    save_obj(p_lens, r"data_for_seqseq/max_len_350/post_lengths_350.pickle")
    save_obj(c_idx, r"data_for_seqseq/max_len_350/comment_indices_350.pickle")
    save_obj(c_lens, r"data_for_seqseq/max_len_350/comment_lengths_350.pickle")
    save_obj(seqseq_char_array, r"data_for_seqseq/max_len_350/char_embeddings_df.pickle")
    save_obj(seqseq_char_idx_dict, r"data_for_seqseq/max_len_350/char_idx_dict.pickle")
    save_obj(seqseq_idx_char_dict, r"data_for_seqseq/max_len_350/idx_char_dict.pickle")
    save_obj(c_labels, r"data_for_seqseq/max_len_350/comment_labels_350.pickle")
    
    test = 6824
    print ("".join(seqseq_idx_char_dict[i] for i in p_idx[test]))
    print ("".join(seqseq_idx_char_dict[i] for i in c_idx[test]))
    