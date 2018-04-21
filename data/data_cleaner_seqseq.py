# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 15:58:45 2018

@author: Moomootank
"""
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

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
    
def remove_non_english_chars(s):
    #removes non-english chars from the string
    return s.encode(encoding = 'utf-8').decode('ascii', 'ignore')
    
def find_all_unique_characters(df, column):
    """
    df: dataframe of comments data
    column: column name of column in dataframe 
    returns a series
    """
    all_text = "".join(df[column].values) #This will be a string
    character_counts = pd.Series(list(all_text)).value_counts()
    
    return character_counts

def save_obj(obj, location):
    with open(location, 'wb') as file:
        pickle.dump(obj, file)

def replace_rare_characters(df, column, to_not_replace, replace_with, new_col_name):
    '''
    Function that replaces fairly rare characters in the text, decreasing feature space
    to_not_replace: characters to NOT replace. replace_with: what to replace them with. 
    Chose the negative as it avoids a lot of funny errors
    '''
    
    replace_this = "[^{pat}]".format(pat = to_not_replace)
    #print(replace_this)
    df[new_col_name] = df[column].str.replace(replace_this, replace_with)
    
    return df

#=====

def clean_and_obtain_chars(df, column, replace_with, new_col_name, cutoff, wanted_characters = None):
    '''
    Wrapper function that replaces fairly rare characters in the text.
    Also returns dictionary of wanted (common) characters
    Arguments are the same as replace_rare_characters
    '''
    character_counts = find_all_unique_characters(df, column)
    if wanted_characters is None:
        #wanted_characters can be a pd.series that you already want to use
        wanted_characters = character_counts.loc[character_counts > cutoff] #Completely arbitrary
    
    wanted_characters_keys = "".join(wanted_characters.sort_index().keys())        
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

def break_comments(indices, lengths, cutoff):
    '''
    Seperates indices into two components: the prompt, and the tbg (to be generated)
    '''
    to_keep = lengths >= cutoff + 1 #We want tbg to be at least 1
    print ("{n} comments meet the required length".format(n = to_keep.sum()))
    
    kept_indices = indices[to_keep]
    prompts = kept_indices[:, :cutoff]
    tbg = kept_indices[:, cutoff:]
    
    return prompts, tbg, to_keep

#=====Function for combining posts and comments
    
def combine_posts_and_prompts(post_idxs, prompt_idxs, SEP_CHAR, END_CHAR, char_dict,
                              WANTED_LENGTH):
    #Function for combining posts and prompts
    num_posts = post_idxs.shape[0]
    indices = []
    lengths = []
    for i in range(num_posts):
        if i%10000==0:
            print ("Completed:" , i)
        post = post_idxs[i]
        prompt = prompt_idxs[i]
        post.append(char_dict[SEP_CHAR]) #append the separation character
        post.extend(prompt)
        extended, length = add_end_char_tail(post, WANTED_LENGTH, END_CHAR, char_dict)
        indices.append(extended)
        lengths.append(length)
    
    return np.array(indices), np.array(lengths)
        

if __name__ == "__main__":
    #=====Load the raw data=====
    raw_posts_url = r"raw_data/posts_2017_dt.csv"
    raw_posts = pd.read_csv(raw_posts_url, index_col = 0, encoding = "windows-1252")
    
    raw_comments_url = r"raw_data/comments_2017_dt.csv"
    raw_comments = pd.read_csv(raw_comments_url, index_col = 0, encoding = "windows-1252")
    raw_comments['comment_text'] = raw_comments['comment_text'].str.lower() #Removes caps for now
    
    #=====Creates indices from comments=====
    is_english = raw_comments['comment_text'].apply(lambda x: isEnglish(x))
    raw_comments_subset = raw_comments.loc[is_english] #Removes entries with non-standard characters
    clean_comments, wanted_chars = clean_and_obtain_chars(raw_comments_subset, "comment_text",
                                                          "", "cleaned_text", 20000)
    
    #=====Create embedding matrices=====
    END_TOKEN = "~"
    wanted_chars[END_TOKEN] = 0 #Add an end token into the list
    START_TOKEN = "*"
    wanted_chars[START_TOKEN] = 0
    SEP_TOKEN = "|"
    wanted_chars[SEP_TOKEN] = 0
    
    char_idx_dict, idx_char_dict = create_mapping_dict(wanted_chars)
    char_array = create_char_array(wanted_chars)
    
    MAX_SEQ_LEN = 350
    #Indices are for the clean comments: Not spliced yet!
    indices, str_lengths = map_characters_to_indices(clean_comments, "cleaned_text", 
                                                     char_idx_dict, MAX_SEQ_LEN, 
                                                     END_TOKEN, True)
    
    prompts, tbg, to_keep = break_comments(indices, str_lengths, 50)
    tbg = np.insert(tbg, 0, char_idx_dict[START_TOKEN], axis = 1 )[:, :-1] #Keep len at 300
    tbg_labels = tbg[:, 1:]
    tbg_labels = np.insert(tbg_labels, tbg_labels.shape[1], char_idx_dict[END_TOKEN], axis = 1 )
    
    #=====Creates indices from posts=====
    raw_posts['post_text'] = raw_posts['post_text'].str.lower().apply(lambda x: 
        remove_non_english_chars(x)) #Remove non-english chars from posts
    
    clean_posts, chars_in_posts = clean_and_obtain_chars(raw_posts, "post_text",
                                                          "", "cleaned_posts", None, wanted_chars)
    
    matched_posts = clean_posts.loc[clean_comments['post_id'].values] 
    subsetted_posts = matched_posts[to_keep]
    assert subsetted_posts.shape[0]==prompts.shape[0], "Len_posts does not match len_comments"
        
    POST_SEQ_LEN = 250
    post_indices, post_lengths = map_characters_to_indices(subsetted_posts, "cleaned_posts", 
                                                           char_idx_dict, POST_SEQ_LEN, 
                                                           END_TOKEN, extend = False)
    
    print("Starting combination")
    combined, combined_lengths = combine_posts_and_prompts(post_indices, 
                                                           prompts, SEP_TOKEN,
                                                           END_TOKEN, char_idx_dict,
                                                           300)
    
    save_obj(tbg, r"data_for_seqseq/dual_enc_max_300/tbg_indices.pickle")
    save_obj(tbg_labels, r"data_for_seqseq/dual_enc_max_300/tbg_labels.pickle")
    save_obj(char_array, r"data_for_seqseq/dual_enc_max_300/char_embeddings_df.pickle")
    save_obj(combined, r"data_for_seqseq/dual_enc_max_300/post_indices.pickle")
    save_obj(char_idx_dict, r"data_for_seqseq/dual_enc_max_300/char_idx_dict.pickle")
    save_obj(idx_char_dict, r"data_for_seqseq/dual_enc_max_300/idx_char_dict.pickle")
    
    #=====Create TFRecord file for easier loading=====
    train_filename = r"data_for_seqseq/dual_enc_max_300/train_data.tfrecords"
    writer = tf.python_io.TFRecordWriter(train_filename)
    
    for i in range(combined.shape[0]):
        post = tf.train.Feature(int64_list=tf.train.Int64List(value= combined[i]))
        c = tf.train.Feature(int64_list=tf.train.Int64List(value= tbg[i]))
        c_label = tf.train.Feature(int64_list=tf.train.Int64List(value= tbg_labels[i]))
        feature = {'post': post, 'comment': c, 'label': c_label}
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
    
    writer.close()
    