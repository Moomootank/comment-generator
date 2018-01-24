# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 09:50:52 2018

@author: schia

Cleans data for feeding into the generator
"""
import numpy as np
import pandas as pd

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

#=====Functions that change attributes of the data=====

def replace_rare_characters(df, column, to_replace, replace_with, new_col_name):
    '''
    Function that replaces fairly rare characters in the text, decreasing feature space
    to_replace: characters to NOT replace. replace_with: what to replace them with. 
    Chose the negative as it avoids a lot of funny errors
    '''
    
    replace_this = "[^{pat}]".format(pat = to_replace)
    print(replace_this)
    df[new_col_name] = df[column].str.replace(replace_this, replace_with)
    
    return df

def clean_and_obtain_chars(df, column, replace_with, new_col_name):
    '''
    Wrapper function that replaces fairly rare characters in the text.
    Also returns dictionary of wanted (common) characters
    Arguments are the same as replace_rare_characters
    '''
    character_counts = find_all_unique_characters(df, column)
    wanted_characters = character_counts.loc[character_counts > 20000] #Completely arbitrary
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

def map_chars(string, char_dict, SEQ_LEN, END_TOKEN):
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
    indexed.append(char_dict[END_TOKEN]) #appends the end_token
    
    #Append end tokens until we get SEQ_LEN. Makes storing easier
    string_length = len(indexed)
    if string_length<SEQ_LEN:
        difference = SEQ_LEN - string_length
        end_list = [char_dict[END_TOKEN] for i in range(difference)]
        indexed.extend(end_list)
        assert len(indexed)==SEQ_LEN, "Index array is not of the required length"
        
    return indexed, string_length

def map_characters_to_indices(df, column_to_map, char_dict, SEQ_LEN, END_TOKEN):
    '''
    Function that creates np matrix of character index mappings
    df: comment_df
    column_to_map: column of text that you want to do mappings from
    char_dict: char to index dict
    '''
    #For tomorrow. Basically just truncate the sequences by seq_len. then create array of index mappings
    
    items = df[column_to_map].apply(lambda x: map_chars(x, char_dict, SEQ_LEN, END_TOKEN))
    #Items is a series. the values are an array where idx 0 = the indexed str, 1 = the length
    indices = [i[0] for i in items.values]
    lengths = [i[1] for i in items.values]
    
    return np.array(indices), np.array(lengths)
    
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
                                                        SEQ_LEN, END_TOKEN)
    
    
    