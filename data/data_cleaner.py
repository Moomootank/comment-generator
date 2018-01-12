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
    

if __name__ == "__main__":
    raw_posts_url = r"raw_data/posts_2017_dt.csv"
    raw_posts = pd.read_csv(raw_posts_url, index_col = 0, encoding = "windows-1252")
    
    raw_comments_url = r"raw_data/comments_2017_dt.csv"
    raw_comments = pd.read_csv(raw_comments_url, index_col = 0, encoding = "windows-1252")
    raw_comments['comment_text'] = raw_comments['comment_text'].str.lower()
    
    is_english = raw_comments['comment_text'].apply(lambda x: isEnglish(x))
    raw_comments_subset = raw_comments.loc[is_english]
    
    character_counts = find_all_unique_characters(raw_comments_subset, "comment_text")
    wanted_characters = character_counts.loc[character_counts > 20000] #Completely arbitrary
    wanted_characters_keys = "".join(wanted_characters.sort_index().keys())
    
    test_df = replace_rare_characters(raw_comments_subset, "comment_text",
                            wanted_characters_keys, "", "replaced_text")
    
    test = find_all_unique_characters(test_df, "replaced_text")
    