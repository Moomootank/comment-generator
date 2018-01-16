# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 10:36:50 2018

@author: schia
"""

import pandas as pd

import facebook as fb
import urllib.request
import newspaper
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException

class CommentsArticleCrawler():
    def __init__(self, page_name, token):
        '''
        page_name: One crawler for one facebook page for simplicity
        token: facebook comment crawler
        '''
        self.page = page_name
        self.token = token
        self.graph = fb.GraphAPI(token, timeout = 10)
    
    
    #=====Functions that identify articles to collect data from=====
    def filter_posts(self, key_words, string):
        #HELPER function to identify posts that we are interested in
        '''
        page_posts: dict returned by the fb API
        key_words: words to filter the posts by
        string: string to check whether key words are present in
        '''
     
        return any(word in string.lower() for word in key_words)
        
    
    def find_relevant_fb_articles(self, key_words, start_date, end_date, 
                               article_limit):
        #Function to collect article data
        '''
        key_words: Only selected articles which contain these words, otherwise, None
        start_date: Start date to search for articles
        end_date: end date to search for articles
        article_limit: limit to number of articles that the crawler will pull out
        
        Returned:
        post_dict: Dictionary to host facebook posts in        
        article_dict: Dictionary to host the articles in
        '''
        page_posts = self.graph.get_connections(self.page , "posts", 
                                                limit = article_limit, 
                                                since = start_date, 
                                                until = end_date)
        
        page_posts_data = page_posts['data']
        print ("Number of posts: ", len(page_posts_data))
        gathered_post_ids = []       
        post_dict = {}
        article_dict = {}
        driver = webdriver.Firefox() # import os, os.cwd(), put geckodriver in there
        for i in range(len(page_posts_data)):
            try:
                post = page_posts_data[i]
                post_id = post['id']
                attachments = self.graph.get_connections(post_id, "attachments")
                
                attachment_info = attachments['data'][0]
                link_in_post = attachment_info['url']
                title_of_link = attachment_info['title']
                
                if not self.filter_posts(key_words, title_of_link):
                    print ("Post is irrelevant to search")
                    print (title_of_link)
                    continue
                
                driver.get(link_in_post)
                url = driver.current_url #This is the javascript redirected url
                
                if "facebook.com" in url:
                    print ("This post likely does not link to an article")
                    continue
                
                #driver.close() #Close the browser to avoid getting spammed

                article = newspaper.Article(url, "en")
                article.download()
                article.parse()
                
                #=====Start adding stuff to the dictionaries=====
                post_dict[post_id] = [article.title, post['created_time'], url, 
                                      post['message']]
                
                article_dict[post_id] = article               
                gathered_post_ids.append(post_id)
                
            except KeyError:
                print ("This article has no attachments or url on facebook")
                continue
            except TimeoutException:
                print ("Time out exception has been thrown. Just go to next article, since we don't care about any particular article.")
                continue
            except UnicodeEncodeError:
                print ("UnicodeEncodeError thrown. Just go to the next one lol, too many possible reasons why")
                continue
            except IndexError:
                print ("Index Error. Happens very rarely. Continue.")
                continue
                
        print ("{num} articles successfully processesed".format(num = 
               len(gathered_post_ids)))
        driver.quit()
        return post_dict, article_dict, gathered_post_ids
    
    #=====If we don't care about the linked article, just want all posts, use this:=====
    def collect_fb_posts(self, start_date, end_date, post_limit):
        page_posts = self.graph.get_connections(self.page , "posts", 
                                                limit = post_limit, 
                                                since = start_date, 
                                                until = end_date)
        
        page_posts_data = page_posts['data']
        print ("Number of posts: ", len(page_posts_data))
        gathered_post_ids = []  
        post_dict = {}
        
        for i in range(len(page_posts_data)):
            try:
                post = page_posts_data[i]
                post_id = post['id']
                
                post_dict[post_id] = [post['message'], post['created_time']]
                gathered_post_ids.append(post_id)
                
            except KeyError:
                print ("This post has no message on facebook")
                continue
            
        return post_dict, gathered_post_ids
    
    #=====Functions that collect comments data=====
    def collect_comments_for_article(self, post_ids, comments_limit):
        
        comments_dict = {}
        for post_id in post_ids:
            comments = self.graph.get_connections(post_id, "comments", 
                                                  limit = comments_limit)
            
            comments_data = comments['data']
            
            for comment in comments_data:
                '''
                Would love to have commenter_id as well, but seems
                unnecessary, and a little too tedious to set up
                '''
                comment_id = comment['id']
                comments_dict[comment_id] = [post_id, comment['message']]
            
            
        return comments_dict
            
        
                
        

        
        
        
        
        
        