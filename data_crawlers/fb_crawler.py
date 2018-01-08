# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 18:37:38 2017

@author: Moomootank
"""

import facebook as fb
import urllib.request
import newspaper
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException

from bs4 import BeautifulSoup

def save_raw_html_files(graph, page_id, num_posts, html_save_loc):
    #Unrelated to project. Saving raw html files for TT's work
    page_posts = graph.get_connections(page_id , "posts", limit = num_posts)
    page_posts_data = page_posts['data']
    
    article_index = 0
    success = 0
    driver = webdriver.Firefox()
    driver.set_page_load_timeout(30)
    
    
    for post in page_posts_data:
        try:
            print ("Examining article {n}".format(n = article_index))
            attachments = graph.get_connections(post['id'], "attachments")
            link_in_post = attachments['data'][0]['url']
                       
            driver.get(link_in_post)
            redirected_url  = driver.current_url #Javascript redirect smh

            
            if "facebook.com" in redirected_url:
                print ("Redirect failed - probably not an article")                
                continue
            
            
            article = newspaper.Article(redirected_url, "en")
            article.download()
            
            html_file = article.html
            #final_save_loc = html_save_loc + str(article_index) + ".html"
            final_save_loc = html_save_loc + post['id'] + ".html"
            
            f = open(final_save_loc,'w')
            f.write(html_file)
            f.close()

            article_index = article_index + 1
            success = success + 1
        except KeyError:
            print ("This article has no attachments or url on facebook")
            continue
        except TimeoutException:
            print ("Time out exception has been thrown. Just go to next article, since we don't care about any particular article.")
            continue
        except UnicodeEncodeError:
            print ("UnicodeEncodeError thrown. Just go to the next one lol, too many possible reasons why")
            continue
    
    print ("Number of successful html files downloaded: {n}".format(n = success))
    driver.close()
    return
    
if __name__ == "__main__":
    #My apps user-access token. I didn't set the permissions for some of them though. Hope that's fine
    token = "INSERT TOKEN HERE"
    graph = fb.GraphAPI(token, timeout = 10)
     
    #====Testing to see if this works=====
    page_id = "thestraitstimes"
    save_loc = r"D:\Data Science\Projects\commenter_simulator\data_crawlers\saved_html_files\thestraitstimes\st_article_"
    #save_raw_html_files(graph, page_id, 100, save_loc)
    
    page_posts = graph.get_connections(page_id , "posts", limit = 100, since = "2017-12-22", until = "2017-12-23")
    page_posts_data = page_posts['data']
    
    attachments = graph.get_connections(page_posts_data[1]['id'], "attachments")
    link_in_post = attachments['data'][0]['url']
    
    driver = webdriver.Firefox() # import os, os.cwd(), put geckodriver in there
    driver.get(link_in_post)
    url = driver.current_url
    driver.close()

    article = newspaper.Article(url, "en")
    article.download()
    article.parse()
    
    


    

    