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

from article_and_comments_crawler import CommentsArticleCrawler

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

#=====These are functions that actually matter for the project=====

def crawl_through_month(crawler, year, month, day_start, day_end, post_limit, key_words):
    '''
    year: a string with the year (eg. 2017)
    month: string in the format "01, 02, 12" etc.
    day_end: int saying the last day you want to get stuff from
    
    post_limit: int of maximum posts you want to take per session
    key_words: key words to filter posts by
    '''
    
    #for now
    posts = []
    articles = []
    comments = []
    
    for i in range(day_start, day_end + 1):
        start_day = "{year}-{month}-{day}".format(year = year, month = month,
                     day = i)
        end_day = "{year}-{month}-{day}".format(year = year, month = month,
                     day = i + 1)
        
        print (start_day)
        print (end_day)
        
        post_dict, article_dict, gathered_post_ids = \
        crawler.find_relevant_fb_posts(key_words, 
                                       start_day, end_day, post_limit)
        
        comments_dict = crawler.collect_comments_for_article(gathered_post_ids, 
                                                             post_limit)
        
        posts.append(post_dict)
        posts.append(article_dict)
        posts.append(comments_dict)
    
    return posts, articles, comments
    
if __name__ == "__main__":
    #My apps user-access token. I didn't set the permissions for some of them though. Hope that's fine
    token = "EAACVbwskeJEBAJqq0YBZBigWbL4ogo1lTsjw1sSIXypZBMMlRcVNtXHKZBsBnbuBwMs74giHjdrUGONLtuemYjDixDuqrkK1XP4aoNkHZB4QIZCgljpVZBR7blPLv1BcK8H0Y09iw9jHGNq1s8ZCuJ2H1aZAk1kJkZAv6KA5OmKZBNwgZDZD"
    graph = fb.GraphAPI(token, timeout = 10)
     
    #====Testing to see if this works=====
    page_id = "nytimes"
    
    crawler = CommentsArticleCrawler(page_id, token)
    
    post_dict, article_dict, gathered_post_ids = \
    crawler.find_relevant_fb_posts(["trump"], "2017-11-01", "2017-11-15", 100)
    comments_dict = crawler.collect_comments_for_article(gathered_post_ids, 100)
    
    
    '''
    posts, articles, comments = crawl_through_month(crawler, 2018, "01", 1, 9, 
                                                    100, ["trump"])
    
    '''


    

    