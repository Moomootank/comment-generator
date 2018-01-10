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

def crawl_month_for_articles(crawler, year, month, day_start, day_end, 
                             post_limit, comment_limit, key_words):
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
        crawler.find_relevant_fb_articles(key_words, 
                                       start_day, end_day, post_limit)
        
        comments_dict = crawler.collect_comments_for_article(gathered_post_ids, 
                                                             post_limit)
        
        posts.append(post_dict)
        posts.append(article_dict)
        posts.append(comments_dict)
    
    return posts, articles, comments

def crawl_month_for_posts(crawler, start_dates, end_dates, post_limit, comment_limit):
    #Crawls through month for post messages and comments (no articles!)
    '''
    crawler: The crawler object
    start_dates: array, each index is the start date of a month you want to look at
    end_dates: array, each index is the end date of a period. Make sure the indices align with start_dates
    post_limit: Limit of number of posts per time perido you want to look at
    comment_limit: Limit of number of comments per post you want to look at
    '''
    
    assert len(start_dates)==len(end_dates), "Lengths of date arrays are not equal"
    post_array = []
    comments_array = []
    
    for i in range(len(start_dates)):
        start_date = start_dates[i]
        end_date = end_dates[i]
        print ("Gathering for period that starts on: ", start_date)
        
        post_dict, gathered_post_ids = crawler.collect_fb_posts(
                                    start_date, end_date, post_limit)
        
        comments_dict = crawler.collect_comments_for_article(gathered_post_ids,
                                                             comment_limit)
        
        post_array.append(post_dict)
        comments_array.append(comments_dict)
        
    return post_array, comments_array
        
        
    
if __name__ == "__main__":
    #My apps user-access token. I didn't set the permissions for some of them though. Hope that's fine
    token = "INSERT TOKEN HERE"
    graph = fb.GraphAPI(token, timeout = 10)
     
    #====Testing to see if this works=====
    page_id = "INSERT PAGE HERE"
    
    crawler = CommentsArticleCrawler(page_id, token)    
    start_dates = ["2017-{mon}-01".format(mon=i) for i in range(1,13)]
    end_dates = ["2017-{mon}-01".format(mon=i) for i in range(2,13)]
    end_dates.append("2018-01-01")
    
    post_array, comments_array = \
    crawl_month_for_posts(crawler, start_dates[:3], end_dates[:3], 100, 1000)
    


    

    