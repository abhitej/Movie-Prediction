# -*- coding: utf-8 -*-
"""

@author: Abhitej Kodali
"""
from bs4 import BeautifulSoup
import time, os, requests
from selenium import webdriver
from selenium.common.exceptions import ElementNotVisibleException
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import StaleElementReferenceException
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from fake_useragent import UserAgent
import pandas as pd

def click_more(driver):
    while True:
        try:
            elem = driver.find_element_by_xpath('//*[@id="show-more-btn"]/button')
            driver.execute_script("arguments[0].click();",elem)
        except ElementNotVisibleException:
            break
        except NoSuchElementException:
            break
        except StaleElementReferenceException:
            break
    return driver
    
def movie_links(url,movielist):
    ua=UserAgent()
    dcap = dict(DesiredCapabilities.PHANTOMJS)
    dcap["phantomjs.page.settings.userAgent"] = (ua.random)
    service_args=['--ssl-protocol=any','--ignore-ssl-errors=true']
    driver = webdriver.Chrome('chromedriver.exe',desired_capabilities=dcap,service_args=service_args)
    driver.get(url)
    driver = click_more(driver)
    soup = BeautifulSoup(driver.page_source,'lxml')
    movies = soup.find_all('div',{'class':'movie_info'})
    
    for movie in movies:
        u = movie.find('a',href=True)['href']
        t = movie.find('h3',{'class' : 'movieTitle'}).text
        movielist.append((t,u))
    
    driver.close()
    return movielist

def scrape_movies(df):
    
    mainurl = "https://www.rottentomatoes.com"
    remove_punctuation_map = dict((ord(char), None) for char in '\/*?:"<>|')
    if not os.path.exists('Page Links/'):os.mkdir('Page Links/')
    
    for url in df['url']:
        movie = str(df.loc[df['url']==url,'Movie'].iloc[0]).translate(remove_punctuation_map)
        FullPath=os.path.join(os.getcwd() + '\Page Links',movie+".html")
        if not os.path.isfile(FullPath):
            movie_url = mainurl + url
            html=requests.get(movie_url)
            content = html.text.encode('utf-8','ignore')
            with open(FullPath,'wb') as fw: 
                fw.write(content)
            fw.close()
    
    return True

if __name__ =='__main__':
    list_movies = []
    d = {1:'Action', 2:'Animation', 4:'Art & Foreign', 5:'Classics', 6:'Comedy', 8:'Documentary', 9:'Drama',
         10:'Horror', 11:'Kids & Family', 13:'Mystery',  14:'Sci-fi & Fantasy', 18:'Romance'}
    start = time.time()
    for i in d:
        url = "https://www.rottentomatoes.com/browse/dvd-streaming-all?minTomato=0&maxTomato=100&services=amazon;hbo_go;itunes;netflix_iw;vudu;amazon_prime;fandango_now&genres="+str(i)+"&sortBy=release"
        list_movies = movie_links(url,list_movies)
        print("Genre done: ",str(d[i]))
    
    movie_list = pd.DataFrame(list_movies, columns=['Movie','url'])
    movie_list = movie_list.drop_duplicates().reset_index(drop=True)
    movie_list.to_csv("Movielist.csv", index = False)
    end1 = time.time() - start
    p = scrape_movies(movie_list)
    end2 = time.time() - end1
    if p:
        print("Process completed")
        print("Movie list making time: ",round(end1,4),"sec")
        print("Scraping time: ", str(end2))