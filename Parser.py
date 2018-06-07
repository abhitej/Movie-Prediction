# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 17:43:15 2018

@author: Abhitej Kodali
"""

from bs4 import BeautifulSoup
import pandas as pd
import os
import codecs
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from datetime import timedelta
from datetime import datetime

def movie_extract(records, file):
    soup = BeautifulSoup(file,'html.parser')
    
    movie_rat = 'NaN'
    movie_studio = 'NaN'
    movie_genre = 'NaN'
    movie_boxoffice = 0
    movie_release = 'NaN'
    movie_runtime = 'NaN'
    movie_director = 'NaN'
    movie_writer = 'NaN'
    plot = 'NaN'
    aud_score = 0
    avg_crit_rat = 0
    tomatometer = 0
    avg_aud_rat = 0.
    user_aud_rat = 0
    fresh_crit_rat = 0
    rotten_crit_rat = 0
    movie = 'NaN'
    
    #Movie Title
    try:
        movie = soup.find('h1',{'id':'movie-title'}).text.replace("\n","").strip()
    except:
        movie = 'NaN'
    
    #Tomato-meter Score
    try:
        tomatometer = int(soup.find('span',{'class':'meter-value superPageFontColor'}).text.replace("%","").strip())
    except:
        tomatometer = 0
    
    a = soup.find_all('div',{'class':'superPageFontColor'})
    a_1 = soup.find('div',{'class':'critic-score meter'})
    if a_1:
        if 'Rating' in a[0].text:
            a1 = a[0].text.replace("Average Rating:","").replace("\n","").replace("\\n","").strip().split('/')
            if isinstance(float(a1[0]),float):
                avg_crit_rat = float(a1[0])/10
        if 'Fresh' in a[2].text:
            fresh_crit_rat = int(a[2].text.replace("Fresh:","").replace("\n","").replace("\\n","").strip())
        if 'Rotten' in a[3].text:
            rotten_crit_rat = int(a[3].text.replace("Rotten:","").replace("\n","").replace("\\n","").strip())
    
    reviews_crit_count = fresh_crit_rat + rotten_crit_rat
    
    #Audiene metrics - Popcornmeter
    b = soup.find('div',{'class':'audience-score meter'})
    
    try:
        aud_score = int(b.text.replace("\n","").replace("liked it","").replace("\\n","").replace("%","").strip())
    except:
        aud_score = 0
    try:
        b1 = soup.find('div',{'class':'audience-info hidden-xs superPageFontColor'}).text.replace(",","").replace("\\n","").strip().split()
        b2 = b1[2].split('/')
        if b2[0]!='N':
            avg_aud_rat = float(b2[0])/5
        user_aud_rat = int(b1[5])
    except:
        avg_aud_rat = 0.
        user_aud_rat = 0
    
    #Movie Info
    c1 = soup.find_all('div',{'class':'meta-value'})
    c2 = soup.find_all('div',{'class':'meta-label subtle'})
    try:
        plot = soup.find('div',{'id':'movieSynopsis'}).text.strip()
    except:
        plot = 'NaN'
    
    for c in range(len(c2)):
        if c2[c].text.strip() == 'Rating:':
            c4 = c1[c].text.strip().split()
            movie_rat = c4[0]
        elif c2[c].text.strip() == 'Studio:':
            movie_studio = c1[c].text.replace("\n","").strip('\n').strip()
        elif c2[c].text.strip() == 'Runtime:':
            movie_runtime = c1[c].text.replace("\n","").strip()
        elif c2[c].text.strip() == 'Box Office:':
            movie_boxoffice = int(c1[c].text.replace(",","").replace("$","").strip())
        elif c2[c].text.strip() == 'In Theaters:':
            c3 = c1[c].text.replace(",","").replace("Wide","").replace("\n","").replace("\\n","").replace("limited","").replace("wide","").strip()
            movie_release = c3.replace("\xc2\xa0","").strip()
        elif c2[c].text.strip() == 'Directed By:':
            movie_director = c1[c].text.replace("\n","").strip()
        elif c2[c].text.strip() == 'Written By:':
            movie_writer = c1[c].text.replace("\n","").strip()
        elif c2[c].text.strip() == 'Genre:':
            c3 = c1[c].text.strip().replace("\n","").split()
            movie_genre = " ".join(c3)
        else:
            continue
    
    d1 = soup.find_all('div',{'class':'cast-item media inlineBlock '})
    d2=['NaN']*6
    
    if d1:
        for d in range(len(d1)):
            d2[d] = d1[d].div.span.text.strip()
       
    records.append((movie,tomatometer,aud_score,avg_aud_rat,user_aud_rat,avg_crit_rat,reviews_crit_count,
                    fresh_crit_rat,rotten_crit_rat,movie_rat,plot,movie_studio,movie_director,movie_writer,
                    movie_genre,movie_runtime,movie_release,movie_boxoffice,d2[0],d2[1],d2[2],d2[3],d2[4],d2[5]))
    
    return records

def holiday_checker(release_date):
    
    vReturn = 0
    cal = calendar()
    startDate = datetime.strptime(release_date, '%m/%d/%Y') - timedelta(days=7)
    endDate = datetime.strptime(release_date, '%m/%d/%Y') + timedelta(days=5)
    
    holidays = cal.holidays(start=startDate, end=endDate).to_pydatetime()
    print(holidays)
    if holidays:vReturn=1    
    return vReturn

def movie_cleaning(df):
    #Cleaning Process
    df = df.replace(r'\\n','',regex=True)
    df = df.replace(r"\\'","'",regex=True)
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    df['Release_date'] = pd.to_datetime(df['Release_date'],format="%b %d %Y",errors='ignore')
    df['Runtime'] = pd.to_numeric(df['Runtime'].map(lambda x: str(x)[:-8]))
    cal = calendar()
    holidays = cal.holidays(start=df['Release_date'].min(), end=df['Release_date'].max())
    df['Holiday'] = df['Release_date'].isin(holidays)

    g = pd.Series(df['Genre'])
    g1 = []
    
    for i in range(len(g)):
        g3 = str(g[i]).split(', ')
        for g4 in g3:
            g1.append(str(g4).strip())
    
    g5 = sorted(set(g1))
    
    for i in range(len(g5)):
        if g5[i]!='NaN':
            df[g5[i]]=0
    
    for i in range(len(g)):
        g3 = str(g[i]).split(', ')
        for g4 in g3:
            g6 = str(g4).strip()
            if g6 in df.columns:
                df.set_value(i,g6,1)
    
    return df


if __name__=='__main__':
    records = []
    bad_count = 0
    directory = os.path.normpath(r"Page Links")
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".html"):
                f=codecs.open(os.path.join(subdir, file),'r',encoding="utf8")
                a = f.read()
                if 'movie-title' in a:
                    records = movie_extract(records, a)
                    f.close()
                else:
                    bad_count = bad_count + 1
                    f.close()
                    continue
    
    movie_list = pd.DataFrame(records,columns=['Movie','Tomatometer','Audience_score','Audience_average_rating',
                                               'Audience_user_count','Critic_average_rating','Critic_reviews_count',
                                               'Critic_fresh_rating','Critic_rotten_rating','Movie_rating',
                                               'Movie_plot','Studio','Director','Writer','Genre','Runtime',
                                               'Release_date','Box_office','Actor_1','Actor_2','Actor_3',
                                               'Actor_4','Actor_5','Actor_6'])
    
    #movie cleaning process
    movie_list = movie_cleaning(movie_list)
    
    #verification process
    print(movie_list.head())
    print("\n Bad Records: ", bad_count)
    
    #writing to csv file
    movie_list.to_csv("Full Movie List.csv", index=False)
    