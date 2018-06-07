# -*- coding: utf-8 -*-
"""
Created on Wed May 30 18:51:13 2018

@author: Abhitej Kodali
"""

import pandas as pd
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import TensorBoard
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import metrics

def compute_rating(rating):
        if rating == 'R':
            return 26.91

        if rating == 'NR':
            return 0.82
        
        if rating == 'PG-13':
            return 47.77  

        if rating == 'PG':
            return 20.18
    
        if rating == 'G':
            return 4.28
    
        if rating == 'NC17':
            return 0.02
    
def Action_Adventure(a):
    if a == 1:
        return 60

def Comedy(b):
    if b == 1:
        return 32
    
def Documentary(c):
    if c == 1:
        return 2  
    
def Drama(d):
    if d == 1:
        return 35
    
def Horror(e):
    if e == 1:
        return 10
    
def Musical_Arts(f):
    if f == 1:
        return 4
    
def Mystery_Suspense(g):
    if g == 1:
        return 18
    
def Romance(h):
    if h == 1:
        return 10
    
def Western(i):
    if i == 1:
        return 1


def new_features(df):
    
    #Label
    df['label'] = df['Tomatometer'] - df['Audience_score']

    
    # Holiday Column: if a movie was released on a holiday or not
    df['Release_date'] = pd.to_datetime(df['Release_date'])
    cal = calendar()
    holidays = cal.holidays(start=df['Release_date'].min(skipna = True), end=df['Release_date'].max(skipna = True))
    df['Holiday'] = df['Release_date'].isin(holidays)

    # Break the Genres into separate columns and mark them as 1 or 0.
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
                df.at[i,g6] = 1
    
    # Calculate studio rating and director rating for their past contributions before a movie was released
    # If the director or studio was making their first movie then their average rating will be 0 for previous movies.
    # Two version of ratings were created. One using critics' rating and another from audiences' perspective.
    
    df = df.sort_values("Release_date", ascending=True)
    
    # Director's ratings
    df['TOM_M']=pd.DataFrame(df.groupby("Director")['Tomatometer'].apply(lambda x: x.shift().expanding(min_periods=1).mean()))
    df['AS_M']=pd.DataFrame(df.groupby("Director")['Audience_score'].apply(lambda x: x.shift().expanding(min_periods=1).mean()))
    df['AAR_M']=pd.DataFrame(df.groupby("Director")['Audience_average_rating'].apply(lambda x: x.shift().expanding(min_periods=1).mean()))
    df['CRC_M']=pd.DataFrame(df.groupby("Director")['Critic_reviews_count'].apply(lambda x: x.shift().expanding(min_periods=1).mean()))
    df['CFR_M']=pd.DataFrame(df.groupby("Director")['Critic_fresh_rating'].apply(lambda x: x.shift().expanding(min_periods=1).mean()))
    df['CAR_M']=pd.DataFrame(df.groupby("Director")['Critic_average_rating'].apply(lambda x: x.shift().expanding(min_periods=1).mean()))
    df['CRR_M']=pd.DataFrame(df.groupby("Director")['Critic_rotten_rating'].apply(lambda x: x.shift().expanding(min_periods=1).mean()))
    
    # Studio Ratings
    df['S_TOM_M']=pd.DataFrame(df.groupby("Studio")['Tomatometer'].apply(lambda x: x.shift().expanding(min_periods=1).mean()))
    df['S_AS_M']=pd.DataFrame(df.groupby("Studio")['Audience_score'].apply(lambda x: x.shift().expanding(min_periods=1).mean()))
    df['S_AAR_M']=pd.DataFrame(df.groupby("Studio")['Audience_average_rating'].apply(lambda x: x.shift().expanding(min_periods=1).mean()))
    df['S_CRC_M']=pd.DataFrame(df.groupby("Studio")['Critic_reviews_count'].apply(lambda x: x.shift().expanding(min_periods=1).mean()))
    df['S_CFR_M']=pd.DataFrame(df.groupby("Studio")['Critic_fresh_rating'].apply(lambda x: x.shift().expanding(min_periods=1).mean()))
    df['S_CAR_M']=pd.DataFrame(df.groupby("Studio")['Critic_average_rating'].apply(lambda x: x.shift().expanding(min_periods=1).mean()))
    df['S_CRR_M']=pd.DataFrame(df.groupby("Studio")['Critic_rotten_rating'].apply(lambda x: x.shift().expanding(min_periods=1).mean()))
    
    # Removing the movies that have very low Tomatometer and Critic Score
    df = df.drop(df[(df.Tomatometer <20) | (df.Audience_score < 20)].index)
    
    df.rename(columns={'Action & Adventure': 'Action_Adventure', 'Musical & Performing Arts': 'Musical_Arts', 
                       'Mystery & Suspense': 'Mystery_Suspense', 0:'Month',1:'Date',3:'Year'}, inplace=True)
    
    # Some more assumptions
    df['AA_Genre_Value']=df['Action_Adventure'].apply(Action_Adventure)
    df['Com_Genre_Value']=df['Comedy'].apply(Comedy)
    df['Doc_Genre_Value']=df['Documentary'].apply(Documentary)
    df['Dra_Genre_Value']=df['Drama'].apply(Drama)
    df['Horr_Genre_Value']=df['Horror'].apply(Horror)
    df['Mus_Genre_Value']=df['Musical_Arts'].apply(Musical_Arts)
    df['Mys_Genre_Value']=df['Mystery_Suspense'].apply(Mystery_Suspense)
    df['Rom_Genre_Value']=df['Romance'].apply(Romance)
    df['Wes_Genre_Value']=df['Western'].apply(Western)
    df['Movie_rating_value']=df['Movie_rating'].apply(compute_rating)
    
    df.fillna(0,inplace=True)
    
    df['Total_Genre_Value'] = df.AA_Genre_Value+df.Com_Genre_Value+df.Com_Genre_Value+df.Doc_Genre_Value+df.Dra_Genre_Value+df.Horr_Genre_Value+df.Mus_Genre_Value+df.Mys_Genre_Value+df.Rom_Genre_Value+df.Wes_Genre_Value
    
    return(df)

def baseline_model():

    model = Sequential()
    model.add(Dense(120, input_dim=28,  activation='relu', kernel_initializer='normal', name='Layer_1'))
    model.add(Dropout(0.6))
    
    model.add(Dense(120, activation='relu', kernel_initializer='normal', name='Layer_2'))
    model.add(Dropout(0.6))
    
    
    model.add(Dense(1, kernel_initializer='normal', name='Output_Layer'))
    
    #sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    
    model.compile(loss='mean_squared_error', optimizer= 'Adadelta')
    
    return model

if __name__=='__main__':
    movies = pd.read_csv("Full Movie List.csv")
    df = new_features(movies)
    
    X = df[[ 'Runtime',
       'CAR_M', 'TOM_M',  'AAR_M', 'CRC_M',  'CRR_M','Movie_rating_value', 'AA_Genre_Value','Com_Genre_Value',
       'Doc_Genre_Value', 'Dra_Genre_Value', 'Horr_Genre_Value', 'Mus_Genre_Value', 'Mys_Genre_Value', 'Rom_Genre_Value', 
       'Wes_Genre_Value', 'Total_Genre_Value', 'S_CAR_M', 'S_TOM_M',  'S_AAR_M', 'S_CRC_M',  'S_CRR_M']]

    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=123)
    
    scale = MinMaxScaler(feature_range=(0, 1))
    X_train = scale.fit_transform(X_train)
    X_test = scale.transform(X_test)
    
    test = SelectKBest(score_func=chi2, k=10)
    fit = test.fit(X, y)
    # summarize scores
    np.set_printoptions(precision=3)
    print(fit.scores_)
    features = fit.transform(X)
    # summarize selected features
    print(features[0:20,:])
    pca = PCA(n_components=20)
    fit = pca.fit(X)
    print(fit.components_)
    
    tensorboard = TensorBoard(log_dir='logs', write_graph=True, histogram_freq=0, write_images=False)

    model = KerasRegressor(build_fn=baseline_model, epochs=501, batch_size=256,verbose=1, callbacks=[tensorboard])
    
    model.fit(X_train,y_train)

    preds = model.predict(X_test)
    
    model_json = model.model.to_json()
    model.model.save_weights('model_weights.h5')
    
    print('MAE:', metrics.mean_absolute_error(y_test, preds))
    print('MSE:', metrics.mean_squared_error(y_test, preds))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, preds)))
    