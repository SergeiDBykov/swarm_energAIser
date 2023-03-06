import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from pytrends.request import TrendReq

import os
import time

from sklearn.linear_model import LinearRegression



def download_google_trends(geo_id='GB-ENG',     
                           list_primary_use = ['School','Education','Office'],
                           start_year=2011,
                           end_year=2015,
                           show_viz=True):
    
    '''
    Download Google Trends data with specified region, topics, and years 
    
    geo_id: The region where the search volumes are collected
    list_primary_use: initial topics of google trends (more relevant topics will be suggested based on these ones)
    start_year, end_year: beginning and ending of collection period
    show_viz: whether or not the trends plots are displayed during the scraping
    '''
    

    # Collect suggested topics based on given ones
    df_suggs = pd.DataFrame()
    for category in list_primary_use:
        try:
            #print(category)
            pytrends = TrendReq(hl='en-US', tz=360)
            kw_list = [category]
            pytrends.build_payload(kw_list)
            suggs = pytrends.related_topics()
            df_temp = pd.DataFrame(suggs[category]['top'])
            df_temp['category'] = category
            df_suggs = pd.concat([df_suggs, df_temp],axis=0,ignore_index=True)
            #print(df_temp)
            #print('\n')    
        except:
            print('error')
       
    # Clean and rename the dataframe of suggested topics    
    df_suggs = df_suggs.rename(columns={'topic_mid':'mid', 'topic_title':'title', 'topic_type':'type'})
    df_suggs = df_suggs.drop_duplicates(subset=['link']).reset_index(drop=True)
       
    # Transform dataframe into list for downloading
    title_list = df_suggs.loc[:, 'title'].to_list()
    type_list = df_suggs.loc[:, 'type'].to_list()
    category_list = df_suggs.loc[:, 'category'].to_list()
    mid_list = df_suggs.loc[:, 'mid'].to_list()
    
    
    # Collect data via pytrends
    df_trend = pd.DataFrame()
    
    for title, title_type, category, mid in zip(title_list, type_list, category_list, mid_list):
        print(geo_id +'/'+ category +'/'+ title)
    
        try:    
            for year in np.arange(start_year, end_year+1, 1):
                year = str(int(year))
                df_trend_temp = pd.DataFrame()
    
                pytrends = TrendReq(hl='en-US', tz=360)
                pytrends.build_payload([mid], timeframe= year +'-01-01 ' + year + '-07-01',geo=geo_id,gprop='')
                df_trend_temp_half1 = pytrends.interest_over_time()
                time.sleep(1)
    
                pytrends = TrendReq(hl='en-US', tz=360)
                pytrends.build_payload([mid], timeframe= year +'-06-01 ' + year + '-12-31',geo=geo_id,gprop='')
                df_trend_temp_half2 = pytrends.interest_over_time()
                time.sleep(1)
    
                dataset = pd.concat([df_trend_temp_half1.loc[year +'-06-01 ':year +'-07-01 ', mid].rename('X'),
                                     df_trend_temp_half2.loc[year +'-06-01 ':year +'-07-01 ', mid].rename('y')], axis=1).dropna()
    
                reg_lr = LinearRegression().fit(dataset[['X']], dataset[['y']])
    
                df_trend_temp_half1[mid] = reg_lr.predict(df_trend_temp_half1[[mid]])
    
                df_trend_temp = pd.concat([df_trend_temp_half1.loc[year +'-01-01 ':year +'-06-30 '],
                                           df_trend_temp_half2.loc[year +'-07-01 ':year +'-12-31 ']], axis=0)
                df_trend_temp = df_trend_temp[mid].rename('value')
                df_trend_temp = df_trend_temp.reset_index()
    
                #df_trend_temp.set_index('date').plot(figsize=(15,3), title = title + ' / ' + year)
                #weekly_trend_year.plot(figsize=(15,3))
                #plt.show()
    
                df_trend_temp['geo_id'] = geo_id
                df_trend_temp['title'] = title
                df_trend_temp['type'] = title_type
                df_trend_temp['category'] = category
                df_trend_temp['year'] = year
    
                df_trend_temp['value'] = (df_trend_temp['value']-df_trend_temp['value'].mean())/df_trend_temp['value'].std()
    
                df_trend = pd.concat([df_trend, df_trend_temp], axis=0, ignore_index=True)

            if show_viz == True:     
                df_trend.loc[(df_trend['geo_id']==geo_id)&(df_trend['title']==title)].set_index('date').plot(figsize=(15,3), title=geo_id +'/'+ category +'/'+ title)
                plt.show()
            
            print('successful')
            print('----------------------------------------------------------------------------------------------------')
    
    
        except:
            print('error')
            
    df_trend = df_trend.sort_values(['geo_id','category','title','date'])
    
    return df_trend
