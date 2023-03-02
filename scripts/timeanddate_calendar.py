import pandas as pd
import numpy as np

    


def timeanddate_calendar(geo_id='germany',     
                        start_year=2011,
                        end_year=2015):
    
    '''
    Download calendar data from timeanddate.com
    
    geo_id: The region where the search volumes are collected
    start_year, end_year: beginning and ending of collection period
    show_viz: whether or not the trends plots are displayed during the scraping
    '''


    df_holiday = []
    
    for year in np.arange(start_year, end_year+1):
        
        year = int(year) 
        holiday_temp = pd.read_html('https://www.timeanddate.com/holidays/'+geo_id+'/'+str(year)+'?hol=9')[0]
        holiday_temp.columns = holiday_temp.columns.get_level_values(0)
        holiday_temp = holiday_temp[~holiday_temp['Date'].str.contains('Observed').fillna(False)]
        holiday_temp = holiday_temp[holiday_temp['Type'].str.contains('oliday').fillna(False)]
        holiday_temp = holiday_temp.dropna(how='all')
        holiday_temp = holiday_temp[['Date', 'Name', 'Type']]
        holiday_temp['Date'] = str(year) + ' ' + holiday_temp['Date']
        holiday_temp['Date'] = pd.to_datetime(holiday_temp['Date'])
        
        df_holiday.append(holiday_temp)
        
    df_holiday = pd.concat(df_holiday, axis=0, ignore_index=True)
    
    df_holiday = df_holiday.drop_duplicates(subset=['Date'])
    df_holiday = df_holiday.set_index('Date').asfreq('D')
    df_holiday.loc[df_holiday.index.weekday>=5, 'Name'] = 'weekend'
    df_holiday.loc[df_holiday.index.weekday>=5, 'Type'] = 'weekend'
    df_holiday['Name'] = df_holiday['Name'].fillna('weekday')
    df_holiday['Type'] = df_holiday['Type'].fillna('weekday')
    df_holiday.columns = 'holiday_' + df_holiday.columns
    
    df_holiday = df_holiday.reset_index()
    df_holiday = df_holiday.rename(columns={'Date':'date'}) 
                
    
    return df_holiday
