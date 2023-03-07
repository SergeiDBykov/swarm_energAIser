import streamlit as st
import sys
#sys.path.append('../../') #Seems not work in Windows

from darts import TimeSeries
from darts.metrics import mape
from darts import concatenate
from darts.utils.timeseries_generation import datetime_attribute_timeseries as dt_attr #I copied the utils into the demo folder
from darts.dataprocessing.transformers import Scaler
from darts.metrics import smape, mape, mase, r2_score
from darts.models import CatBoostModel
from darts.models import LightGBMModel

from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

import lightgbm as lgb

from scripts.utils import data_path, set_mpl, read_london, add_datetime_features, get_spectrogram
from scripts.calendar import timeanddate_calendar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

import cufflinks as cf

import os
import platform
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0].parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

set_mpl()

#Page title
st.set_page_config(
    page_title="Siemens 2023 Hackathon - Tech for Sustainability",
)
st.markdown('# London dataset')



#DATE_COLUMN = 'date/time'
#DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
#            'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

# Note: Don't know why the read_london() didnt work if importing from utilies...
# Thus, put the read_london in the script directly and set the absolute path
# data_path = r'C:\Users\user\swarm_energAIser\0_data'
data_path = os.path.join(ROOT, "0_data")


@st.cache_data
def read_london():
    """
    # Read the data from the dataset
    """

    print(f"""
    Loading London data from {data_path}.
    Weather from `meteostat` package.
    STD and ToU tariffs are separated.
    Data resampled (mean) to 1H resolution from original 30min resolution.
    reutrns:
    df_std: pd.DataFrame with STD tariff data
    df_tou: pd.DataFrame with ToU tariff data
    df_weather: pd.DataFrame with weather data
    df_twitter: pd.DataFrame with twitter data (see `0_data/2.2_london_twitter.ipynb` for details)
    
    """)

    london_path = data_path+'/London_drive/'

    std_dile = 'london_std.pkl_gz'
    tou_file = 'london_tou.pkl_gz'
    weather_title = 'london_weather.csv'
    twitter_file = 'twitter_london.pkl'

    try:
        df_std = pd.read_pickle(london_path+std_dile, compression='gzip')
        df_tou = pd.read_pickle(london_path+tou_file, compression='gzip')
        df_weather = pd.read_csv(london_path+weather_title, index_col=0)
        df_weather.index = pd.to_datetime(df_weather.index)
        df_twitter = pd.read_pickle(london_path+twitter_file)
    except:
        print('(Windows load)')
        df_std = pd.read_pickle((london_path+std_dile).replace('/', '\\'), compression='gzip')
        df_tou = pd.read_pickle((london_path+tou_file).replace('/', '\\'), compression='gzip')
        df_weather = pd.read_csv((london_path+weather_title).replace('/', '\\'), index_col=0)
        df_weather.index = pd.to_datetime(df_weather.index)
        df_twitter = pd.read_pickle((london_path+twitter_file).replace('/', '\\'))
        
    df_twitter.index = df_twitter.index.tz_convert(None) #remove timezone info
    
    return [df_std, df_tou, df_weather, df_twitter]


@st.cache_data
def data_cleaning(data_std):
    """Data cleaning"""
    missed_frac = data_std.isna().mean(axis = 0).sort_values(ascending = False)
    data = data_std.drop(missed_frac[missed_frac > 0.3].index, axis = 1)
    homes_col = data.columns
    cols = data.columns
    return data, homes_col, cols


@st.cache_data
def process_data(data, _cols):
    """ Clustering and get spectrogram"""
    data_cluster = data[_cols[ (data[_cols].mean() > 0.1) & (data[cols].mean() < 0.13) ]]
    data_cluster = data_cluster.sample(n=house_number, axis=1, random_state=2023)
    data_cluster = data_cluster.dropna(how='all')
    power_avg = data_cluster.mean(axis=1).to_frame()
    power_avg.columns = ['power_avg']
    #get spectrogram
    power_avg_spe = get_spectrogram(power_avg, 'power_avg', 24*7, 24, plot = False)
    power_avg_spe = power_avg_spe/np.max(power_avg_spe, axis = 0)
    return power_avg, power_avg_spe, power_avg_spe

@st.cache_data
def data_preparation():
    """Prepare dataset for modeling"""
    df_dataset = power_avg.rename(columns={'power_avg':'load'}).copy()
    df_dataset = df_dataset.asfreq('H')
    
    index_temp = df_dataset.index.copy()
    
    df_dataset['date'] = pd.to_datetime(df_dataset.index.date)
    df_dataset['year'] = df_dataset.index.year.astype('int')
    df_dataset['weekday'] = df_dataset.index.weekday
    df_dataset['hour'] = df_dataset.index.hour + df_dataset.index.minute/60
    df_dataset['timeofweek'] = df_dataset['hour'] + df_dataset['weekday']*24
    
    df_dataset['load_shift24'] = df_dataset['load'].shift(24)
    df_dataset['load_shift168'] = df_dataset['load'].shift(168)
    
    df_dataset = df_dataset.merge(weather['temp'], left_index=True, right_index=True)
    df_dataset['temp_roll24_mean'] = df_dataset['temp'].rolling(24).mean()
    
    df_holiday = timeanddate_calendar(geo_id='uk',start_year=2011,end_year=2015+1)
    df_holiday_encode = df_holiday.copy()
    df_holiday_encode[['holiday_Name', 'holiday_Type']] = df_holiday_encode[['holiday_Name', 'holiday_Type']].astype('str').apply(preprocessing.LabelEncoder().fit_transform)
    df_holiday_encode.columns = df_holiday_encode.columns+'_encode'
    df_holiday_encode = df_holiday_encode.rename(columns={'date_encode':'date'})
    df_dataset = df_dataset.merge(df_holiday_encode, on='date')
    
    #df_dataset = df_dataset.drop('date',axis=1)
    df_dataset.index = index_temp

    return df_dataset

def data_split(data, tp='2013-01-01'):
    """split time series for training, validation and test """

    ratio_train, ratio_val, ratio_test = 0.6, 0.2, 0.2
    data = data.loc[:tp]
    train = data.iloc[:round(len(data)*ratio_train)]
    test = data.iloc[-round(len(data)*ratio_test):]
    val = data.drop(train.index).drop(test.index)

    return train, val, test

@st.cache_resource
def model_train(): 
    """Define and train models"""
    traindata = train.dropna()
    list_feat = list(traindata.select_dtypes(['int','float']).columns)
    list_feat.remove('load')
    
    ## Models for predicting 1 day
    naive_model_1day = LinearRegression()
    naive_model_1day.fit(traindata[['load_shift168']], traindata['load'])

    LR_model_1day = LinearRegression()
    LR_model_1day.fit(traindata[list_feat], traindata['load'])

    LGB_model_1day = lgb.LGBMRegressor()
    LGB_model_1day.fit(traindata[list_feat], traindata['load'])

    models_1day = [naive_model_1day, LR_model_1day, LGB_model_1day]
    
    ## Models for predicting 1 week
    naive_model_1week = LinearRegression()
    naive_model_1week.fit(traindata[['load_shift168']], traindata['load'])

    LR_model_1week = LinearRegression()
    LR_model_1week.fit(traindata[list_feat].drop('load_shift24', axis=1), traindata['load'])

    LGB_model_1week = lgb.LGBMRegressor()
    LGB_model_1week.fit(traindata[list_feat].drop('load_shift24', axis=1), traindata['load'])

    models_1week = [naive_model_1week, LR_model_1week, LGB_model_1week]  
    
    names = ['Naive', 'Linear regression', 'LightGBM']
    return models_1day, models_1week, names, list_feat


def user_input_features():
    horizon_str = st.sidebar.selectbox(
        'What is the horizon for your prediction?',
        ('1 Week (7days)','1 Day'))
    house_number = st.sidebar.slider(
        label='How many houses do you want to aggregate?', 
                             min_value=50, max_value=400, value=200, step=50)
    
    # features = pd.DataFrame(data, index=[0])
    show_raw        = st.sidebar.checkbox('Show raw data')
    show_preprocess = st.sidebar.checkbox('Show preprocessed data')
    show_train_val  = st.sidebar.checkbox('Show data split')
    user_inputs = {
        'horizon_str': horizon_str,
        'house_number': house_number,
        'show_raw': show_raw,
        'show_preprocess': show_preprocess,
        'show_train_val': show_train_val
        }
    return user_inputs
    
data_load_state = st.sidebar.text('Loading data...')
data_std, _, weather, twitter = read_london()
data_load_state.text("Done! (using st.cache_data)")

st.sidebar.header('User Input Parameters')

inputs = user_input_features()
horizon_str, house_number = inputs['horizon_str'], inputs['house_number']
show_raw, show_preprocess, show_train_val = \
    inputs['show_raw'], inputs['show_preprocess'], inputs['show_train_val']



    
# st.markdown('## 1. Modeling settings')
# # Setting of prediction horizon
# st.markdown('### 1.1 Prediction horison:')

# st.write('You selected:', horizon_str)

if horizon_str == '1 Day':
    horizon = 24 #used for the modeling
elif horizon_str == '1 Week (7days)':
    horizon = 24*7 #used for the modeling

# Setting of house_number for averaging energy use
# st.markdown('### 1.2 Number of aggregated houses:')

# st.write('You selected house_number:', house_number)

# Show the imported dataset with first 100 rows
# show_raw = st.button('Show raw data')
# if show_raw:

st.write('#### Aggregating ', house_number, ' houses, training model to predict for ', horizon_str)

# st.markdown('## 2. Data preprocessing')

# Data cleaning
data, homes_col, cols = data_cleaning(data_std)


#select homes with mean energy consumption above between 0.1 and 0.13
# Average energy use by indicated number of house_number
# To-do: instead of random sampling, select houses around median to avoid outliers
power_avg, power_avg_spe, power_avg_spe = process_data(data, cols)

if show_raw:
    st.markdown('### Showing the first 100 samples of raw data')
    st.write(data_std.head(100))

if show_preprocess:
    st.markdown('### Showing the 100 samples of preprocessed')
    st.write(power_avg.head(100))


# Prepare dataset for modeling
df_dataset = data_preparation()

# Split and preprocess the dataset
train, val, test = data_split(df_dataset)

# Draw the split result of the dataset
if show_train_val:
    fig,  ax =  plt.subplots( figsize = (10,5))
    train[['load']].plot(ax = ax, label="training data")
    val[['load']].plot(ax = ax, label="validation data")
    test[['load']].plot(ax = ax, label="test data")
    st.markdown('### Data Split')
    st.pyplot(fig)

# Define and train models
models_1day, models_1week, names, list_feat = model_train()

y_test_1day = []
y_test_1week = []

R2_1day = []
R2_1week = []

MAPE_1day = []
MAPE_1week = []

for model_1day, model_1week, name in zip(models_1day, models_1week, names):
    if name == 'Naive':
        pred_1day = model_1day.predict(test[['load_shift168']])
        pred_1week = model_1week.predict(test[['load_shift168']])
    else:
        pred_1day = model_1day.predict(test[list_feat])
        pred_1week = model_1week.predict(test[list_feat].drop('load_shift24', axis=1))
    
    y_test_1day.append(pd.Series(pred_1day))
    y_test_1week.append(pd.Series(pred_1week))

    r2_1day = r2_score(test['load'], pred_1day)
    r2_1week = r2_score(test['load'], pred_1week)

    mape_1day = mean_absolute_percentage_error(test['load'], pred_1day)
    mape_1week = mean_absolute_percentage_error(test['load'], pred_1week)
    
    R2_1day.append(r2_1day)
    R2_1week.append(r2_1week)
    
    MAPE_1day.append(mape_1day)
    MAPE_1week.append(mape_1week)  
   
y_test_1day = pd.concat(y_test_1day, axis=1)
y_test_1day.columns = names
y_test_1day.index = test.index
y_test_1day = pd.concat([y_test_1day, test[['load']]], axis=1)

y_test_1week = pd.concat(y_test_1week, axis=1)
y_test_1week.columns = names
y_test_1week.index = test.index
y_test_1week = pd.concat([y_test_1week, test[['load']]], axis=1)

metrics = pd.DataFrame([R2_1day, R2_1week, MAPE_1day, MAPE_1week],
                       columns=names,
                       index=['R2_1day', 'R2_1week', 'MAPE_1day', 'MAPE_1week'])


# time-series plot
fig_1day = y_test_1day.iloc[:24].iplot(asFigure=True, title='Prediction result of 1-day forecasting')
st.plotly_chart(fig_1day)
fig_1week = y_test_1week.iloc[:24*7].iplot(asFigure=True, title='Prediction result of 1-week forecasting')
st.plotly_chart(fig_1week)

# metrics
st.write(metrics)