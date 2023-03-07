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

from scripts.utils import data_path, set_mpl, read_london, add_datetime_features, get_spectrogram
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

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
def data_preparation(power_avg, weather):
    """Prepare dataset for modeling"""
    target = TimeSeries.from_dataframe(power_avg, freq = 'H')
    hodidays_covariates = target.add_holidays("UK")['holidays']
    frequency_covariates = TimeSeries.from_dataframe(power_avg_spe[['24.00', '12.00', '6.00']], freq = 'H')
    temperature_history = weather['temp']/np.max(weather['temp'])
    #perturb temperature to avoid perfect correlation
    rolling_std = temperature_history.rolling(24*3).std()
    temperature_forecast = temperature_history + np.random.normal(0, rolling_std, size = len(temperature_history))
    #temperature_covariate_past = TimeSeries.from_dataframe(temperature_forecast.to_frame(), freq = 'H')
    #temperature_covariate_future  = TimeSeries.from_dataframe(temperature_forecast.to_frame(), freq = 'H')
    temperature_covariate  = TimeSeries.from_dataframe(temperature_forecast.to_frame(), freq = 'H') #we can use it as past and future, although it is not perfect for past predictions
    #datetime encodings (normalized)
    datetime_covatiates = concatenate(
        [
            dt_attr(time_index = target.time_index, attribute =  "hour", one_hot = False, cyclic = False )/24,
            dt_attr(time_index = target.time_index, attribute =  "day_of_week", one_hot = False, cyclic = False )/7,
            dt_attr(time_index = target.time_index, attribute =  "month", one_hot = False, cyclic = False )/12,
            dt_attr(time_index = target.time_index, attribute =  "day_of_year", one_hot = False, cyclic = False )/365,
        ],
        axis="component",
    )
    return target, hodidays_covariates, frequency_covariates, datetime_covatiates

def data_split(target, tp='2013-01-01'):
    """split time series for training, validation and test """

    # target, _ = target.split_before("pd.Timestamp(tp)")
    target, _ = target.split_before(pd.Timestamp(tp))
    train, val = target.split_before(0.6)
    val, test = val.split_before(0.5)
    scaler = Scaler(scaler=MaxAbsScaler())
    train = scaler.fit_transform(train)
    val = scaler.transform(val)
    test = scaler.transform(test)
    val_len = len(val)
    return train, val, test

@st.cache_resource
def model_train(_train, _val, _datetime_covatiates, _hodidays_covariates, _frequency_covariates, _lags_horizon=24): 
    """Define and train models"""
    lags_horizon = 24
    model_naive = _val.shift(+24*7) #repeat value from 24*7 hours ago
    model_naive_train = _train.shift(+24*7) 
    model_catb = CatBoostModel(lags_future_covariates = (_lags_horizon, _lags_horizon) , lags_past_covariates=lags_horizon,
                        )#learning_rate = 0.007, num_trees = 1000, early_stopping_rounds = 5)
    model_lgbm = LightGBMModel(lags_future_covariates = (_lags_horizon, _lags_horizon) , lags_past_covariates=lags_horizon,
                        )#learning_rate = 0.007, num_trees = 1000, early_stopping_rounds = 5)
    cov_args = {"future_covariates": [_datetime_covatiates, _hodidays_covariates],
            "past_covariates": [_frequency_covariates]}
    model_catb.fit(_train, **cov_args)
    model_lgbm.fit(_train, **cov_args)
    models = [model_naive, model_catb, model_lgbm]
    names = ['naive',  'catboost', 'LightGBMModel']
    return models, names, model_naive_train

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
target, hodidays_covariates, frequency_covariates, datetime_covatiates = data_preparation(power_avg, weather)

# Split and preprocess the dataset
train, val, test = data_split(target)

# Draw the split result of the dataset
if show_train_val:
    fig,  ax =  plt.subplots( figsize = (10,5))
    train.plot(ax = ax, label="training data")
    val.plot(ax = ax, label="validation data")
    test.plot(ax = ax, label="test data")
    st.markdown('### Data Split')
    st.pyplot(fig)




# Define and train models
models, names, model_naive_train = model_train(train, val, datetime_covatiates, hodidays_covariates, frequency_covariates, _lags_horizon=24)
model_naive, model_catb, model_lgbm = models

# Plot of modeling results (static plot in matplotlib)
fig,  ax =  plt.subplots( figsize = (12,6))
train.tail(horizon).plot(ax = ax, label = 'train', lw = 2, alpha = 0.2)
val.head(horizon).plot(ax = ax, label = 'val', lw = 2, alpha = 0.2)
for model, name  in zip(models, names):
    if name == 'naive':
        pred_cv = model_naive 
        pred_train = model_naive_train
    else:
        pred_cv = model.predict(n=horizon)
        pred_train = model.predict(n=horizon, series=train[0: -horizon])
    
    
    print(f'{name} mape - train: {mape(train, pred_train):3g}')
    print(f'{name} mape - val: {mape(val, pred_cv):3g}')
    pred_train.tail(horizon).plot(ax = ax,  ls = '--', lw = 1, alpha = 0.3, label = name+':train')
    color =  ax.get_lines()[-1].get_color()
    pred_cv.head(horizon).plot(ax = ax,  ls = '--', lw = 2, alpha = 0.5, color = color, label = name+':CV')

    print(f'{name} r2_score - train: {r2_score(train, pred_train):3g}')
    print(f'{name} r2_score - val: {r2_score(val, pred_cv):3g}')
    pred_train.tail(horizon).plot(ax = ax,  ls = '--', lw = 1, alpha = 0.3, label = name+':train')
    color =  ax.get_lines()[-1].get_color()
    pred_cv.head(horizon).plot(ax = ax,  ls = '--', lw = 2, alpha = 0.5, color = color, label = name+':CV')     
#put legend outside
# ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
st.markdown('## Prediction')
st.pyplot(fig)
