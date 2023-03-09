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
from sklearn.decomposition import PCA
import lightgbm as lgb

from scripts.utils import data_path, set_mpl, read_london, add_datetime_features, get_spectrogram
from scripts.calendar import timeanddate_calendar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

import cufflinks as cf
import pydeck as pdk

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
    page_title="energAIser - Siemens 2023 Hackathon - Tech for Sustainability",
)
st.markdown('# London dataset')

############ Functions ############ 

#DATE_COLUMN = 'date/time'
#DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
#            'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

# Note: Don't know why the read_london() didnt work if importing from utilies...
# Thus, put the read_london in the script directly and set the absolute path
# data_path = r'C:\Users\user\swarm_energAIser\0_data'
# data_path = os.path.join(ROOT, "0_data")

# @st.cache_data
# def read_data():
#     """
#     # Read the data from the dataset
#     """

#     print(f"""
#     Loading London data from {data_path}.
#     Weather from `meteostat` package.
#     STD and ToU tariffs are separated.
#     Data resampled (mean) to 1H resolution from original 30min resolution.
#     reutrns:
#     df_std: pd.DataFrame with STD tariff data
#     df_tou: pd.DataFrame with ToU tariff data
#     df_weather: pd.DataFrame with weather data
#     df_twitter: pd.DataFrame with twitter data (see `0_data/2.2_london_twitter.ipynb` for details)
    
#     """)

#     london_path = data_path+'/London_drive/'

#     std_dile = 'london_std.pkl_gz'
#     tou_file = 'london_tou.pkl_gz'
#     weather_title = 'london_weather.csv'
#     twitter_file = 'twitter_london.pkl'

#     try:
#         df_std = pd.read_pickle(london_path+std_dile, compression='gzip')
#         df_tou = pd.read_pickle(london_path+tou_file, compression='gzip')
#         df_weather = pd.read_csv(london_path+weather_title, index_col=0)
#         df_weather.index = pd.to_datetime(df_weather.index)
#         df_twitter = pd.read_pickle(london_path+twitter_file)
#     except:
#         print('(Windows load)')
#         df_std = pd.read_pickle((london_path+std_dile).replace('/', '\\'), compression='gzip')
#         df_tou = pd.read_pickle((london_path+tou_file).replace('/', '\\'), compression='gzip')
#         df_weather = pd.read_csv((london_path+weather_title).replace('/', '\\'), index_col=0)
#         df_weather.index = pd.to_datetime(df_weather.index)
#         df_twitter = pd.read_pickle((london_path+twitter_file).replace('/', '\\'))
        
#     df_twitter.index = df_twitter.index.tz_convert(None) #remove timezone info
    
#     return [df_std, df_tou, df_weather, df_twitter]
#%% 新增的functions! :D
 
def geomap_pydeck(lat, lon, number_units):
    '''
    Geomap of the specified lat and lon with indicated number_units 
    (randomness has been added in order to spread locations of units wider)
    '''
    chart_data = pd.DataFrame(
       np.random.randn(number_units, 2) / [50, 50] + [lat, lon],
       columns=['lat', 'lon'])    
    
    st.pydeck_chart(pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(
            latitude=lat,
            longitude=lon,
            zoom=11,
            pitch=50,
        ),
        layers=[
            pdk.Layer(
               'HexagonLayer',
               data=chart_data,
               get_position='[lon, lat]',
               radius=200,
               elevation_scale=4,
               elevation_range=[0, 1000],
               pickable=True,
               extruded=True,
            ),
            pdk.Layer(
                'ScatterplotLayer',
                data=chart_data,
                get_position='[lon, lat]',
                get_color='[200, 30, 0, 160]',
                get_radius=200,
            ),
        ],
    ))

#%%

def read_data():
    """A wrapper for data reading with st.cache_data decorator"""
    # global data_load_state
    data_load_state = st.text('Loading data...')
    # df_std, df_tou, df_weather, df_twitter = read_london()
    df_std, df_tou, df_weather, df_twitter = read_london()
    data_load_state.text("Data loading done!")
    return df_std, df_tou, df_weather, df_twitter


@st.cache_data
def data_cleaning(data_std):
    """Data cleaning"""
    missed_frac = data_std.isna().mean(axis = 0).sort_values(ascending = False)
    data = data_std.drop(missed_frac[missed_frac > 0.3].index, axis = 1)
    homes_col = data.columns
    cols = data.columns
    return data, homes_col, cols


@st.cache_data
def process_data(data, _cols, house_number):
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
    df_dataset['temp_roll24_std'] = df_dataset['temp'].rolling(24).std()
    
    df_holiday = timeanddate_calendar(geo_id='uk',start_year=2011,end_year=2015+1)
    df_holiday_encode = df_holiday.copy()
    df_holiday_encode[['holiday_Name', 'holiday_Type']] = df_holiday_encode[['holiday_Name', 'holiday_Type']].astype('str').apply(preprocessing.LabelEncoder().fit_transform)
    df_holiday_encode.columns = df_holiday_encode.columns+'_encode'
    df_holiday_encode = df_holiday_encode.rename(columns={'date_encode':'date'})
    df_dataset = df_dataset.merge(df_holiday, on='date')
    df_dataset = df_dataset.merge(df_holiday_encode, on='date')
    
    #df_dataset = df_dataset.drop('date',axis=1)
    df_dataset.index = index_temp

    return df_dataset

def data_split(data, tp='2013-01-01', train_ratio=0.6, val_ratio=0.5):
    """split time series for training, validation and test 
    train_ratio = ratio of train - train + val + test
    val_ratio   = ratio of val - val + test
    """

    
    data = data.loc[:tp]
    train = data.iloc[:round(len(data)*train_ratio)]
    test = data.iloc[-round(len(data)*((1-train_ratio)*(1-val_ratio))):]
    val = data.drop(train.index).drop(test.index)

    return train, val, test

@st.cache_resource
def model_train(_train): 
    """Define and train models"""
    n_estimators = 50
    
    traindata = _train.dropna()
    list_feat = list(traindata.select_dtypes(['int','float']).columns)
    list_feat.remove('load')
    #list_feat.remove('year')
    #list_feat.remove('holiday_Name_encode')
   
    ## Models for predicting 1 day
    naive_model_1day = LinearRegression()
    naive_model_1day.fit(traindata[['load_shift168']], traindata['load'])

    LR_model_1day = LinearRegression()
    LR_model_1day.fit(traindata[list_feat], traindata['load'])

    LGB_model_1day = lgb.LGBMRegressor(n_estimators=n_estimators)
    LGB_model_1day.fit(traindata[list_feat], traindata['load'])

    models_1day = [naive_model_1day, LR_model_1day, LGB_model_1day]
    
    ## Models for predicting 1 week
    naive_model_1week = LinearRegression()
    naive_model_1week.fit(traindata[['load_shift168']], traindata['load'])

    LR_model_1week = LinearRegression()
    LR_model_1week.fit(traindata[list_feat].drop('load_shift24', axis=1), traindata['load'])

    LGB_model_1week = lgb.LGBMRegressor(n_estimators=n_estimators)
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
    



############ Layout ############ 
st.sidebar.header('User Input Parameters')

############ Input ############ 
inputs = user_input_features()
# data_load_state.text("Done! (using st.cache_data)")

# data_load_state = st.sidebar.text('Loading data...')

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

############ Output ############ 
st.sidebar.write('#### Training model to predict substations of size ', house_number, ' for ', horizon_str)

# read data   
data_std, df_tou, df_weather, df_twitter = read_data()

# Data cleaning
data, homes_col, cols = data_cleaning(data_std)


#select homes with mean energy consumption above between 0.1 and 0.13
# Average energy use by indicated number of house_number
# To-do: instead of random sampling, select houses around median to avoid outliers
power_avg, power_avg_spe, power_avg_spe = process_data(data, cols, house_number)

if show_raw:
    st.markdown('### Showing the first 100 samples of raw data')
    st.write(data_std.head(100))

if show_preprocess:
    st.markdown('### Showing the 100 samples of preprocessed')
    st.write(power_avg.head(100))


# Prepare dataset for modeling
df_dataset = data_preparation(power_avg, df_weather)

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
models_1day, models_1week, names, list_feat = model_train(train)

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

#%% Section1: Basic info
st.markdown('## 1. Where and what is the dataset?')

# geomap of the location
st.markdown('### 1.1 Geographical location:')
geomap_pydeck(lat=51.5, lon=-0.136, number_units=house_number)

# time-series plot of the dataset with normalized values and moving avg
st.markdown('### 1.2 Time series in the dataset:')
dataset_plot = df_dataset[['load', 'temp']].copy()
dataset_plot = dataset_plot.merge(df_twitter[['tweets_total']], left_index=True, right_index=True)
dataset_plot_mov_avg = dataset_plot.rolling(24).mean()
dataset_plot = (dataset_plot-dataset_plot.mean())/dataset_plot.std()
dataset_plot = dataset_plot.iplot(asFigure=True, title='Normalized time series of the dataset')
st.plotly_chart(dataset_plot)

dataset_plot_mov_avg.columns = dataset_plot_mov_avg.columns+' (moving avg.)'
#dataset_plot = pd.concat([dataset_plot, dataset_plot_mov_avg], axis=1)
dataset_plot_mov_avg = (dataset_plot_mov_avg-dataset_plot_mov_avg.mean())/dataset_plot_mov_avg.std()
dataset_plot_mov_avg = dataset_plot_mov_avg.iplot(asFigure=True, title='Normalized time series of the dataset (moving average)')
st.plotly_chart(dataset_plot_mov_avg)

#%% Section2: EDA
st.markdown('## 2. Exploratory Data Analysis (EDA) and characterization')

# weekly_profile
weekly_profile = plt.figure(figsize=(10, 3))
sns.scatterplot(data=df_dataset, x="timeofweek", y="load", alpha=0.05)
plt.title('Weekly profiles of energy use')
plt.ylim((df_dataset['load'].min(), df_dataset['load'].mean()+df_dataset['load'].std()*5))
st.pyplot(weekly_profile)

# Scatter plot for demand and outdoor temperature
df_plot = df_dataset.resample('D').mean().copy()
df_plot = df_plot[df_plot>(df_plot.mean()-df_plot.std()*3)]
df_plot = df_plot[df_plot<(df_plot.mean()+df_plot.std()*5)]
df_plot['weekday/weekend'] = 'weekday'
df_plot.loc[df_plot['weekday']>4, 'weekday/weekend'] ='weekend'

# Scatter plot for demand and outdoor temperature
df_plot = df_dataset.resample('D').mean().copy()
df_plot = df_plot[df_plot>(df_plot.mean()-df_plot.std()*3)]
df_plot = df_plot[df_plot<(df_plot.mean()+df_plot.std()*5)]
df_plot['weekday/weekend'] = 'weekday'
df_plot.loc[df_plot['weekday']>4, 'weekday/weekend'] ='weekend'
temp_vs_load = plt.figure(figsize=(10, 5))
sns.scatterplot(x='temp', y='load', hue="weekday/weekend",
           data=df_plot, alpha=0.5)
plt.title('Load v.s. outdoor temperature')
st.pyplot(temp_vs_load)

# Boxplot of day types
df_plot = df_dataset.resample('D').mean().copy()
df_plot = df_plot[df_plot>(df_plot.mean()-df_plot.std()*3)]
df_plot = df_plot[df_plot<(df_plot.mean()+df_plot.std()*5)]
df_plot = df_plot.merge(df_dataset[['holiday_Type_encode','holiday_Type']],on='holiday_Type_encode')
df_plot.loc[df_plot['holiday_Type'].str.contains('Holiday'), 'holiday_Type'] = 'Holiday'
daytype_vs_load = plt.figure(figsize=(10, 5))
sns.boxplot(x='holiday_Type', y='load', data=df_plot)
plt.title('Load v.s. day types')
st.pyplot(daytype_vs_load)

# Corr bewteen load and all other variables
dataset_plot = df_dataset[['load', 'temp']].copy()
dataset_plot = dataset_plot.merge(df_twitter, left_index=True, right_index=True)
dataset_plot = dataset_plot.resample('D').mean()

#pca = PCA(n_components=2)
#df_PCA = pd.DataFrame(pca.fit_transform(dataset_plot.T), columns=['PCA1', 'PCA2'], index=dataset_plot.columns)
#PCA_fig = df_PCA.reset_index().iplot(kind='scatter',x='PCA1',y='PCA2',categories='index', 
#                       asFigure=True, title='Distance between time series')
#st.plotly_chart(PCA_fig)

corr_dataset = dataset_plot.corr()
if len(corr_dataset)>10: # Remove irrelevant variables if the heatmap is bigger than 10 x 10
    corr_dataset = corr_dataset[corr_dataset['load'].abs().sort_values(ascending=False)[:10].index]
    corr_dataset = corr_dataset.loc[corr_dataset.columns]
corr_heatmap = plt.figure(figsize=(10, 6))
sns.heatmap(corr_dataset, annot=True, cmap="coolwarm")
plt.title('Correlations between variables')
st.pyplot(corr_heatmap)

corr_dataset = corr_dataset['load'].drop('load')
corr_dataset = corr_dataset[corr_dataset.abs().sort_values(ascending=False).index]
for row in corr_dataset.items():
    var_name, corr_value = row
    if corr_value > 0.6:
        st.write('#### **Load** is highly correlated to ', '`{}`'.format(var_name), ' with correlation coeff of ', '`{}`'.format(str(round(corr_value,4))) )
        print('Load is highly correlated to '+var_name+' with correlation coeff of ' + str(round(corr_value,4)))
    elif corr_value < -0.6:
        st.write('#### **Load** is highly correlated to negative', '`{}`'.format(var_name), ' with correlation coeff of ', '`{}`'.format(str(round(corr_value,4))))
        print('Load is highly correlated to negative '+var_name+' with correlation coeff of ' + str(round(corr_value,4)))        

#%% Section4: Modeling results
st.markdown('## 4. How is the prediction result?')

# Feature importance
df_importance = pd.DataFrame(data=models_1day[-1].feature_importances_, index=list_feat, columns=['Feature importance'])
df_importance = df_importance.sort_values('Feature importance', ascending=False)
st.write(df_importance)

# time-series plot
d = st.date_input(
    "The start day for visualizing prediction results",
    test.iloc[0]['date'].date())
st.write('You select:', d)

fig_1day = y_test_1day.loc[d:].iloc[:24].iplot(asFigure=True, title='Prediction result of 1-day forecasting')
st.plotly_chart(fig_1day)
fig_1week = y_test_1week.loc[d:].iloc[:24*7].iplot(asFigure=True, title='Prediction result of 1-week forecasting')
st.plotly_chart(fig_1week)

# metrics
st.write(metrics)

