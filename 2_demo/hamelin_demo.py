import streamlit as st
import mpld3
import streamlit.components.v1 as components
import sys
import os
import platform
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0].parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
data_path = os.path.join(ROOT, "0_data")
from scripts.utils import data_path, set_mpl, read_hamelin, add_datetime_features, get_spectrogram

from darts import TimeSeries
from darts.metrics import mape
from darts import concatenate
from darts.utils.timeseries_generation import datetime_attribute_timeseries as dt_attr #I copied the utils into the demo folder
from darts.dataprocessing.transformers import Scaler
from darts.metrics import smape, mape, mase, r2_score
from darts.models import CatBoostModel, LightGBMModel, RegressionEnsembleModel


from sklearn.preprocessing import MaxAbsScaler


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use("agg")
import seaborn as sns
from IPython.display import display
from scipy import signal
import plotly.graph_objects as go


set_mpl()
#Page title
st.set_page_config(
    page_title="Siemens 2023 Hackathon - Tech for Sustainability",
)
st.markdown('# Hamelin dataset')


############ Functions ############ 
@st.cache_data
def read_data():
    """A wrapper for data reading with st.cache_data decorator"""
    # global data_load_state
    data_load_state = st.text('Loading data...')
    energy, weather, metadata = read_hamelin()
    data_load_state.text("Data loading done!")
    return energy, weather, metadata

@st.cache_data
def data_cleaning(energy):
    """Data cleaning"""
    power_substation = energy[['P_substation']].fillna(method='ffill')

    power_substation_spe = get_spectrogram(power_substation, 'P_substation', 24*7, 24, plot = False)
    power_substation_spe = power_substation_spe/np.max(power_substation_spe, axis = 0)


    return power_substation, power_substation_spe


@st.cache_data
def data_preparation(power_substation, power_substation_spe, weather):
    """ time series modelling with random forest and covariates """

    target = TimeSeries.from_dataframe(power_substation, freq = 'H')

    hodidays_covariates = target.add_holidays("DE", state = "NI")['holidays']

    frequency_covariates = TimeSeries.from_dataframe(power_substation_spe[['24.00', '12.00', '6.00']], freq = 'H')


    temperature_history = weather['WEATHER_T']/np.max(weather['WEATHER_T'])
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
    return target, datetime_covatiates, hodidays_covariates, frequency_covariates, temperature_covariate


def data_split(target, tp='2019-09-01', train_ratio=0.6, val_ratio=0.5):
    """split time series for training, validation and test 
    train_ratio = ratio of train - train + val + test
    val_ratio   = ratio of val - val + test
    """

    target, _  = target.split_before(pd.Timestamp(tp))
    train, val = target.split_before(0.6)
    val, test  = val.split_before(0.5)

    scaler = Scaler(scaler=MaxAbsScaler())
    train  = scaler.fit_transform(train)
    val    = scaler.transform(val)
    test   = scaler.transform(test)

    val_len = len(val)

    return train, val, test

def model_train(_train, _val, 
    _datetime_covatiates, _hodidays_covariates, _frequency_covariates, 
    _lags_horizon=24,
    lr=0.007, 
    num_trees=1000, 
    es=5
    ): 
    """Define and train models"""


    model_naive = val.shift(+24) #repeat value from 24 hours ago
    model_naive_train = train.shift(+24) 

    model = CatBoostModel(lags_future_covariates=(_lags_horizon, _lags_horizon), lags_past_covariates=_lags_horizon,
                          learning_rate=lr, num_trees=num_trees, early_stopping_rounds=es)




    cov_args = {"future_covariates": [_datetime_covatiates, _hodidays_covariates],
            "past_covariates": [_frequency_covariates]}

    model.fit(_train, **cov_args)

    # models = [model_naive, model]
    # names  = ['naive',  'catboost']
    models = dict((name, model) for (name, model) in zip(['naive',  'catboost'], [model_naive, model]))
    return models, model_naive_train#, names

def user_input_features():
    horizon_str = st.sidebar.selectbox(
        'What is the horizon for your prediction?',
        ('1 Week (7days)','1 Day'))
    # house_number = st.sidebar.slider(
    #     label='How many houses do you want to aggregate?', 
    #                          min_value=50, max_value=400, value=200, step=50)
    
    # features = pd.DataFrame(data, index=[0])
    # show_raw        = st.sidebar.checkbox('Show raw data')
    # show_preprocess = st.sidebar.checkbox('Show preprocessed data')
    show_train_val  = st.sidebar.checkbox('Show data split', value=True)
    user_inputs = {
        'horizon_str': horizon_str,
        # 'house_number': house_number,
        # 'show_raw': show_raw,
        # 'show_preprocess': show_preprocess,
        'show_train_val': show_train_val
        }
    return user_inputs

def show_split(train, val, test):
    """Draw the split result of the dataset"""
    train = train.pd_dataframe()
    val   = val.pd_dataframe()
    test  = test.pd_dataframe()
    fig  = go.Figure()

    fig.add_trace(go.Scatter(x=train.index, y=train['P_substation'], name="Train"))
    fig.add_trace(go.Scatter(x=val.index, y=val['P_substation'], name="Validate"))
    fig.add_trace(go.Scatter(x=test.index, y=test['P_substation'], name="Test"))
    # fig.update_layout(legend_title_text = "Contestant")
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Energy")
    # fig.show()

    # fig,  ax =  plt.subplots( figsize = (10,5))
    # train.plot(ax = ax, label="train") # x=train.time_index, y=train.
    # val.plot(ax = ax, label="validation")
    # test.plot(ax = ax, label="test")
    # ax.legend()
    plt.tight_layout()
    # fig_html = mpld3.fig_to_html(fig)

    # st.pyplot(fig)
    # ax.xaxis_date()
    # components.html(mpld3.fig_to_html(fig), height=1000, width=1200)
    return fig


############ Layout ############ 
st.sidebar.header('User Input Parameters')



############ Input ############ 
inputs = user_input_features()
horizon_str = inputs['horizon_str']
# horizon_str, house_number = inputs['horizon_str'], inputs['house_number']
# show_raw, show_preprocess, 
show_train_val = inputs['show_train_val']
    # inputs['show_raw'], inputs['show_preprocess'], 


if horizon_str == '1 Day':
    horizon = 24 #used for the modeling
elif horizon_str == '1 Week (7days)':
    horizon = 24*7 #used for the modeling


############ Output ############ 
st.sidebar.write('#### Training model to predict for substations for ', horizon_str)

# read data   
energy, weather, metadata = read_data()#

# Data cleaning
power_substation, power_substation_spe = data_cleaning(energy)

# Prepare dataset for modeling
target, datetime_covatiates, hodidays_covariates, frequency_covariates, temperature_covariate = \
    data_preparation(power_substation, power_substation_spe, weather)

# Split and preprocess the dataset
train, val, test = data_split(target)

# if show_raw:
#     st.markdown('### Showing the first 100 samples of raw data')
#     st.write(data_std.head(100))

# if show_preprocess:
#     st.markdown('### Showing the 100 samples of preprocessed')
#     st.write(power_avg.head(100))


# Draw the split result of the dataset
if show_train_val:
    fig = show_split(train, val, test)
    st.markdown('### Data Split')
    st.plotly_chart(fig)


# # Define and train models
models, model_naive_train = model_train(train, val, datetime_covatiates, hodidays_covariates, frequency_covariates, 
    _lags_horizon=24, lr=0.007, num_trees=1000, es=5) #, names 

# horizon = 24*7*1 #one week ahead


def plot_predictions(train, val, models, horizon):
    fig = go.Figure()

    train_ = train.pd_dataframe()
    val_   = val.pd_dataframe()

    fig.add_trace(go.Scatter(x=train_.tail(horizon).index, y=train_.tail(horizon)['P_substation'], name="train"))
    fig.add_trace(go.Scatter(x=val_.head(horizon).index, y=val_.head(horizon)['P_substation'], name="Validate"))
    
    # fig,  ax =  plt.subplots( figsize = (12,6))
    # train.tail(horizon).plot(ax = ax, label = 'train', lw = 2, alpha = 0.2)
    # val.head(horizon).plot(ax = ax, label = 'val', lw = 2, alpha = 0.2)

    df_out_ = {}
    for name, model in models.items():
        if name == 'naive':
            pred_cv    = model
            pred_train = model_naive_train
            pred_cv_    = pred_cv.pd_dataframe()
            pred_train_ = pred_train.pd_dataframe()
        else:
            pred_cv    = model.predict(horizon)
            pred_train = model.predict(24*7, train[0: -24*7])
            pred_cv_    = pred_cv.pd_dataframe()
            pred_train_ = pred_train.pd_dataframe()

        # print(f'{name} mape - train: {mape(train, pred_train):3g}')
        # print(f'{name} mape - CV: {mape(val, pred_cv):3g}')
        df_out_[f'{name} mape - train'] = round(mape(train, pred_train), 4)
        df_out_[f'{name} mape - CV']    = round(mape(val, pred_cv), 4)
        # pred_train.tail(horizon).plot(ax = ax,  ls='--', lw=1, alpha=0.3, label=name+':train')
        # color =  ax.get_lines()[-1].get_color()
        # pred_cv.head(horizon).plot(ax=ax,  ls='--', lw=2, alpha=0.5, color = color, label=name+':CV')
        fig.add_trace(go.Scatter(x=pred_train_.tail(horizon).index, y=pred_train_.tail(horizon)['P_substation'], name=name+':train'))
        fig.add_trace(go.Scatter(x=pred_cv_.head(horizon).index, y=pred_cv_.head(horizon)['P_substation'], name=name+':CV'))
    df_out = pd.DataFrame.from_dict(df_out_, orient='index')
    df_out.reset_index(inplace=True)
    temp = df_out['index'].str.split(' - ',expand=True)
    df_out['split'] = temp[1]
    df_out['model'] = temp[0].str.split(' ', n=0, expand=True)[0]
    df_out.drop(columns=['index'], inplace=True)
    df_out.set_index(df_out['model'], inplace=True)
    df_out.drop(columns=['model'], inplace=True)
    df_out_show = pd.pivot(df_out, columns='split', values=0)
    df_out_show = df_out_show[['train', 'CV']]
    df_out_show = df_out_show.loc[['catboost', 'naive']]


    #put legend outside
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # # fig.update_layout(legend_title_text = "Contestant")
    fig.update_xaxes(title_text="Time")
    # # fig.update_yaxes(title_text="Number Eaten")
    return fig, df_out_show

fig_pred, df_out_show = plot_predictions(train, val, models, horizon)
st.markdown('##### Prediction')
# st.pyplot(fig_pred)
st.plotly_chart(fig_pred)
st.write("###### MAPE (Mean Absolute Percentage Error)")
st.write(df_out_show)
