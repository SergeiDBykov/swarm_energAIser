import sys
# sys.path.append('../')
import os
import platform
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0].parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from scripts.utils import set_mpl, get_spectrogram
set_mpl()

from scripts.utils import read_london as read_london_orig
from scripts.utils import read_hamelin as read_hamelin_orig
from scripts.utils import read_trentino as read_trentino_orig


from darts import TimeSeries
from darts.metrics import mape, smape
from darts import concatenate
from darts.utils.timeseries_generation import datetime_attribute_timeseries as dt_attr

from darts.dataprocessing.transformers import Scaler
from darts.metrics import smape, mape, mase
from sklearn.preprocessing import MaxAbsScaler
from darts.models import LightGBMModel, LinearRegressionModel


import geopandas as gpd
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
import contextily as ctx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
data_path = os.path.join(ROOT , '0_data/')


#add decorator st.cache to make it faster to read_london

#@st.cache(hash_funcs={xr.core.dataarray.DataArray: id})
@st.cache_data
def read_london(num_homes = 300):
    london_std, _, london_weather, london_twitter = read_london_orig()

    london_std = london_std.query('index>"2012-01-01"')
    london_weather = london_weather.query('index>"2012-01-01"')
    london_twitter = london_twitter.query('index>"2012-01-01"')
        
    missed_frac = london_std.isna().mean(axis = 0).sort_values(ascending = False)
    #drop columns with more than xx% missing values
    data = london_std.drop(missed_frac[missed_frac > 0.3].index, axis = 1)
    homes_col = data.columns
    cols = data.columns


    #select homes with mean energy consumption close to  0.1
    data_cluster = data[cols[ (data[cols].mean() > 0.1) & (data[cols].mean() < 0.15) ]]
    data_cluster = data_cluster.T.sample(num_homes, random_state = 42).T
    power_avg = data_cluster.mean(axis=1).to_frame()
    power_avg.columns = ['power_avg']




    target_orig = TimeSeries.from_dataframe(power_avg, freq = 'H')

    hodidays_covariates = target_orig.add_holidays("UK")['holidays']


    temperature_history = london_weather['temp']/np.max(london_weather['temp'])
    #perturb temperature to avoid perfect correlation
    rolling_std = temperature_history.rolling(24*3).std()
    temperature_forecast = temperature_history + np.random.normal(0, rolling_std, size = len(temperature_history))

    temperature_covariate  = TimeSeries.from_dataframe(temperature_forecast.to_frame(), freq = 'H') #we can use it as past and future, although it is not perfect for past predictions

    #twitter_covariate = TimeSeries.from_dataframe(london_twitter[['tweets_total']], freq = 'H')
    twitter_covariate = TimeSeries.from_dataframe(london_twitter, freq = 'H')

    datetime_covatiates = concatenate(
        [
            dt_attr(time_index = target_orig, attribute =  "hour", one_hot = False, cyclic = True ),
            dt_attr(time_index = target_orig, attribute =  "day_of_week", one_hot = False, cyclic = True ),
            dt_attr(time_index = target_orig, attribute =  "month", one_hot = False, cyclic = True ),
            dt_attr(time_index = target_orig, attribute =  "day_of_year", one_hot = False, cyclic = True ),
        ],
        axis="component",
    )


    output_dict = {'original_dataset': power_avg,
                    'darts_dict': 
                    {
                        'target_orig': target_orig,
                        'hodidays_covariates': hodidays_covariates,
                        'temperature_covariate': temperature_covariate,
                        'twitter_covariate': twitter_covariate,
                        'datetime_covatiates': datetime_covatiates
                    }
                    }



    return output_dict




def predict_london(input_dict, timestamp = '2013-03-07', horizon = 24*7, price_weights = False):
    #copy input_dict to avoid changing it
    input_dict = input_dict.copy()


    target_orig = input_dict['darts_dict']['target_orig']
    hodidays_covariates = input_dict['darts_dict']['hodidays_covariates']
    temperature_covariate = input_dict['darts_dict']['temperature_covariate']
    twitter_covariate = input_dict['darts_dict']['twitter_covariate']
    datetime_covatiates = input_dict['darts_dict']['datetime_covatiates']



    train, test = target_orig.split_before(pd.Timestamp(timestamp))
    test = test.head(horizon*2)

    scaler = Scaler(scaler=MaxAbsScaler())
    train = scaler.fit_transform(train)
    test = scaler.transform(test)
    target = scaler.transform(target_orig)



    lags_horizon = list(np.hstack([np.arange(1, 25), [168]]))
    lags_horizon = [int(x) for x in lags_horizon]
    lags_horizon_past = [-int(x) for x in lags_horizon]
    lags_horizon_past.sort()
    lags_horizon_future = lags_horizon + lags_horizon_past



    model_naive = LinearRegressionModel(lags_past_covariates=[-168])
    model_naive.fit(train, past_covariates = target)
    model_naive.model.coef_ = np.array([[1.0]])
    model_naive.model.intercept_=0.0




    cov_args = {"future_covariates": [datetime_covatiates,hodidays_covariates, temperature_covariate],
            "past_covariates": [twitter_covariate],}

    lgbm_args = {'verbose':-1, "force_col_wise":True, 'n_estimators':85, 'max_depth': 4}
    print('LGBM ARGS', lgbm_args)
    model = LightGBMModel(lags_future_covariates = lags_horizon_future , lags_past_covariates=lags_horizon_past, **lgbm_args  )                                
    model.fit(train, **cov_args)

    

    
    model_no_human = LightGBMModel(lags_future_covariates = lags_horizon_future , lags_past_covariates=None, **lgbm_args) 

    model_no_human.fit(train, future_covariates = [datetime_covatiates, temperature_covariate])

    output_dict = {'Naive': {'model': model_naive},
            'EAI': {'model': model},
            'EAI (no human behaviour)': {'model': model_no_human},
            }


    if price_weights:
        print('Using price weights')
        hourly_weights = {0:122,
                            1:112,
                            2:111,
                            3:110,
                            4:111,
                            5:122,
                            6:140,
                            7:147,
                            8:164,
                            9:158,
                            10:144,
                            11:137,
                            12:126,
                            13:125,
                            14:125,
                            15:131,
                            16:141,
                            17:147,
                            18:159,
                            19:162,
                            20:148,
                            21:136,
                            22:130,
                            23:122}


        weights = np.ones(len(train))*1
        weights = weights[horizon:]

        hours = train[horizon:].time_index.hour
        #assign weights according to the hour of the training datum
        weights = np.array([hourly_weights[x] for x in hours])

    
        model_weighted = LightGBMModel(lags_future_covariates = lags_horizon_future , lags_past_covariates=lags_horizon_past, **lgbm_args  )                                
        model_weighted.fit(train, **cov_args, sample_weight = weights)

        
        
        model_no_human_weighted = LightGBMModel(lags_future_covariates = lags_horizon_future , lags_past_covariates=None, **lgbm_args) 

        model_no_human_weighted.fit(train, future_covariates = [datetime_covatiates, temperature_covariate], sample_weight = weights)

        output_dict['EAI_w'] = {'model': model_weighted}
        output_dict['EAI_w (no human behaviour)'] = {'model': model_no_human_weighted}




    for model_name, model_params in output_dict.items():
            model = model_params['model']
            pred_test = model.predict(horizon, train)
            pred_train = model.predict(horizon, train.head(-horizon), )

            output_dict[model_name]['pred_test'] = pred_test
            output_dict[model_name]['pred_train'] = pred_train


    pred_test_ensemble = (output_dict['Naive']['pred_test']+output_dict['EAI']['pred_test'])/2

    pred_train_ensemble = (output_dict['Naive']['pred_train']+output_dict['EAI']['pred_train'])/2

    output_dict['Ensemble'] = {'pred_test': pred_test_ensemble, 'pred_train': pred_train_ensemble} 

    output_dict['train'] = train.tail(horizon*2)
    output_dict['test'] = test


    #make metrics_dict with all metrics for all models
    #for each model dictuonary should have train and test values for each metric
    #e.g. {'Naive': {'train': {'mape': 0.1, 'smape': 0.2}, 'test': {'mape': 0.1, 'smape': 0.2}}} etc


    #list of dict 
    list_metrics_dict = []

    #for model_name in ['Naive', 'EAI (no human behaviour)', 'EAI',  'Ensemble']:
    for model_name in output_dict.keys():
        if model_name in ['train', 'test']:
            continue
        for metrics in [mape, smape]:
            metrics_name = metrics.__name__
            pred_train = output_dict[model_name]['pred_train']
            pred_test = output_dict[model_name]['pred_test']

            metrics_train =metrics(pred_train, output_dict['train'].tail(horizon))
            metrics_test = metrics(pred_test, output_dict['test'].head(horizon))

            list_metrics_dict.append({'model': model_name, 'metrics': metrics_name, 'train': metrics_train, 'test': metrics_test})



    df_metrics = pd.DataFrame(list_metrics_dict).set_index(['metrics', 'model'])

    #two decimal points format but not using round to avoid rounding errors
    df_metrics = df_metrics.applymap(lambda x: f'{x:.2f}')


    #df_metrics = df_metrics.T

    if horizon == 24*7:
        # make test be 'Week ahead' and train be 'Week before'
        df_metrics = df_metrics.rename(columns={'train': 'Last week', 'test': 'Week ahead'})
    elif horizon == 24:
        # make test be 'Day ahead' and train be 'Day before'
        df_metrics = df_metrics.rename(columns={'train': 'Last day', 'test': 'Day ahead'})
    else:
        pass
    
    df_metrics = df_metrics.sort_index(level=0)

    output_dict['metrics'] = df_metrics


    return output_dict



@st.cache_data
def read_hamelin():
    energy, weather, metadata, twitter, trends = read_hamelin_orig()
   
    HOME_mean =  energy.loc[:, energy.columns.str.contains('HOME')].mean(axis=1).resample('1D').mean()
    TOT_mean =  energy.loc[:, energy.columns.str.contains('TOT')].mean(axis=1).resample('1D').mean()
    HEAT_mean =  energy.loc[:, energy.columns.str.contains('HEAT')].mean(axis=1).resample('1D').mean()
    substation =  energy.loc[:, energy.columns.str.contains('substation')].resample('1D').mean()


    HOME_mean = HOME_mean.rename('HOME').to_frame()
    TOT_mean = TOT_mean.rename('TOT').to_frame()
    HEAT_mean = HEAT_mean.rename('HEAT').to_frame()
    substation = substation
    
    excel = trends[['Microsoft Excel']]

    output_dict = {
        'Home': HOME_mean,
        'Heat': HEAT_mean,
        'Total': TOT_mean,
        'Substation': substation,
        'Excel': excel,
        }

    return output_dict


@st.cache_data
def read_trentino():
    telecom, energy, lines, twitter = read_trentino_orig()

    gridfile = data_path+'Trentino_drive/trentino-grid.geojson'
    grid = gpd.read_file(gridfile)
    grid = grid.to_crs(epsg=3857)
    grid['NR_UBICAZIONI'] = grid['cellId'].map(lines.reset_index(drop = False, inplace = False).groupby('SQUAREID')['NR_UBICAZIONI'].sum())
    output_dict = {
        "telecom": telecom, 
        "energy": energy, 
        "lines": lines, 
        "twitter": twitter,
        "grid": grid 
    }
    return output_dict

def plot_map_orig(grid):
    fig,  ax =  plt.subplots(1,1, figsize = (5,5))
    grid.plot(ax = ax, color = 'r', alpha = 0.2) 
    grid.plot(ax = ax, column = 'NR_UBICAZIONI', legend = False, alpha = 0.5, cmap = 'viridis', edgecolor = 'k', linewidth = 0.1)
    ctx.add_basemap(ax, source=ctx.providers.Stamen.TonerLite)
    # ax.set_title('Number of customers per grid cell')
    ax.axis('off')
    plt.tight_layout()
    return fig

def plot_select_map(grid, cell_id):
    grid['select'] = 'others'
    grid.loc[grid['cellId']==cell_id, 'select'] = 'selected'

    fig,  ax =  plt.subplots(1,1, figsize = (5,5))
    grid.plot(ax=ax, color = 'r', alpha = 0.2) 
    grid.plot(ax=ax, column = 'select', legend=False, alpha = 0.5, cmap = 'viridis', edgecolor = 'k', linewidth = 0.1)
    ctx.add_basemap(ax, source=ctx.providers.Stamen.TonerLite)

    # ax.set_title(f'Cell {cell_id}')
    ax.axis('off')
    plt.tight_layout()
    return fig

# def get_cell_info(cell_id):


def plot_correlation(cell_id, energy, telecom, twitter, cell2line):
    
    line_id   = cell2line[cell_id]
    internet  = telecom.query('CellID==@cell_id').drop('CellID', axis = 1)['internet'].resample('1H').mean()
    smsin     = telecom.query('CellID==@cell_id').drop('CellID', axis = 1)['smsin'].resample('1H').mean()
    smsout    = telecom.query('CellID==@cell_id').drop('CellID', axis = 1)['smsout'].resample('1H').mean()
    callout   = telecom.query('CellID==@cell_id').drop('CellID', axis = 1)['callout'].resample('1H').mean()
    callin    = telecom.query('CellID==@cell_id').drop('CellID', axis = 1)['callin'].resample('1H').mean()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.update_layout(
            autosize=False,
            width=1200,
            height=600,
            xaxis=dict(title="Date"),
            yaxis=dict(title='Power consumption (scaled)'),
            yaxis2=dict(title='Proxy data',
                    overlaying='y',
                    side='right'))

    # # create a plotly figure object
    # # add traces for each line plot
    fig.add_trace(go.Scatter(x=energy.index, y=energy[line_id]*3, name='Energy'), secondary_y=False)
    fig.add_trace(go.Scatter(x=twitter.index, y=twitter['tweets_total'], name='Twitter - Total Tweets'), secondary_y=True)
    fig.add_trace(go.Scatter(x=callout.index, y=callout, name='Telecom - Call'), secondary_y=False)
    # fig.add_trace(go.Scatter(x=callin.index, y=callin, name='Telecom - Call-in'))
    # fig.add_trace(go.Scatter(x=smsin.index, y=smsin, name='Telecom - SMS-in'))
    fig.add_trace(go.Scatter(x=smsout.index, y=smsout, name='Telecom - SMS'), secondary_y=False)

    #fig.add_trace(pl.Scatter(x=internet.index, y=internet, name='INternet'))

    # update layout with axis limits
    fig.update_layout(xaxis_range=[energy.index.min(), energy.index.max()])

    return fig
