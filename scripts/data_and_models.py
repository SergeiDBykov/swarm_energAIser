import sys
sys.path.append('../../')

from scripts.utils import set_mpl
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



import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr




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




def predict_london(input_dict, timestamp = '2013-03-07', horizon = 24*7):
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

    lgbm_args = {'verbose':0, "force_col_wise":True}
    model = LightGBMModel(lags_future_covariates = lags_horizon_future , lags_past_covariates=lags_horizon_past, **lgbm_args  )                                
    model.fit(train, **cov_args)

    
    
    model_no_human = LightGBMModel(lags_future_covariates = lags_horizon_future , lags_past_covariates=None, **lgbm_args) 

    model_no_human.fit(train, future_covariates = [datetime_covatiates, temperature_covariate])

    output_dict = {'Naive': {'model': model_naive},
            'EAI': {'model': model},
            'EAI (no human behaviour)': {'model': model_no_human},
            }




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


    metrics_dict = {}

    for metrics in [mape, smape]:
        for model_name in ['Naive', 'EAI (no human behaviour)', 'EAI',  'Ensemble']:
            pred_train = output_dict[model_name]['pred_train']
            pred_test = output_dict[model_name]['pred_test']

            metrics_train =metrics(pred_train, output_dict['train'].tail(horizon))
            metrics_test = metrics(pred_test, output_dict['test'].head(horizon))

            metrics_dict[model_name] = {'train': metrics_train, 'test': metrics_test}

    df_metrics = pd.DataFrame(metrics_dict).T

    #two decimal points format but not using round to avoid rounding errors
    df_metrics = df_metrics.applymap(lambda x: f'{x:.2f}')

    df_metrics = df_metrics.T

    output_dict['metrics'] = df_metrics

    return output_dict

