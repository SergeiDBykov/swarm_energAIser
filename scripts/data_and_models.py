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


    #make metrics_dict with all metrics for all models
    #for each model dictuonary should have train and test values for each metric
    #e.g. {'Naive': {'train': {'mape': 0.1, 'smape': 0.2}, 'test': {'mape': 0.1, 'smape': 0.2}}} etc


    #list of dict 
    list_metrics_dict = []

    for model_name in ['Naive', 'EAI (no human behaviour)', 'EAI',  'Ensemble']:
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
    # read data
    energy, weather, metadata = read_hamelin_orig()

    # data cleaning
    power_substation = energy[['P_substation']].fillna(method='ffill')

    power_substation_spe = get_spectrogram(power_substation, 'P_substation', 24*7, 24, plot = False)
    power_substation_spe = power_substation_spe/np.max(power_substation_spe, axis = 0)

    target_orig = TimeSeries.from_dataframe(power_substation, freq = 'H')

    hodidays_covariates = target_orig.add_holidays("DE", state = "NI")['holidays']

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
            dt_attr(time_index = target_orig.time_index, attribute =  "hour", one_hot = False, cyclic = False )/24,
            dt_attr(time_index = target_orig.time_index, attribute =  "day_of_week", one_hot = False, cyclic = False )/7,
            dt_attr(time_index = target_orig.time_index, attribute =  "month", one_hot = False, cyclic = False )/12,
            dt_attr(time_index = target_orig.time_index, attribute =  "day_of_year", one_hot = False, cyclic = False )/365,
        ],
        axis="component",
    )

    output_dict = {'original_dataset': energy,
                    'darts_dict': 
                    {
                        'target_orig': target_orig,
                        'hodidays_covariates': hodidays_covariates,
                        'temperature_covariate': temperature_covariate,
                        'frequency_covariates': frequency_covariates,
                        'datetime_covatiates': datetime_covatiates
                    }
                    }


    return output_dict


def predict_hemalin(input_dict, timestamp='2019-09-01', horizon=24*7):
    #copy input_dict to avoid changing it
    input_dict = input_dict.copy()


    target_orig = input_dict['darts_dict']['target_orig']
    hodidays_covariates = input_dict['darts_dict']['hodidays_covariates']
    temperature_covariate = input_dict['darts_dict']['temperature_covariate']
    frequency_covariates = input_dict['darts_dict']['frequency_covariates']
    datetime_covatiates = input_dict['darts_dict']['datetime_covatiates']

    train, test = target_orig.split_before(pd.Timestamp(timestamp))
    test = test.head(horizon*2)

    scaler = Scaler(scaler=MaxAbsScaler())
    train  = scaler.fit_transform(train)
    test   = scaler.transform(test)
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




    cov_args = {"future_covariates": [datetime_covatiates, hodidays_covariates, temperature_covariate],
            "past_covariates": [frequency_covariates],}

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


    #list of dict 
    list_metrics_dict = []

    for model_name in ['Naive', 'EAI (no human behaviour)', 'EAI',  'Ensemble']:
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
def read_trentino(nmin=100, nmax=400, cellsmax=2, ids_residential=[2738, 5201, 5230]):
    telecom, energy, lines = read_trentino_orig()

    lines_sum = lines.reset_index(inplace=False).groupby("LINESET")\
        .agg({"SQUAREID": "count", "NR_UBICAZIONI": "sum"})\
            .rename(columns={"SQUAREID": "n_cells", "NR_UBICAZIONI": "n_customers"})\
            .sort_values(by=["n_cells", "n_customers"])
    
    cells_sum = lines.reset_index(inplace=False).groupby("SQUAREID")\
    .agg({"LINESET": "count", "NR_UBICAZIONI": "sum"})\
            .rename(columns={"LINESET": "n_lines", "NR_UBICAZIONI": "n_customers"})\
            .sort_values(by=["n_lines", "n_customers"])

    # filter
    query = "n_customers >= @nmin and n_customers <= @nmax and n_cells <= @cellsmax"
    lines_select = lines_sum.query(query)

    # find all cells served by these eligible lines
    cell_lines = pd.merge(lines_select, lines, left_index=True, right_on="LINESET")

    # # these eligible cells are served by not only those eligible lines
    # cells_select = pd.merge(cell_lines.drop(columns=["LINESET", "NR_UBICAZIONI", "n_customers", "n_cells"]), cells_sum, left_index=True, right_index=True, how="left")

    energy_ = energy.transpose().reset_index()
    energy_.rename(columns={"index": "LINE"}, inplace=True)

    # energy for each square - lineset combination
    line_energy = pd.merge(lines.reset_index(), energy_, left_on="LINESET", right_on="LINE")

    # each LINESET serves one or more CELLs
    line_cell = lines.reset_index(inplace=False, drop=False).groupby("LINESET").agg({"SQUAREID": "count", "NR_UBICAZIONI": "sum"})
    line_cell.rename(columns = {"SQUAREID": "n_cells", "NR_UBICAZIONI": "n_clients"}, inplace=True)

    # each CELL has one or more LINESET passing
    cell_line = lines.reset_index(inplace=False, drop=False).groupby("SQUAREID").agg({"LINESET": "count", "NR_UBICAZIONI": "sum"})
    cell_line.rename(columns = {"LINESET": "n_lines", "NR_UBICAZIONI": "n_clients"}, inplace=True)
    cell_line.sort_values(by=["n_clients", "n_lines"], inplace=True)

    
    ids_residential.sort()
    lines_residential = list(cell_lines.query("index in @ids_residential")['LINESET'].sort_index())

    # update line_energy by removing energy negative records
    cols_ts = list(set(line_energy.columns) - set(['SQUAREID', 'LINESET', 'NR_UBICAZIONI', 'LINE'])) # timestamp columns

    line_energy = line_energy.loc[(line_energy[cols_ts]>=0).all(axis=1), :]

    # energy_select = line_energy.query("SQUAREID in @ids_residential and LINESET in @lines_residential")



    output_dict = {
        'original_dataset': {
            'telecom': telecom,
        },
        'processed': {
            'line_energy': line_energy,
            'cols_ts': cols_ts,
            'lines': lines_residential,
            'cell_lines': cell_lines
        }
        
        }
    return output_dict


                   
