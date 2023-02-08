import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import glob
import h5py
from tqdm import tqdm

import warnings
from tables import NaturalNameWarning
warnings.filterwarnings('ignore', category=NaturalNameWarning)

from .utils import data_path



def read_timeseries_s22_year(resol: str = '60min', year: str = '2018') -> pd.DataFrame:
    data_path_s22 = data_path+'Schlemminger2022/'
    print(f'READ FROM {data_path_s22}')
    assert resol in ['10s', '1min', '15min', '60min'], f'Resolution {resol} is not available. Please choose from 10s, 1min, 15min, 60min'

    available_home_numbers = ['3', '4', '5',  '7', '8', '9', '10', '11', '12', '14', '16', '18',
                                '19', '20', '21', '22', '23', '27', '28', '29', '30', '31', '32',
                                '34', '35', '36', '37', '38', '39', '40']
    available_home_numbers_PV = ['13','15', '26', '33']

    #house 24 and 25  ignored since they have less than 2 years of data
    #house 6,17 removed since it has only around 50% of data


    filename_sub = data_path_s22+year+'_data_'+'spatial.hdf5'
    df_substation = pd.read_hdf(filename_sub, key=f'SUBSTATION/{resol}')
    df_substation.index = pd.to_datetime(df_substation.index, unit = 's')
    df_substation = df_substation[['P_TOT']]
    df_substation.columns = ['P_substation']

    df_list = []

    for house_num in available_home_numbers:


        filename = data_path_s22+year+'_data_'+resol+'.hdf5'
        df_1 = pd.read_hdf(filename, key=f'NO_PV/SFH{house_num}/HOUSEHOLD') #HEATPUMP and HOUSEHOLD are summed
        df_1.index = pd.to_datetime(df_1.index, unit = 's')
        df_1 = df_1[[f'P_TOT']]
        df_1.columns = [f'P_HOME_{house_num}']

        df_2 = pd.read_hdf(filename, key=f'NO_PV/SFH{house_num}/HEATPUMP')
        df_2.index = pd.to_datetime(df_2.index, unit = 's')
        df_2 = df_2[[f'P_TOT']]
        df_2.columns = [f'P_HEAT_{house_num}']


        df_comb = df_1.join(df_2)
        df_comb['P_TOT_'+house_num] = df_comb['P_HOME_'+house_num] + df_comb['P_HEAT_'+house_num]

        df_list.append(df_comb)
    
    for house_num in available_home_numbers_PV:
            
        filename = data_path_s22+year+'_data_'+resol+'.hdf5'
        df_1 = pd.read_hdf(filename, key=f'WITH_PV/SFH{house_num}/HOUSEHOLD')
        df_1.index = pd.to_datetime(df_1.index, unit = 's')
        df_1 = df_1[[f'P_TOT']]
        df_1.columns = [f'P_HOME_{house_num}']


        df_2 = pd.read_hdf(filename, key=f'WITH_PV/SFH{house_num}/HEATPUMP')
        df_2.index = pd.to_datetime(df_2.index, unit = 's')
        df_2 = df_2[[f'P_TOT']]
        df_2.columns = [f'P_HEAT_{house_num}']


        df_comb = df_1.join(df_2)
        df_comb['P_TOT_'+house_num] = df_comb['P_HOME_'+house_num] + df_comb['P_HEAT_'+house_num]

        df_list.append(df_comb)
        
        


    df = pd.concat(df_list, axis=1)
    df = pd.concat([df, df_substation], axis=1)

    print(f'DATA LOADED FROM {data_path_s22}. \n Houses number removed: 24, 25. \n Houses with PV: {available_home_numbers_PV} \n HOUSEHOLD and HEATPUMP energy consumption are separated. \n Resolution: {resol} \n Years: {year}')

    return df

def read_timeseries_s22(resol: str = '60min')-> pd.DataFrame:
    df_2018 = read_timeseries_s22_year(resol=resol, year='2018')
    df_2019 = read_timeseries_s22_year(resol=resol, year='2019')
    df_2020 = read_timeseries_s22_year(resol=resol, year='2020')
    df = pd.concat([df_2018, df_2019, df_2020], axis=0)

    df.dropna(inplace=True, how='all')

    return df




def read_weather_s22(resol = '60min'):
    data_path_s22 = data_path+'Schlemminger2022/'
    print(f'READ FROM {data_path_s22}')
    
    years = ['2018', '2019', '2020']
    cols = ['APPARENT_TEMPERATURE_TOTAL', 'ATMOSPHERIC_PRESSURE_TOTAL', 'PRECIPITATION_RATE_TOTAL', 'RELATIVE_HUMIDITY_TOTAL', 'SOLAR_IRRADIANCE_GLOBAL', 'TEMPERATURE_TOTAL']

    cols_aliases = {'APPARENT_TEMPERATURE_TOTAL': 'WEATHER_T_APP', 'ATMOSPHERIC_PRESSURE_TOTAL': 'WEATHER_P_ATM', 'PRECIPITATION_RATE_TOTAL': 'WEATHER_PREC_RATE', 'RELATIVE_HUMIDITY_TOTAL': 'WEATHER_H_REL', 'SOLAR_IRRADIANCE_GLOBAL': 'WEATHER_I_SOLAR', 'TEMPERATURE_TOTAL': 'WEATHER_T'}

    df_years = [] 
    for year in years:
        df_list = []
        filename = data_path_s22+year+'_weather.hdf5'
        print(f'LOADING WEATHER: {filename}')
        for col in cols:
            #print(f'\t fetching {col}...')
            df = pd.read_hdf(filename, key=f'WEATHER_SERVICE/IN/WEATHER_{col}')
            df.index = pd.to_datetime(df.index, unit = 's')
            df.name = cols_aliases[col]
            df.columns = [cols_aliases[col]]

            #index has duplicates, drop them. eg in 2018, there are 40 duplicates in df_list at 22 and 23 hours. keep first
            df = df[~df.index.duplicated(keep='first')]

            df.columns = [cols_aliases[col]]

            df_list.append(df)

        df_year = pd.concat(df_list, axis=1)
        df_years.append(df_year)
    
    df = pd.concat(df_years, axis=0)

    df.dropna(inplace=True, how='all')
    df = df.resample(resol).mean()

    return df




def read_london_file(filename: str) -> pd.DataFrame:
    data_path_london = data_path+'London/'
    df = pd.read_csv(data_path_london+filename, skiprows=1,delim_whitespace=False,header=None, usecols=[0,1,2,3], names=['id','mode','date','power'])
    df['date']=pd.to_datetime(df['date'].astype(str) ,format='%Y-%m-%d %H:%M:%S.%f')
    df['power'] = pd.to_numeric(df['power'], errors='coerce')

    return df


def read_london(file_limit = 10):
    data_path_london = data_path+'London/'

    london_files   = glob.glob(data_path_london+'*.csv')

    dfs = []
    for file in tqdm(london_files[:file_limit]):
        fname = file.split('/')[-1]
        df = read_london_file(fname)
        dfs.append(df)

    df_total = pd.concat(dfs, axis=0)
    
    return df_total
