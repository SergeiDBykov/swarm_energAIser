import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import os
from scipy import signal
import streamlit as st

pd.set_option('display.max_columns', 500)

#rep_path = '/Users/sdbykov/not_work/swarm_energAIser/' #change it here for your local path!
# rep_path = os.getcwd().split('swarm_energAIser')[0]+'swarm_energAIser/'
# data_path = rep_path+'0_data/'

import os
import sys
import platform
from pathlib import Path
FILE = Path(__file__).resolve()
rep_path = FILE.parents[0].parents[0]  # root directory
if str(rep_path) not in sys.path:
    sys.path.append(str(rep_path))  # add ROOT to PATH
if platform.system() != 'Windows':
    rep_path = Path(os.path.relpath(rep_path, Path.cwd()))  # relative
print(rep_path)
data_path = os.path.join(rep_path , '0_data/')


### matplitlib settings

def set_mpl(palette = 'energaiser', desat = 0.8):

    # matplotlib.use('MacOSX') 
    rc = {
        "figure.figsize": [8, 8],
        "figure.dpi": 100,
        "savefig.dpi": 300,
        # fonts and text sizes
        #'font.family': 'sans-serif',
        #'font.family': 'Calibri',
        #'font.sans-serif': 'Lucida Grande',
        'font.style': 'normal',
        "font.size": 18,
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 15,

        # lines
        "axes.linewidth": 1.25,
        "lines.linewidth": 1.75,
        "patch.linewidth": 1,

        # grid
        "axes.grid": True,
        "axes.grid.which": "major",
        "grid.linestyle": "--",
        "grid.linewidth": 0.75,
        "grid.alpha": 0.75,

        # ticks
        "xtick.top": True,
        "ytick.right": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.major.width": 1.25,
        "ytick.major.width": 1.25,
        "xtick.minor.width": 1,
        "ytick.minor.width": 1,
        "xtick.major.size": 7,
        "ytick.major.size": 7,
        "xtick.minor.size": 5,
        "ytick.minor.size": 5,

        'lines.markeredgewidth': 0, #1.5,
        "lines.markersize": 5,
        "lines.markeredgecolor": "k",
        'axes.titlelocation': 'left',
        "axes.formatter.limits": [-3, 3],
        "axes.formatter.use_mathtext": True,
        "axes.formatter.min_exponent": 2,
        'axes.formatter.useoffset': False,
        "figure.autolayout": False,
        "hist.bins": "auto",
        "scatter.edgecolors": "k",
    }




    sns.set_context('notebook', font_scale=1.25)
    matplotlib.rcParams.update(rc)
    if palette == 'shap':
        #colors from shap package: https://github.com/slundberg/shap
        cp = sns.color_palette( ["#1E88E5", "#ff0d57", "#13B755", "#7C52FF", "#FFC000", "#00AEEF"])
        sns.set_palette(cp, color_codes = True, desat = desat)
    elif palette == 'shap_paired':
        #colors from shap package: https://github.com/slundberg/shap, + my own pairing of colors
        cp = sns.color_palette( ["#1E88E5", "#1e25e5", "#ff0d57", "#ff5a8c",  "#13B755", "#2de979","#7C52FF", "#b69fff", "#FFC000", "#ffd34d","#00AEEF", '#3dcaff'])
        sns.set_palette(cp, color_codes = True, desat = desat)
    elif palette == 'energaiser':
        #our color codes
        #5cbd9e
        #195a6a
        #76a362
        #F7ee0c
        #F07318
        #59ae75
        cp = sns.color_palette( ["#f7ee0d", "#6ad0a9", "#f07318", "#5aae74", "#195a6a", "#f74200"])
        sns.set_palette(cp, color_codes = True, desat = desat)

    else:
        sns.set_palette(palette, color_codes = True, desat = desat)
    print('matplotlib settings set')
set_mpl()



def read_hamelin():
    print(f"""
    Loading Hamelin data from {data_path}.
    Houses number removed: 6, 17, 24, 25. 
    Houses with PV: ['13', '15', '26', '33'] 
    HOUSEHOLD and HEATPUMP energy consumption are separated. 
    Resolution: 60min 

    May-June 2019 data for home #34 excluded 
    Data before 2018-05-18 excluded (gaps)
    Data with zero difference between consecutive values dropped (malfunction)

    reutrns:
    df_energy: pd.DataFrame with energy consumption data
    df_weather: pd.DataFrame with weather data
    df_metadata: pd.DataFrame with metadata

    """)

    hamelin_path = data_path+'Hamelin_drive/'

    energy_file = 'hamelin_energy.pkl'
    metadata_file = 'hamelin_metadata.csv'
    weather_file = 'hamelin_weather.pkl'
    twitter_file = 'hamelin_twitter.pkl'

    try:

        df_energy = pd.read_pickle(hamelin_path+energy_file)
        df_metadata = pd.read_csv(hamelin_path+metadata_file,  usecols=[0,1,2], index_col=0)
        df_weather = pd.read_pickle(hamelin_path+weather_file)
    except:
        print('(Windows load)')

        df_energy = pd.read_pickle((hamelin_path+energy_file).replace('/', '\\'))
        df_metadata = pd.read_csv((hamelin_path+metadata_file).replace('/', '\\'),  usecols=[0,1,2], index_col=0)
        df_weather = pd.read_pickle((hamelin_path+weather_file).replace('/', '\\'))


    return [df_energy, df_weather, df_metadata]

def read_london():
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

    london_path = data_path+'London_drive/'

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
    df_twitter = df_twitter.resample('1H').sum() #resample to 1H

    return [df_std, df_tou, df_weather, df_twitter]


def read_trentino():
    print(f"""
    Loading Trentino data from {data_path}.

    Data (energy and telecommunication) are spatially-resolved in grid cells. 
    For the location of grid cells see geojson file in the same folder (trentino-grid.geojson), e.g. plot it on the website https://geojson.io/.

    
    Telecommunication Data (sms, calls, internet)  is not resampled, energy consumption data is resampled to 1H resolution.
    Telecommunication data are only for grid cells with energy consumption data.
    Telecommunication data is for country-code 39 (Italy) only.

    reutrns:
    df_telecom: pd.DataFrame with telecom data (sms, calls, internet)  with arbitrary scale for a given cell and datetime.
    df_line_energy: pd.DataFrame with line energy consumption data. Index - datetime, columns - consumption for each line ID.
    df_line_location: pd.DataFrame with line ID location (cell). Index - cell ID, LINESET - line ID, NR_UBICAZIONI - number of customers on the line
    

    """)



    trentino_path = data_path+'Trentino_drive/'

    telecom_path = trentino_path+'telecom.pkl_gz'
    line_location_path = trentino_path+'line_location.csv'
    line_energy_path = trentino_path+'line_energy.csv'


    try:
        df_telecom = pd.read_pickle(telecom_path, compression='gzip')
        df_line_location = pd.read_csv(line_location_path, index_col=0)
        df_line_energy = pd.read_csv(line_energy_path, index_col=0)
    except:
        print('(Windows load)')
        df_telecom = pd.read_pickle(telecom_path.replace('/', '\\'), compression='gzip')
        df_line_location = pd.read_csv(line_location_path.replace('/', '\\'), index_col=0)
        df_line_energy = pd.read_csv(line_energy_path.replace('/', '\\'), index_col=0)

    df_line_energy.index = pd.to_datetime(df_line_energy.index) #todo is this correct?


    return [df_telecom, df_line_energy, df_line_location]

def add_datetime_features(dfs):
    dfs_out = []
    for df in dfs:
        df = df.copy()
        df['hour'] = df.index.hour
        df['weekday'] = df.index.weekday
        df['month'] = df.index.month
        df['year'] = df.index.year
        df['doy'] = df.index.dayofyear

        df['season'] = df['month'].apply(lambda x: 'winter' if x in [12, 1, 2] else 'spring' if x in [3, 4, 5] else 'summer' if x in [6, 7, 8] else 'autumn')
        df['workday'] = df['weekday'].apply(lambda x: 'workday' if x in [0, 1, 2, 3, 4] else 'weekend')

        dfs_out.append(df)
    
    return dfs_out






def get_spectrogram(df, target_col, window_size, overlap, plot = False):
    """
    gets a spectrogram of a signal.
    returns a pd.DataFrame with the spectrogram: columns - frequencies, index - time, values - power.
    if overlap >1, the spectrogram is interpolated linearly to the original time index of the signal.
    """

    freq, time, S = signal.spectrogram(df[target_col], fs=1,          
                                    nperseg=window_size, noverlap=overlap,
                                    scaling='spectrum', mode='psd', detrend='linear', window = 'tukey')
        
    S_df = pd.DataFrame(S, index = freq, columns = time)
    S_df = S_df.T
    S_df.index = pd.Series(S_df.index).apply(lambda x: df.index[int(x)]).values
    S_df.drop(columns = 0, inplace = True) #drop 0 Hz
    S_df.columns = ["{:.2f}".format(x) for x in list((1/S_df.columns))]

    #reindex: if the timestep is not in the spectrogram, fill it with NaN and interpolate linearly
    S_df = S_df.reindex(df.index).interpolate('linear')

    if plot:
        fig, [ax1, ax2] =  plt.subplots(nrows=2, ncols = 1, sharex = True, gridspec_kw = {'hspace':0, 'height_ratios': [1,2]}, figsize = (12,12))


        ax2.pcolormesh(S_df.index, S_df.columns, np.log10(S_df.T), shading='gouraud', cmap='PiYG')
        #skip every second y tick and totate it by 45 degrees
        ax2.set_yticks(S_df.columns[::3])
        ax2.set_yticklabels(ax2.get_yticklabels(), rotation=45,)

        ax1.plot(df.index, df.iloc[:, 0], 'k-', alpha=0.5, label = 'original')

        plt.ylabel('Frequency [h]')
        plt.xlabel('Time')
        plt.show()
    return S_df
