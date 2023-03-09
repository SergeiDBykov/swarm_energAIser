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

from scripts.utils import set_mpl
# from scripts.data_and_models import read_london, predict_london, read_hamelin, read_trentino
from scripts.data_and_models import *
import streamlit as st
#import streamlit.components.v1 as components
#import mpld3 #conda install mpld3
import plotly
import plotly.express as px
import plotly.graph_objects as go

#streamlit run dashboard.py

#make streamlit wide mode
st.set_page_config(layout="wide", page_title = "EnergAIser forecasting", page_icon="📈",)


st.title("EnergAIser")
#st.markdown("Selected dataset")



country = st.sidebar.selectbox(label = "Selected dataset", index = 0,
                               options = ['<select dataset>', 'London, UK', 'Hamelin, DE', 'Trentino, IT'])


if country=='London, UK':

    horizon = st.sidebar.selectbox(label = "Select forecast horizon", index = 1,
                                options = ['Day ahead', 'Week ahead'])
    horizon = {'Day ahead':24, 'Week ahead': 24*7, '<select horizon>': None}[horizon]

    num_homes = st.sidebar.selectbox(label = "Select substation size", index = 1,
                                options = ['100 Households', '300 Households', '500 Households'])
    st.subheader(f'London, UK, {num_homes}')
    num_homes = {'100 Households':100, '300 Households': 300, '500 Households': 500}[num_homes]


    london_dict = read_london(num_homes = num_homes)


    df = london_dict['original_dataset'].resample('1D').mean()

    fig = go.Figure()
    fig.update_layout(
            autosize=False,
            width=1000,
            height=500)

    fig.add_trace(go.Scatter(x=df.index, y=df['power_avg'],
                        mode='lines',
                        name='Daily average'))
    
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Power consumption (scaled)")

    st.plotly_chart(fig)
    

    st.markdown("Select a forecast starting date")
    start_date = st.date_input('Start date', value = df.index.mean().date(), min_value = df.index.min(), max_value = df.index.max())

    #and update the plot with the selected date as a vertical line
    fig.add_vline(x=start_date)


    plt_twitter =  st.checkbox('Plot tweets volume', value = 0)
    show_full_metrics = st.checkbox('Show full metrics', value = 0)

    start_forecast = st.button('Forecast')
    if start_forecast:
        st.write('Forecasting...')

        predict_dict = predict_london(london_dict, start_date, horizon = horizon)


        train_df = predict_dict['train'].pd_dataframe()
        test_df = predict_dict['test'].pd_dataframe()


        fig = go.Figure()
        fig.update_layout(
            autosize=False,
            width=1000,
            height=500)

        #fig.add_trace(go.Scatter(x=train_df.tail(24).index, y=train_df.tail(24)['power_avg'],
        #                    mode='lines',
        #                    name='train'))
        fig.add_trace(go.Scatter(x=test_df.head(horizon).index, y=test_df.head(horizon)['power_avg'],
                            mode='lines',
                            name='Real data'))   

        #if show_full_metrics:
        #    list_of_models = ['Naive', 'EAI (no human behaviour)', 'EAI',  'Ensemble']
        #else:
        list_of_models = ['Naive', 'EAI (no human behaviour)', 'EAI']

        for model_name in list_of_models:
            model_df = predict_dict[model_name]['pred_test'].pd_dataframe()
            fig.add_trace(go.Scatter(x=model_df.index, y=model_df['power_avg'],
                            mode='lines',
                            name=model_name))
        
        fig.update_layout(
            title="Train and test data",
            xaxis_title="Date",
            yaxis_title="Power consumption (scaled)",
            legend_title="Legend Title",
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="RebeccaPurple"
            )
        )


        st.plotly_chart(fig)

        metrics = predict_dict['metrics']
        st.write('MAPE error metrics:')
        st.write(metrics.query('model in ["Naive", "EAI"]').loc['mape', :][['Week ahead']])


        if show_full_metrics:
            st.write('MAPE error metrics:')
            st.write(metrics.loc['mape', :][['Week ahead', 'Last week']])
            st.write('SMAPE error metrics:')
            st.write(metrics.loc['smape', :][['Week ahead', 'Last week']])


        if plt_twitter:
            #plot twitter data and energy consumption on the same figure with two y axes
            tweets_df = london_dict['darts_dict']['twitter_covariate'].pd_dataframe()
            twets_clipped = tweets_df.loc[start_date:start_date+pd.Timedelta(hours = horizon)]
            
            fig = plotly.subplots.make_subplots(specs=[[{"secondary_y": True}]])

            # Add traces
            fig.add_trace(go.Scatter(x=twets_clipped.index, y=twets_clipped['tweets_total'],
                                mode='lines',
                                name='tweets'),
                                secondary_y=False)

            fig.add_trace(go.Scatter(x=test_df.head(horizon).index, y=test_df.head(horizon)['power_avg'],
                            mode='lines',
                            name = 'Energy'), secondary_y=True)   

            # Set x-axis title
            fig.update_xaxes(title_text="Date")

            # Set y-axes titles
            fig.update_yaxes(title_text="Tweets per hour", secondary_y=False)
            fig.update_yaxes(title_text="Hourly energy consumption (scaled)", secondary_y=True)
            #remove grid from the secondary y axis
            fig.update_yaxes(showgrid=False, secondary_y=True)


            st.plotly_chart(fig)




elif country=='Hamelin, DE':
    st.subheader(f'Hamelin, DE, 70 household substation')
    hamelin_dict = read_hamelin()

    home_power = hamelin_dict['Home']
    heating_power = hamelin_dict['Heat']
    total_power = hamelin_dict['Total']
    substation_power = hamelin_dict['Substation']
    excel_trend = hamelin_dict['Excel']


    df = substation_power.resample('1D').mean()

    fig = go.Figure()
    fig.update_layout(
            autosize=False,
            width=1000,
            height=500)

    fig.add_trace(go.Scatter(x=df.index, y=df['P_substation']/df['P_substation'].mean(),
                        mode='lines',
                        name='Energy consumption (scaled)'))
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Energy consumption (scaled)")


    st.plotly_chart(fig)



    trends = st.button('Plot Google Trends')
    if trends:
        st.write('Average household consumption (40 homes)')

        fig = plotly.subplots.make_subplots(specs=[[{"secondary_y": True}]])
        fig.update_layout(
            autosize=False,
            width=1000,
            height=500)

        fig.add_trace(go.Scatter(x=home_power.index, y=home_power['HOME']/home_power['HOME'].mean(),
                            mode='lines',
                            name='Household consumption (no heating)'),
                            secondary_y=False)

        #fig.add_trace(go.Scatter(x=heating_power.index, y=heating_power['HEAT']/#heating_power['HEAT'].mean(),
        #                    mode='lines',
        #                    name='Heating only consumption'),
        #                    secondary_y=False)

        fig.add_trace(go.Scatter(x=excel_trend.index, y=excel_trend['Microsoft Excel']+excel_trend['Microsoft Excel'].min(),
                        mode='lines',
                        name = 'Microsoft Excel populatiry'), secondary_y=True)   

        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Energy consumption (scaled)", secondary_y=False)
        fig.update_yaxes(title_text="Populatiry", secondary_y=True)
        fig.update_yaxes(showgrid=False, secondary_y=True)

        st.plotly_chart(fig)



# elif country=='Hamelin, DE':
#     horizon = st.sidebar.selectbox(label = "Select forecast horizon", index = 0,
#                                 options = ['<select horizon>', 'Day ahead', 'Week ahead'])
#     horizon = {'Day ahead':24, 'Week ahead': 24*7, '<select horizon>': None}[horizon]

#     st.subheader('Hamelin, DE') 

#     hamelin_dict = read_hamelin()

#     df = hamelin_dict['original_dataset'].resample('1D').mean()

#     fig = go.Figure()
#     fig.update_layout(
#             autosize=False,
#             width=1000,
#             height=500)

#     fig.add_trace(go.Scatter(x=df.index, y=df['P_substation'],
#                         mode='lines',
#                         name='Daily average'))
#     st.plotly_chart(fig)

#     st.markdown("Select a forecast starting date")
#     start_date = st.date_input('Start date', value = df.index.mean().date(), min_value = df.index.min(), max_value = df.index.max())

#     #and update the plot with the selected date as a vertical line
#     fig.add_vline(x=start_date)

#     show_full_metrics = st.checkbox('Show full metrics', value = 0)
#     #ask whether to proceed with the forecast
#     if st.button('Forecast'):
#         st.write('Forecasting...')

#         predict_dict = predict_hamelin(hamelin_dict, start_date, horizon = horizon)


#         train_df = predict_dict['train'].pd_dataframe()
#         test_df = predict_dict['test'].pd_dataframe()


#         fig = go.Figure()
#         fig.update_layout(
#             autosize=False,
#             width=1000,
#             height=500)

#         fig.add_trace(go.Scatter(x=train_df.tail(24).index, y=train_df.tail(24)['P_substation'],
#                             mode='lines',
#                             name='train'))
#         fig.add_trace(go.Scatter(x=test_df.head(horizon).index, y=test_df.head(horizon)['P_substation'],
#                             mode='lines',
#                             name='test'))   

#         for model_name in ['Naive', 'EAI (no human behaviour)', 'EAI',  'Ensemble']:
#             model_df = predict_dict[model_name]['pred_test'].pd_dataframe()
#             fig.add_trace(go.Scatter(x=model_df.index, y=model_df['P_substation'],
#                             mode='lines',
#                             name=model_name))
            
        
        
#         fig.update_layout(
#             title="Train and test data",
#             xaxis_title="Date",
#             yaxis_title="Power consumption",
#             legend_title="Legend Title",
#             font=dict(
#                 family="Courier New, monospace",
#                 size=18,
#                 color="RebeccaPurple"
#             )
#         )


#         st.plotly_chart(fig)

#         metrics = predict_dict['metrics']
#         st.write('MAPE error metrics:')
#         st.write(metrics.query('model in ["Naive", "EAI"]').loc['mape', :][['Week ahead']])


#         if show_full_metrics:
#             st.write('MAPE error metrics:')
#             st.write(metrics.loc['mape', :][['Week ahead', 'Last week']])
#             st.write('SMAPE error metrics:')
#             st.write(metrics.loc['smape', :][['Week ahead', 'Last week']])
   


elif country=='Trentino, IT':
    cellid_ = st.sidebar.selectbox(label = "Select cell ID", index = 0,
                                options = ['2738', '5201', '5230'])

    
    st.subheader('Trentino, IT') 
    trentino_dict = read_trentino()
    line_energy, cell_lines, cols_ts = trentino_dict['processed']['line_energy'], \
        trentino_dict['processed']['cell_lines'], \
        trentino_dict['processed']['cols_ts']
    telecom = trentino_dict['original_dataset']['telecom']
    cellid = int(cellid_)

    lineid = cell_lines.query("index == @cellid")['LINESET'].values[0]

    energy_select = line_energy.query("SQUAREID == @cellid and LINESET == @lineid")#.sort_values("LINESET")
    energy_cell = energy_select[cols_ts].transpose()
    energy_cell.columns = energy_select['SQUAREID']#.astype(str)#  + '-' +energy_select['LINESET'].astype(str) 
    energy_cell.index = pd.to_datetime(energy_cell.index)
    energy_cell.index.name = "date"
    energy_cell.sort_index(inplace=True)
    telecom_cell = telecom.query("CellID == @cellid").sort_index(inplace=False)
    
    merged = pd.merge(energy_cell, telecom_cell.reset_index().groupby("datetime").mean(), 
                      left_index=True, right_index=True, how="right")
    merged.rename(columns={cellid: "energy"}, inplace=True)

    print(merged.columns, merged.shape)

    fig = go.Figure()

    trace1 = go.Scatter(x=merged.index, y=merged['energy'], name="energy", mode='lines',
                            yaxis='y1')
    fig.add_trace(trace1)

    for col in ['smsin', 'smsout', 'callin', 'callout', 'internet']:
            trace = go.Scatter(x=merged.index, y=merged[col], name=col, mode='lines',
                                    yaxis='y2')
            fig.add_trace(trace)

    # Add figure title
    fig.update_layout(
        autosize=False,
        width=1000,
        height=500,
        title=f"Cell {cellid}",
        xaxis_title="Date",
        yaxis=dict(title='Energy'),
        yaxis2=dict(title='Activities',
                overlaying='y',
                side='right'),
        # legend_title="Legend Title",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="RebeccaPurple"
        ),
        legend=dict(
            # title="Legend Title",
            x=1.1,
            y=1,
            traceorder='normal',
            font=dict(
                size=10),
        ),
        )

    st.plotly_chart(fig)

    st.markdown("Select a date to inspect")
    _date = st.date_input('Start date', value=merged.index.mean().date(), min_value=merged.index.min(), max_value = merged.index.max())
    print(_date, type(_date))


   

else:
    st.subheader('Select a dataset')

#streamlit run dashboard.py
