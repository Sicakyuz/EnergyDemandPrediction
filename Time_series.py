import streamlit as st
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from file_upload import upload_file,load_default_file

# Load data from a file embedded in the app

#DEFAULT_FILE_PATH = '/Users/asmir/Desktop/MyProjects/EnergyDemandPrediction/Book2.xlsx'

def prepare_data(df):
    df['TARIH'] = pd.to_datetime(df['TARIH'], errors='coerce')
    df.dropna(subset=['TARIH'], inplace=True)
    df.sort_values(by=['ILLER', 'TARIH'], inplace=True)
    df['YEARMONTH'] = df['TARIH'].dt.to_period('M')
    df_grouped = df.groupby(['ILLER', 'YEARMONTH']).agg({'TUKETIM_GENEL_TOPLAM': 'sum'}).reset_index()
    df_grouped['TARIH'] = df_grouped['YEARMONTH'].dt.to_timestamp()
    df_grouped.set_index('TARIH', inplace=True)
    return df_grouped


# Build and forecast using the SARIMA model
def sarima_model_and_forecast(df, city, order, seasonal_order, steps):
    city_data = df[df['ILLER'] == city]
    model = SARIMAX(city_data['TUKETIM_GENEL_TOPLAM'], order=order, seasonal_order=seasonal_order)
    results = model.fit(disp=False)

    # Actual values
    actuals = city_data['TUKETIM_GENEL_TOPLAM']

    # Forecasts
    forecast = results.get_forecast(steps=steps)
    forecast_df = forecast.conf_int()
    forecast_df['Forecast'] = forecast.predicted_mean
    forecast_df = forecast_df[['Forecast', 'lower TUKETIM_GENEL_TOPLAM', 'upper TUKETIM_GENEL_TOPLAM']]

    # Combine forecasts with actual values
    combined_df = pd.concat([actuals, forecast_df], axis=1)
    return combined_df, results


# Main app
def main_City():
    df = st.session_state['data']
    if df is not None:
        df_processed = prepare_data(df)
        st.header("Select City for Forecasting")
        city = st.selectbox('Choose a city', df_processed['ILLER'].unique(),
                            help="Select the city for which the forecast will be made. The model will be trained and forecasts will be generated for this city.")
        st.header("Parameter Selection")

        # Model parameters
        p = st.number_input(
            'AR degree (p)', min_value=0, max_value=5, value=1,
            help="The autoregressive (AR) degree specifies how far back the model will look for dependencies in past values. For example, if p=2, the model will consider the last two observations when making forecasts."
        )
        d = st.number_input(
            'Differencing degree (d)', min_value=0, max_value=2, value=1,
            help="The differencing degree specifies how many times differencing will be applied to the time series to make it stationary. d=1 typically means the series will be differenced once to try to make it stationary."
        )
        q = st.number_input(
            'MA degree (q)', min_value=0, max_value=5, value=1,
            help="The moving average (MA) degree specifies how far back the model will look for dependencies in forecast errors. If q=2, the model will consider the last two forecast errors."
        )
        st.header("Seasonal Degrees")
        P = st.slider(
            'Seasonal AR degree (P)', 0, 5, 1,
            help="The seasonal AR degree specifies how far back the model will look for dependencies in seasonal past values."
        )
        D = st.slider(
            'Seasonal differencing degree (D)', 0, 2, 1,
            help="The seasonal differencing degree specifies the number of times seasonal differencing will be applied to the series to achieve seasonal stationarity."
        )
        Q = st.slider(
            'Seasonal MA degree (Q)', 0, 5, 1,
            help="The seasonal MA degree specifies how far back the model will look for dependencies in seasonal forecast errors."
        )
        s = st.slider(
            'Seasonal period (s)', 1, 12, 12,
            help="The seasonal period specifies the frequency of the data. For example, s=12 is typically used for monthly data."
        )
        steps = st.number_input(
            'Number of forecast steps', 1, 24, 12,
            help="The number of forecast steps specifies how many steps ahead the model will forecast."
        )

        if st.button('Train Model and Forecast'):
            combined_df, results = sarima_model_and_forecast(df_processed, city, (p, d, q), (P, D, Q, s), steps)

            # Store results in session state
            st.session_state['combined_df'] = combined_df
            st.session_state['results'] = results
            st.session_state['city'] = city
            st.session_state['steps'] = steps

            # Visualize actual and forecasted values
            fig = px.line(combined_df, x=combined_df.index, y='TUKETIM_GENEL_TOPLAM',
                          labels={'value': 'Consumption', 'index': 'Date'},
                          title=f'Energy Consumption Forecasts for {city}')
            fig.update_layout(height=500, width=900, title_font_size=24,
                              font=dict(family="Arial, sans-serif"))

            # Add forecasted values to the plot
            fig.add_trace(go.Scatter(x=combined_df.index, y=combined_df['Forecast'], mode='lines', name='Forecast',
                                     line=dict(color='red', width=2)))
            fig.add_trace(go.Scatter(x=combined_df.index, y=combined_df['lower TUKETIM_GENEL_TOPLAM'], mode='lines',
                                     name='Lower Confidence Interval', fill='tonexty', line=dict(color='lightblue')))
            fig.add_trace(go.Scatter(x=combined_df.index, y=combined_df['upper TUKETIM_GENEL_TOPLAM'], mode='lines',
                                     name='Upper Confidence Interval', fill='tonexty', line=dict(color='lightblue')))
            st.plotly_chart(fig, use_container_width=True)

            # Display forecast table
            st.subheader('Forecast Table')
            st.write(combined_df.tail(steps))  # Only show the forecast steps

        # Check if model results are in session state and display the summary
        if 'results' in st.session_state:
            show_summary = st.checkbox("Show Model Summary")
            if show_summary:
                st.subheader('Model Summary')
                st.text(st.session_state['results'].summary().as_text())

            # Decompose the series and display decomposition plot
            result = seasonal_decompose(
                df_processed[df_processed['ILLER'] == st.session_state['city']]['TUKETIM_GENEL_TOPLAM'],
                model='additive', period=12)
            fig_decomposition = make_subplots(rows=3, cols=1, subplot_titles=("Trend", "Seasonal", "Residual"))
            fig_decomposition.add_trace(
                go.Scatter(x=result.trend.index, y=result.trend, mode='lines', name='Trend', line=dict(color='red')),
                row=1, col=1)
            fig_decomposition.add_trace(
                go.Scatter(x=result.seasonal.index, y=result.seasonal, mode='lines', name='Seasonal',
                           line=dict(color='green')), row=2, col=1)
            fig_decomposition.add_trace(go.Scatter(x=result.resid.index, y=result.resid, mode='lines', name='Residual',
                                                   line=dict(color='blue')), row=3, col=1)
            fig_decomposition.update_layout(title=f"Decomposition Plot for {st.session_state['city']}", height=1000,
                                            width=900,
                                            title_font_size=24,
                                            font=dict(family="Arial, sans-serif"))
            st.plotly_chart(fig_decomposition, use_container_width=True)

    else:
        st.error("Data is not loaded. Please upload data through the main interface.")


