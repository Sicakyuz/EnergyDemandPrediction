import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet
import plotly.graph_objects as go

# Function to load and preprocess data
def preprocess_data(df):
    if 'TARIH' not in df.columns or 'TUKETIM_GENEL_TOPLAM' not in df.columns:
        st.error("Columns 'TARIH' and/or 'TUKETIM_GENEL_TOPLAM' are missing from the data.")
        st.write("Data columns:", df.columns)
        return None

    df['TARIH'] = pd.to_datetime(df['TARIH'], errors='coerce')
    df = df.dropna(subset=['TARIH'])

    min_date = df['TARIH'].min()
    max_date = df['TARIH'].max()
    if min_date.year < 2018 or max_date.year > 2023:
        st.error("Date range is out of bounds. Ensure dates are within a valid range.")
        return None

    df['YEAR'] = df['TARIH'].dt.year
    df['MONTH'] = df['TARIH'].dt.month
    df['YEARMOMTH'] = df['TARIH'].dt.to_period('M')
    df.set_index('TARIH', inplace=True)

    df_monthly = df.groupby(['YEAR', 'MONTH']).agg({'TUKETIM_GENEL_TOPLAM': 'sum'}).reset_index()
    df_monthly['TARIH'] = pd.to_datetime(df_monthly[['YEAR', 'MONTH']].assign(DAY=1))
    df_monthly = df_monthly.set_index('TARIH').asfreq('MS').fillna(method='ffill')

    return df_monthly
def prepare_test_data(df, forecast_horizon):
    test_data = df[-forecast_horizon:].copy()
    test_data.reset_index(drop=True, inplace=True)
    return test_data

def prepare_ml_data(df):
    df = df.copy()
    df['TARIH'] = df.index.astype(np.int64) // 10 ** 9
    df = df.drop(columns=['TARIH'])

    X = df.drop(columns=['TUKETIM_GENEL_TOPLAM'])
    y = df['TUKETIM_GENEL_TOPLAM']
    X = X.select_dtypes(include=[np.number])
    X = X.values
    y = y.values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

def train_sarima_model(df):
    model = SARIMAX(df['TUKETIM_GENEL_TOPLAM'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    results = model.fit()
    return results

def train_ml_model(X_train, y_train, model_type):
    if model_type == 'Random Forest':
        model = RandomForestRegressor()
    elif model_type == 'Gradient Boosting':
        model = GradientBoostingRegressor()
    elif model_type == 'Linear Regression':
        model = LinearRegression()
    else:
        raise ValueError("Invalid model type")

    model.fit(X_train, y_train)
    return model

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    nonzero_elements = y_true != 0
    return np.mean(np.abs((y_true[nonzero_elements] - y_pred[nonzero_elements]) / y_true[nonzero_elements])) * 100

def evaluate_model(model, X_test=None, y_test=None, model_type=None, scaler=None, df=None, forecast_horizon=30):
    if df is None:
        raise ValueError("Dataframe must be provided for model evaluation")

    if model_type in ['Random Forest', 'Gradient Boosting', 'Linear Regression']:
        if X_test is None or y_test is None:
            raise ValueError("X_test and y_test must be provided for ML model evaluation")

        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        mape = mean_absolute_percentage_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)


    elif model_type == 'Prophet':
        df_prophet = df.reset_index()
        df_prophet.columns = ['ds', 'y']  # Make sure this matches your DataFrame structure
        future = model.make_future_dataframe(periods=forecast_horizon, freq='M')
        forecast = model.predict(future)
        predictions = forecast['yhat'][-forecast_horizon:].values
        y_test = df_prophet['y'][-forecast_horizon:].values

        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        mape = mean_absolute_percentage_error(y_test, predictions)

    elif model_type == 'SARIMA':
        start = df.index[-1] + pd.DateOffset(months=1)
        forecast = model.get_forecast(steps=forecast_horizon)
        predictions = forecast.predicted_mean
        y_test = df['TUKETIM_GENEL_TOPLAM'][-forecast_horizon:].values

        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        mape = mean_absolute_percentage_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)


    else:
        raise ValueError("Invalid model type")

    return rmse, mae, mape,mse, predictions

def plot_seasonal_decomposition(df):
    result = seasonal_decompose(df['TUKETIM_GENEL_TOPLAM'], model='additive', period=12)
    fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    axs[0].plot(result.observed, label='Observed', color='blue')
    axs[0].set_title('Observed')
    axs[1].plot(result.trend, label='Trend', color='orange')
    axs[1].set_title('Trend')
    axs[2].plot(result.seasonal, label='Seasonal', color='green')
    axs[2].set_title('Seasonal')
    axs[3].plot(result.resid, label='Residual', color='red')
    axs[3].set_title('Residual')

    for ax in axs:
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    st.pyplot(fig)

def plot_results(df, predictions):
    if predictions is not None and not df.empty:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(df.index, df['TUKETIM_GENEL_TOPLAM'], label='Actual', color='royalblue', linestyle='--', linewidth=2)
        ax.plot(df.index[-len(predictions):], predictions, label='Predicted', color='tomato', linestyle='-',
                linewidth=2)
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('Energy Consumption', fontsize=14)
        ax.set_title('Energy Consumption Forecast', fontsize=16)
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
    else:
        st.write("No predictions available to plot.")

def plot_results_with_sub(df, predictions):
    fig, axs = plt.subplots(2, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [3, 1]})

    axs[0].plot(df.index, df['TUKETIM_GENEL_TOPLAM'], label='Actual', color='royalblue', linestyle='--', linewidth=2)
    axs[0].plot(df.index[-len(predictions):], predictions, label='Predicted', color='tomato', linestyle='-',
                linewidth=2)
    axs[0].set_xlabel('Date', fontsize=14)
    axs[0].set_ylabel('Energy Consumption', fontsize=14)
    axs[0].set_title('Energy Consumption Forecast', fontsize=16)
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(df.index[-len(predictions):], df['TUKETIM_GENEL_TOPLAM'][-len(predictions):], label='Actual',
                color='royalblue', linestyle='--', linewidth=2)
    axs[1].plot(df.index[-len(predictions):], predictions, label='Predicted', color='tomato', linestyle='-',
                linewidth=2)
    axs[1].set_xlabel('Date', fontsize=14)
    axs[1].set_ylabel('Energy Consumption', fontsize=14)
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    st.pyplot(fig)


def prepare_prophet_data(df):
    # Prepare dataframe for Prophet
    df_prophet = df[['TUKETIM_GENEL_TOPLAM']].reset_index()
    df_prophet.columns = ['ds', 'y']  # Rename columns to 'ds' and 'y'
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])  # Ensure 'ds' column is datetime

    # Print dataframe columns and types for debugging
    print("Prepared Prophet Data:")
    print(df_prophet.head())
    print(df_prophet.columns)
    print(df_prophet.dtypes)

    # Check if the dataframe contains required columns
    if 'ds' not in df_prophet.columns or 'y' not in df_prophet.columns:
        raise ValueError("Dataframe must have columns 'ds' and 'y'.")

    return df_prophet


def train_prophet_model(df_prophet):
    model = Prophet(yearly_seasonality=True)
    model.fit(df_prophet)
    return model


def forecast_prophet(model, df_prophet, periods=30):
    future = model.make_future_dataframe(periods=periods, freq='M')
    forecast = model.predict(future)
    return forecast


def evaluate_prophet_forecast(df_prophet, forecast):
    df_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].set_index('ds')
    df_real = df_prophet.set_index('ds')

    df_combined = df_forecast.join(df_real, how='left')
    df_combined = df_combined.dropna()

    y_true = df_combined['y']
    y_pred = df_combined['yhat']

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    metrics = {
        'Mean Absolute Error (MAE)': mae,
        'Mean Squared Error (MSE)': mse,
        'Root Mean Squared Error (RMSE)': rmse,
        'Mean Absolute Percentage Error (MAPE)': mape
    }

    return metrics



def show_forecast_table(forecast):
    # 'ds', 'yhat', 'yhat_lower', 'yhat_upper' sütunlarıyla bir DataFrame oluştur
    forecast_table = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()

    # Tarihleri daha okunabilir bir formata dönüştürme
    forecast_table['ds'] = forecast_table['ds'].dt.strftime('%Y-%m-%d')

    # Tabloyu Streamlit'de gösterme
    st.write("### Forecast Table", forecast_table.style.format({"yhat": "{:,.2f}", "yhat_lower": "{:,.2f}", "yhat_upper": "{:,.2f}"}))




def plot_prophet_forecast(df_prophet, forecast):
    # Plotly ile tahmin grafiğini oluşturma
    fig = go.Figure()

    # Gerçek değerler çizgisi (Mavi)
    fig.add_trace(go.Scatter(
        x=df_prophet['ds'],
        y=df_prophet['y'],
        mode='lines',
        name='Actual Values',
        line=dict(color='blue', width=2)  # Mavi çizgi
    ))

    # Tahmin çizgisi (Kırmızı)
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        mode='lines',
        name='Forecasted Value',
        line=dict(color='red', width=2)  # Kırmızı çizgi
    ))

    # Tahmin aralığı (Şeffaf gri gölge)
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_upper'],
        mode='lines',
        name='Upper Bound',
        line=dict(color='rgba(255,0,0,0)'),
        fill='tonexty',
        fillcolor='rgba(255,0,0,0.2)'
    ))

    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_lower'],
        mode='lines',
        name='Lower Bound',
        line=dict(color='rgba(255,0,0,0)'),
        fill='tonexty',
        fillcolor='rgba(255,0,0,0.2)'
    ))

    # Grafiği özelleştirme
    fig.update_layout(
        title='Prophet Forecast',
        xaxis_title='Date',
        yaxis_title='Value',
        legend=dict(x=0, y=1.0),
        template='plotly_white'
    )

    # Grafiği gösterme
    st.plotly_chart(fig)



def display_metrics(mae, mse, rmse, mape):
    # Sayısal değerlerin string olarak formatlanması
    mae_str = "{:,.2f}".format(mae)
    mse_str = "{:,.2f}".format(mse)
    rmse_str = "{:,.2f}".format(rmse)
    mape_str = "{:.2f}".format(mape)

    # Streamlit'te biçimlendirilmiş metni göstermek
    st.markdown(f"""
    <div style="font-size: 20px; font-weight: bold; color: #0044cc; background-color: #e6f0ff; padding: 10px; border-radius: 5px; margin-bottom: 10px; font-family: sans-serif;">
        Mean Absolute Error (MAE): <span style="color: #0033a0;">{mae_str}</span>
    </div>
    <div style="font-size: 20px; font-weight: bold; color: #0044cc; background-color: #e6f0ff; padding: 10px; border-radius: 5px; margin-bottom: 10px; font-family: sans-serif;">
        Mean Squared Error (MSE): <span style="color: #0033a0;">{mse_str}</span>
    </div>
    <div style="font-size: 20px; font-weight: bold; color: #0044cc; background-color: #e6f0ff; padding: 10px; border-radius: 5px; margin-bottom: 10px; font-family: sans-serif;">
        Root Mean Squared Error (RMSE): <span style="color: #0033a0;">{rmse_str}</span>
    </div>
    <div style="font-size: 20px; font-weight: bold; color: #0044cc; background-color: #e6f0ff; padding: 10px; border-radius: 5px; margin-bottom: 10px; font-family: sans-serif;">
        Mean Absolute Percentage Error (MAPE): <span style="color: #0033a0;">{mape_str}%</span>
    </div>
    """, unsafe_allow_html=True)


def main_TR():
    # Streamlit application code
    st.title('Energy Consumption Forecasting')

    df = st.session_state['data']
    if df is not None:
        df_preprocessed = preprocess_data(df)

    if df_preprocessed is not None:
        st.write("Data Preprocessed Successfully")
        st.dataframe(df_preprocessed.head())

        # Model Selection
        model_type = st.selectbox("Select the model to train",
                                  ["SARIMA", "Prophet", "Random Forest", "Gradient Boosting", "Linear Regression"])
        forecast_horizon = st.slider("Select forecast horizon", min_value=1, max_value=36, value=12, step=1)

        if model_type in ["Random Forest", "Gradient Boosting", "Linear Regression"]:
            X, y, scaler = prepare_ml_data(df_preprocessed)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = train_ml_model(X_train, y_train, model_type)
            st.write(f"Training {model_type} Model...")
            rmse, mae, mape,mse, predictions = evaluate_model(model, X_test, y_test, model_type=model_type, scaler=scaler,
                                                          df=df_preprocessed, forecast_horizon=forecast_horizon)
            st.success(f"{model_type} Trained Successfully")
            display_metrics(mae, mse, rmse, mape)
            plot_results(df_preprocessed, predictions)

        elif model_type == 'SARIMA':
            model = train_sarima_model(df_preprocessed)
            st.write("Training SARIMA Model...")
            rmse, mae, mape,mse, predictions = evaluate_model(model, model_type='SARIMA', df=df_preprocessed,
                                                          forecast_horizon=forecast_horizon)
            st.success("SARIMA Model Trained Successfully")
            metrics=display_metrics(mae, mse, rmse, mape)

            plot_results(df_preprocessed, predictions)
        elif model_type == 'Prophet':
                st.write("Training Prophet Model...")
                try:
                    df_prophet = prepare_prophet_data(df_preprocessed)

                    # Print the prepared Prophet data for debugging
                    st.success("Prepared Prophet Data:")
                    st.dataframe(df_prophet.head())

                    st.write("Preparing Prophet model...")
                    model = train_prophet_model(df_prophet)
                    st.success("Prophet Model Trained Successfully")

                    # Slider with unique key

                    forecast = forecast_prophet(model, df_prophet, periods=forecast_horizon)

                    metrics = evaluate_prophet_forecast(df_prophet, forecast)
                    display_metrics(metrics['Mean Absolute Error (MAE)'],
                                    metrics['Mean Squared Error (MSE)'],
                                    metrics['Root Mean Squared Error (RMSE)'],
                                    metrics['Mean Absolute Percentage Error (MAPE)'])
                    show_forecast_table(forecast)
                    st.success("Forecast Generated Successfully")

                    st.write("Plotting forecast...")
                    plot_prophet_forecast(df_prophet, forecast)


                except Exception as e:
                    st.error(f"An error occurred: {e}")

        # Seasonal decomposition plot
        st.subheader("Seasonal Decomposition")
        plot_seasonal_decomposition(df_preprocessed)
