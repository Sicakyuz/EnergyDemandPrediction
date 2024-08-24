import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet
import file_upload

# Function to load and preprocess data
def preprocess_data(df):
    print("Columns in DataFrame:", df.columns)  # Print column names
    print("First few rows of the DataFrame:", df.head())  # Print first few rows

    if 'TARIH' in df.columns:
        df['TARIH'] = pd.to_datetime(df['TARIH'], errors='coerce')
        df = df.dropna(subset=['TARIH'])  # Drop rows with invalid dates

        # Ensure date range is valid
        min_date = df['TARIH'].min()
        max_date = df['TARIH'].max()
        if min_date.year < 2018 or max_date.year > 2024:
            raise ValueError("Date range is out of bounds. Ensure dates are within a valid range.")

        df.set_index('TARIH', inplace=True)
    else:
        st.error("Column 'TARIH' is missing from the data.")
        return None

    df.fillna(method='ffill', inplace=True)
    df = pd.get_dummies(df, columns=['BOLGE', 'ILLER'], drop_first=True)

    return df

def prepare_test_data(df, forecast_horizon):
    test_data = df[-forecast_horizon:].copy()
    test_data['TARIH'] = pd.to_datetime(test_data['TARIH'], errors='coerce')
    test_data = test_data.dropna(subset=['TARIH'])
    test_data = test_data.reset_index(drop=True)  # Reset index to ensure alignment

    return test_data

# Function to prepare data for machine learning models
def prepare_ml_data(df):
    df = df.copy()

    # Convert datetime columns to numeric features or handle them as necessary
    if 'TARIH' in df.columns:
        df['TARIH'] = df['TARIH'].astype(np.int64) // 10 ** 9  # Convert to timestamp
        df = df.drop(columns=['TARIH'])

    X = df.drop(columns=['TUKETIM_GENEL_TOPLAM'])
    y = df['TUKETIM_GENEL_TOPLAM']

    X = X.select_dtypes(include=[np.number])  # Keep only numeric columns

    # Ensure that X and y are numpy arrays
    X = X.values
    y = y.values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

# Function to train SARIMA model
def train_sarima_model(df):
    model = SARIMAX(df['TUKETIM_GENEL_TOPLAM'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    results = model.fit()
    return results

# Function to train Prophet model
def train_prophet_model(df):
    df_prophet = df.reset_index()
    df_prophet.columns = ['ds', 'y'] + list(df_prophet.columns[2:])

    model = Prophet()
    model.fit(df_prophet[['ds', 'y']])
    return model

# Function to train ML models
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

# Function to calculate MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    nonzero_elements = y_true != 0
    return np.mean(np.abs((y_true[nonzero_elements] - y_pred[nonzero_elements]) / y_true[nonzero_elements])) * 100

# Function to evaluate models
def evaluate_model(model, X_test=None, y_test=None, model_type=None, scaler=None, df=None):
    if df is None:
        raise ValueError("Dataframe must be provided for model evaluation")

    if model_type in ['Random Forest', 'Gradient Boosting', 'Linear Regression']:
        if X_test is None or y_test is None:
            raise ValueError("X_test and y_test must be provided for ML model evaluation")

        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        mape = mean_absolute_percentage_error(y_test, predictions)


    elif model_type == 'Prophet':
        if df is None:
            raise ValueError("Dataframe must be provided for Prophet model evaluation")

        num_periods = min(len(df), 365)  # Limit to a reasonable forecast period
        future = model.make_future_dataframe(periods=num_periods, freq='M')

        if future.empty:
            raise ValueError("Future dataframe is empty")

        forecast = model.predict(future)
        predictions = forecast['yhat'][-num_periods:].values  # Align predictions with forecast period
        y_test = df['TUKETIM_GENEL_TOPLAM'][-num_periods:].values  # Align y_test with forecast period

        if len(predictions) != len(y_test):
            raise ValueError(f"Inconsistent lengths: y_test ({len(y_test)}), predictions ({len(predictions)})")

        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        mape = mean_absolute_percentage_error(y_test, predictions)


    elif model_type == 'SARIMA':
        if df is None:
            raise ValueError("Dataframe must be provided for SARIMA model evaluation")

        forecast_horizon = min(len(df) - int(len(df) * 0.8), 365)  # Limit to 1 year or less
        start = df.index[-1] + pd.DateOffset(days=1)
        end = start + pd.DateOffset(days=forecast_horizon)

        try:
            forecast = model.get_forecast(steps=forecast_horizon)
            predictions = forecast.predicted_mean
            y_test = df['TUKETIM_GENEL_TOPLAM'][-forecast_horizon:].values
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            mae = mean_absolute_error(y_test, predictions)
            mape = mean_absolute_percentage_error(y_test, predictions)

        except ValueError as e:
            print(f"Error during SARIMA forecasting: {e}")
            return None, None, None

    else:
        raise ValueError("Invalid model type")

    return rmse, mae, mape, predictions

def plot_results(test_data, predictions):
    import matplotlib.pyplot as plt

    if predictions is not None and not test_data.empty:
        plt.figure(figsize=(12, 6))
        plt.plot(test_data.index, test_data['TUKETIM_GENEL_TOPLAM'], label='Actual', color='blue')
        plt.plot(test_data.index[-len(predictions):], predictions, label='Predicted', color='red')
        plt.xlabel('Date')
        plt.ylabel('Energy Consumption')
        plt.title('Energy Consumption Forecast')
        plt.legend()
        plt.show()
    else:
        print("No predictions available to plot.")

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




def plot_results(test_data, predictions):
    if predictions is not None and not test_data.empty:
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot actual and predicted data
        ax.plot(test_data.index, test_data['TUKETIM_GENEL_TOPLAM'], label='Actual', color='royalblue', linestyle='--', linewidth=2)
        ax.plot(test_data.index[-len(predictions):], predictions, label='Predicted', color='tomato', linestyle='-', linewidth=2)

        # Set labels and title
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('Energy Consumption', fontsize=14)
        ax.set_title('Energy Consumption Forecast', fontsize=16)

        # Set y-axis limits and grid
        ax.set_ylim(bottom=0)  # Set bottom limit of y-axis to 0, adjust as necessary
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))  # Use integer ticks

        # Format x-axis to show years and improve readability
        ax.xaxis.set_major_locator(mdates.YearLocator())  # Place a tick mark for each year
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format ticks as year
        plt.xticks(rotation=45)  # Rotate date labels for readability

        # Add legend and grid
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Display the plot in Streamlit
        st.pyplot(fig)
    else:
        st.write("Predictions cannot be displayed.")


def main_TR():
    data = st.session_state.get('data')
    if data is not None:
        df = preprocess_data(data)
        if df is None:
            st.error("Data preprocessing failed.")
            return

        st.write("Data Preview:")
        st.write(df.head())

        model_type = st.selectbox("Select model",
                                  ["SARIMA", "Prophet", "Random Forest", "Gradient Boosting", "Linear Regression"])

        if model_type in ["Random Forest", "Gradient Boosting", "Linear Regression"]:
            X, y, scaler = prepare_ml_data(df)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = train_ml_model(X_train, y_train, model_type)
            rmse, mae, mape, predictions = evaluate_model(model, X_test, y_test, model_type=model_type, scaler=scaler, df=df)
            st.write(f"Model: {model_type}")
            st.write(f"RMSE: {rmse:.2f}")
            st.write(f"MAE: {mae:.2f}")
            st.write(f"MAPE: {mape:.2f}%")

            st.write("Predictions:")
            st.write(predictions)

            plot_results(df[-len(predictions):], predictions)

        elif model_type == 'Prophet':
            model = train_prophet_model(df)
            rmse, mae, mape, predictions = evaluate_model(model, model_type='Prophet', df=df)
            st.write(f"Model: Prophet")
            st.write(f"RMSE: {rmse:.2f}")
            st.write(f"MAE: {mae:.2f}")
            st.write(f"MAPE: {mape:.2f}%")

            st.write("Predictions:")
            st.write(predictions)

            plot_results(df[-len(predictions):], predictions)

        elif model_type == 'SARIMA':
            model = train_sarima_model(df)
            rmse, mae, mape, predictions = evaluate_model(model, model_type='SARIMA', df=df)
        if rmse is not None and mae is not None and mape is not None and predictions is not None:
            plot_results(df, predictions)  # Adjust plot_results if needed
        else:
            st.write("Model evaluation failed.")

            st.write(f"Model: SARIMA")
            st.write(f"RMSE: {rmse:.2f}")
            st.write(f"MAE: {mae:.2f}")
            st.write(f"MAPE: {mape:.2f}%")

            st.write("Predictions:")
            st.write(predictions)
            # Creating a DataFrame for the predictions and actual values
            predictions_df = df.copy()
            forecast_horizon = len(predictions)
            predictions_df = predictions_df[-forecast_horizon:].copy()
            predictions_df['Predicted'] = predictions
            predictions_df = predictions_df.reset_index()
            predictions_df.columns = ['Date', 'Actual', 'Predicted']  # Adjust columns if necessary

            st.write("Predictions vs Actual Values:")
            st.dataframe(predictions_df)

            # Visualization
            fig, ax = plt.subplots()
            ax.plot(df.index, df['TUKETIM_GENEL_TOPLAM'], label='Actual')
            if predictions is not None:
                ax.plot(df.index[-len(predictions):], predictions, label='Predicted', color='red')
            ax.set_xlabel('Date')
            ax.set_ylabel('Energy Consumption')
            ax.set_title('Energy Consumption Forecast')
            ax.legend()

            st.pyplot(fig)
            plot_results(df[-len(predictions):], predictions)

        plot_seasonal_decomposition(df)
