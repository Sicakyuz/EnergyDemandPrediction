import warnings
import streamlit as st
import numpy as np
from datetime import timedelta, datetime
import time
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score, roc_curve, auc
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import train_test_split
import plotly.express as px
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go
import folium
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster
import branca
import branca.colormap as cm
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

# Settings
pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')

################# DATA LOADING #################################

# Path to the default file
DEFAULT_FILE_PATH = '/Users/asmir/Desktop/Enerji/Enerji/Data/Book2.xlsx'

def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
    else:
        # Load the default file if no file is uploaded
        df = pd.read_excel(DEFAULT_FILE_PATH)
    return df


################# DATA PREPROCESSING #################################

def create_date_features(df):
    if 'TARIH' in df.columns:
        df['TARIH'] = pd.to_datetime(df['TARIH'])
        df['YEAR'] = df['TARIH'].dt.year.astype(int)
        df['MONTH'] = df['TARIH'].dt.month
        df['DAY'] = df['TARIH'].dt.day
        df['DAY_OF_MONTH'] = df['TARIH'].dt.day
        df['DAY_OF_YEAR'] = df['TARIH'].dt.dayofyear
        df['WEEK_OF_YEAR'] = df['TARIH'].dt.isocalendar().week
    else:
        st.error('TARIH sütunu eksik veya yanlış formatta.')
    return df

def preprocess_categorical_features(df):
    # Fill NaN values with a placeholder string before encoding
    cat_columns = df.select_dtypes(include=['object', 'category']).columns
    label_encoders = {}
    for col in cat_columns:
        label_encoders[col] = LabelEncoder()
        df[col] = label_encoders[col].fit_transform(df[col])
    return df, label_encoders


def random_noise(dataframe):
    return np.random.normal(scale=0.01, size=(len(dataframe),))

def add_lag_and_noise_features(df, lags, column):
    for lag in lags:
        df[f'{column}_LAG_{lag}'] = df.groupby(['ILLER',"BOLGE"])[column].transform(
            lambda x: x.shift(lag)) + random_noise(df)
    return df

def add_rolling_mean_features(df, window_sizes, column='TUKETIM_GENEL_TOPLAM'):
    for window in window_sizes:
        for lag in range(1, 13):
            df[f'{column}_ROLLING_MEAN_WINDOW_{window}_LAG_{lag}'] = df.groupby(['ILLER',"BOLGE"])[column].shift(lag).rolling(window=window, min_periods=1, win_type="triang").mean()+random_noise(df)
    return df

def add_ewm_features(df, alphas, lags, column='TUKETIM_GENEL_TOPLAM'):
    for alpha in alphas:
        for lag in lags:
            df[f'{column}_EWM_ALPHA_{str(alpha).replace(".", "")}_LAG_{lag}'] = df.groupby(['ILLER',"BOLGE"])[column].shift(
                    lag).ewm(alpha=alpha).mean()
    return df


def add_seasonality_features(df):
    if 'MONTH' not in df.columns:
        st.error('MONTH column not found. Ensure date features are created first.')
    df['SIN_MONTH'] = np.sin(2 * np.pi * df['MONTH'] / 12)
    df['COS_MONTH'] = np.cos(2 * np.pi * df['MONTH'] / 12)
    return df

# Update your main preprocessing function
def preprocess_data(df, lags, window_sizes, alphas):
    print("Columns before preprocessing:", df.columns)
    df = create_date_features(df)
    print("Columns after creating date features:", df.columns)

    if df['TARIH'].isnull().any():
        st.error("Error in date parsing. Check date formats.")
        return None, None

    # Continue with preprocessing
    df, label_encoders = preprocess_categorical_features(df)
    df = add_lag_and_noise_features(df, lags, 'TUKETIM_GENEL_TOPLAM')
    df = add_rolling_mean_features(df, window_sizes)
    df = add_seasonality_features(df)
    df = add_ewm_features(df, alphas, lags, 'TUKETIM_GENEL_TOPLAM')
    df.sort_values(by=['ILLER',"BOLGE", 'TARIH'], inplace=True)

    return df, label_encoders


################# TRAIN-TEST DATA DEFINITION #################################

# Veri setini ayırma

def split_and_check_data(df, split_date):
    # Ensure split_date is a pandas Timestamp
    split_date = pd.Timestamp(split_date)

    train_data = df[df['TARIH'] < split_date]
    val_data = df[df['TARIH'] >= split_date]

    if train_data.empty or val_data.empty:
        st.error("Training or validation set is empty. Please check your split date.")
        return None, None, False

    return train_data, val_data, True

################# MODEL TRAINING ############################################


# In your training function, ensure the categorical features are listed correctly before training
def train_model(train_data, cat_features):
    X_train = train_data.drop(columns=['TARIH', 'TUKETIM_GENEL_TOPLAM'])
    y_train = train_data['TUKETIM_GENEL_TOPLAM']
    lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=cat_features)

    params = {
        'objective': 'regression',
        'metric': 'l1',
        'num_leaves': 60,
        'learning_rate': 0.05,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'seed': 42
    }

    model = lgb.train(params, lgb_train, num_boost_round=1000, valid_sets=[lgb_train], callbacks=[lgb.early_stopping(50)])
    return model

################# MODEL EVALUATION ##########################################

def evaluate_model(model, val_data):
    X_val = val_data.drop(columns=['TARIH', 'TUKETIM_GENEL_TOPLAM'])
    y_val = val_data['TUKETIM_GENEL_TOPLAM']

    y_pred = model.predict(X_val, num_iteration=model.best_iteration)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    st.write("Doğrulama seti için Ortalama Mutlak Hata (MAE): ", mae)
    st.write("Doğrulama seti için Hata Kareler Ortalaması (RMSE): ", rmse)

    return mae, rmse

################# PREDICTION ################################################

def predict_future_data(model, df, label_encoders, lags, window_sizes, alphas, n_months):
    # Generate future dates
    last_date = pd.to_datetime(df['TARIH']).max()
    future_dates = [last_date + timedelta(days=31*i) for i in range(1, n_months+1)]

    future_df = pd.DataFrame({'TARIH': future_dates})

    # Creating lag features
    for lag in lags:
        future_df[f'TUKETIM_GENEL_TOPLAM_LAG_{lag}'] = df['TUKETIM_GENEL_TOPLAM'].shift(lag)

    # Creating rolling mean features
    for window in window_sizes:
        for lag in range(1, 13):
            future_df[f'TUKETIM_GENEL_TOPLAM_ROLLING_MEAN_WINDOW_{window}_LAG_{lag}'] = df['TUKETIM_GENEL_TOPLAM'].shift(lag).rolling(window=window, min_periods=1, win_type="triang").mean()

    # Creating exponential weighted mean features
    for alpha in alphas:
        for lag in lags:
            future_df[f'TUKETIM_GENEL_TOPLAM_EWM_ALPHA_{str(alpha).replace(".", "")}_LAG_{lag}'] = df['TUKETIM_GENEL_TOPLAM'].shift(lag).ewm(alpha=alpha).mean()

    # Creating date features
    future_df = create_date_features(future_df)

    # Label encoding for categorical features
    for col in label_encoders:
        future_df[col] = label_encoders[col].transform(future_df[col])

    # Making predictions
    X_future = future_df.drop(columns=['TARIH'])
    future_df['TUKETIM_GENEL_TOPLAM_PREDICTED'] = model.predict(X_future)

    return future_df

def debug_data(df):
    st.write("Preview:", df.head())
    st.write("Data Types:", df.dtypes)
    st.write("Null Values:", df.isnull().sum())
    st.write("Describe Data:", df.describe(include='all'))

#


################# MAIN FUNCTION ##############################################

def main():
    st.title("Elektrik Tüketimi Tahmini")

    st.sidebar.title("Parametreler")

    # Data upload
    uploaded_file = st.sidebar.file_uploader("Lütfen dosya yükleyin.", type=["xlsx", "csv"])
    if uploaded_file is not None:
        df=pd.read_excel(uploaded_file)
        df = load_data(uploaded_file)
               # Feature engineering parameters
        st.sidebar.subheader("Öznitelik Mühendisliği Parametreleri")

        # Setting split_date using Streamlit's date_input
        split_date = st.sidebar.date_input("Select a split date",
                                           value=pd.to_datetime('today') - pd.DateOffset(days=30))
        # Öznitelik mühendisliği parametrelerini kullanıcıya bırak
        lags = st.sidebar.multiselect("Lag sayıları", options=[1, 7, 30, 60, 90], default=[1, 7, 30])
        window_sizes = st.sidebar.multiselect("Hareketli ortalama pencere boyutları", options=[3, 7, 15, 30, 60],
                                              default=[3, 7, 15])
        alphas = st.sidebar.multiselect("EWMA alpha değerleri", options=[0.1, 0.3, 0.5, 0.7, 0.9],
                                        default=[0.3, 0.5, 0.7])
        # Data Preprocessing
        df_preprocessed, label_encoders = preprocess_data(df, [1, 7, 30], [7, 15, 30], [0.3, 0.5])
        split_date = pd.Timestamp(split_date)  # Convert to Timestamp here if not yet converted

        st.write(df_preprocessed.head())

        debug_data(df_preprocessed)
        if df_preprocessed is not None:
            st.write("Preprocessed Data:")
            st.write(df_preprocessed.head())
        # Split data


        # Tarihe göre veri setini ayırma
        train_data = df[df['TARIH'] < split_date]
        val_data = df[df['TARIH'] >= split_date]

        print("Eğitim seti boyutu:", train_data.shape)
        print("Doğrulama seti boyutu:", val_data.shape)

        if train_data.empty or val_data.empty:
            print("Eğitim seti veya doğrulama seti boş. Lütfen split tarihini kontrol edin.")
        else:
            print("Her iki set de uygun veri içeriyor.")
        # Train model
        st.sidebar.subheader("Model Eğitimi")
        if st.sidebar.button("Train Model"):
            train_data, val_data, success = split_and_check_data(df_preprocessed, split_date)

            if success:
                st.sidebar.subheader("Model Training")
                cat_features = ['ILLER', 'BOLGE']  # Assuming these are categorical features you've encoded

                model = train_model(train_data, cat_features)
                mae, rmse = evaluate_model(model, val_data)
                st.write(f"Validation MAE: {mae}, RMSE: {rmse}")

            # Predict future data
            st.sidebar.subheader("Gelecek Veri Tahmini")
            # Future predictions
            n_months = st.sidebar.number_input("Months ahead to predict", min_value=1, max_value=12, value=3)
            if st.sidebar.button("Predict Future"):
                future_data = predict_future_data(model, df_preprocessed, label_encoders, lags, window_sizes,
                                                      alphas, n_months)
                st.write("Future Predictions:")
                st.dataframe(future_data[['TARIH', 'TUKETIM_GENEL_TOPLAM_PREDICTED']])
                fig = px.line(future_data, x='TARIH', y='TUKETIM_GENEL_TOPLAM_PREDICTED',
                                  title="Future Consumption Predictions")
                st.plotly_chart(fig)
        else:
            st.error("Failed to create training and validation sets. Please check your data and parameters.")

    else:
        st.info("Lütfen yüklemek için bir Excel dosyası yükleyin.")


if __name__ == '__main__':
    main()