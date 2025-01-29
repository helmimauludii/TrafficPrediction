import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error
from sklearn.inspection import permutation_importance
from pmdarima import auto_arima
import warnings
import time
import itertools

# Set page configuration
st.set_page_config(layout="wide")

# Title
st.title("Cellular Network Traffic Prediction System")

# Sidebar: Choose whether to upload your own data or not
upload_choice = st.sidebar.radio(
    "Do you want to upload your own data?",
    ("No", "Yes")
)

data = None  # Initialize the data variable

# Function for preprocessing the data
def preprocess_data(data):
    # List of columns to exclude from conversion
    excluded_columns = ['4G Avg UL Interference', 'Integrity', '4G RSSI (Cells)', 'Date', 'Time', 'eNodeB Name', 'Cell Name']

    # Convert all columns except the excluded ones to numeric
    for col in data.columns:
        if col not in excluded_columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

    # Convert '4G RSSI (Cells)' to object type
    if '4G RSSI (Cells)' in data.columns:
        data['4G RSSI (Cells)'] = data['4G RSSI (Cells)'].astype(object)

    # Merge 'Date' and 'Time' columns into a 'Datetime' column
    if 'Date' in data.columns and 'Time' in data.columns:
        data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%m/%d/%Y %H:%M', errors='coerce')
        data = data.set_index('Datetime')

    return data

# Condition if the user chooses "No" (use data from GitHub)
if upload_choice == "No":
    github_url = "https://raw.githubusercontent.com/helmimauludii/TrafficPredictionHelmi/main/DataTrafikHourlyDesember-Mei.csv"
    try:
        # Read the data from GitHub
        data = pd.read_csv(github_url)

    except Exception as e:
        st.error(f"Failed to load data from GitHub: {e}")

# Condition if the user chooses "Yes" (upload their own data)
elif upload_choice == "Yes":
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            data = pd.read_csv(uploaded_file)

        except Exception as e:
            st.error(f"Failed to read the uploaded CSV file: {e}")
    else:
        st.write("Please upload your CSV file.")

# Apply preprocessing if data is loaded
if data is not None:
    try:
        data = preprocess_data(data)
    except Exception as e:
        st.error(f"Error during preprocessing: {e}")

    # Sidebar: User input for date range
    date_option = st.sidebar.radio("Pilih rentang tanggal:", ('All Date', 'Custom Date'))
    
    if date_option == 'Custom Date':
        # Input tanggal mulai dan berakhir
        start_date = st.sidebar.date_input("Tanggal mulai", value=pd.to_datetime('2024-02-02'))
        end_date = st.sidebar.date_input("Tanggal akhir", value=pd.to_datetime('2024-04-02'))
        
        # Mengonversi ke string dengan format yang diinginkan untuk filter data
        start_date = start_date.strftime('%Y-%m-%d')
        end_date = end_date.strftime('%Y-%m-%d')

        # Preprocessing: Filter data berdasarkan rentang waktu yang dipilih
        data = data.loc[start_date:end_date]
    else:
        # Preprocessing: Filter data berdasarkan rentang waktu default
        start_date = '2024-02-02 00:00:00'
        end_date = '2024-04-02 00:00:00'
        data = data.loc[start_date:end_date]

    # Sidebar menu
    menu = st.sidebar.radio("Menu", ["Traffic Prediction", "Data Visualization"])

    if menu == "Traffic Prediction":
        # Sidebar: Prediction Configuration
        if 'Cell Name' in data.columns:
            unique_cell_names = data['Cell Name'].unique()
            selected_cell = st.sidebar.selectbox("Select Cell Name", unique_cell_names)
            filtered_data = data[data['Cell Name'] == selected_cell]
        else:
            st.warning("Column 'Cell Name' not found in the dataset.")
            filtered_data = data
        
        # Display Filtered Data
        st.write(f"### Filtered Data for Selected Cell")

        # Show table with all data but initial display is limited to a scrollable view
        st.dataframe(filtered_data, height=200)  # Adjust the height to limit visible rows
        
        # Display number of rows and columns
        num_rows, num_cols = filtered_data.shape
        st.caption(f"Jumlah baris: {num_rows}, Jumlah kolom: {num_cols}")
        
        # Step 5: Preprocessing: Hanya ambil kolom numerik
        filtered_data = filtered_data.select_dtypes(include=[np.number])
        filtered_data.fillna(filtered_data.mean(), inplace=True)

        target_column = st.sidebar.selectbox("Field to predict", filtered_data.select_dtypes(include=[np.number]).columns)

        # Jumlah lag
        num_lags = 3

        # Menambahkan lag features berdasarkan input pengguna
        for lag in range(1, num_lags + 1):
            filtered_data[f"{target_column}_lag{lag}"] = filtered_data[target_column].shift(lag)
        
        # Tambahkan informasi waktu sebagai fitur tambahan
        filtered_data['Hour'] = filtered_data.index.hour
        filtered_data['Day'] = filtered_data.index.day
        filtered_data['Month'] = filtered_data.index.month
        
        # Sidebar: Choose Prediction Type
        prediction_type = st.sidebar.selectbox("Choose Prediction Type", ["Deep Learning", "Machine Learning", "Statistic", "Hybrid"])

        if prediction_type == "Deep Learning":
             # Deep Learning Prediction Configuration
            algorithm = st.sidebar.selectbox("Choose Model", ["LSTM", "GRU"])

            feature_columns = [col for col in filtered_data.columns if col != target_column]
            
            # Pilihan parameter default atau custom
            parameter_mode = st.sidebar.radio("Parameter Mode", ["Default", "Custom"])

            if parameter_mode == "Custom":
                num_units = st.sidebar.number_input("Jumlah Unit", min_value=4, max_value=256, value=128, step=4)
                batch_size = st.sidebar.number_input("Batch Size", min_value=4, max_value=256, value=128, step=4)
                max_epochs = st.sidebar.number_input("Epochs Max", min_value=1, max_value=500, value=100, step=1)
                patience = st.sidebar.slider("Patience (Epoch)", min_value=5, max_value=50, value=10, step=5)
                num_layers = st.sidebar.radio("Jumlah Layer", [1, 2, 3], index=1)
            else:
                # Default Parameters
                num_units = 128
                batch_size = 16
                max_epochs = 100
                patience = 50
                num_layers = 1

            # Training/Test Split
            test_split = st.sidebar.slider("Split for test/training", 0.1, 0.9, 0.3)

            # Menghapus nilai NaN yang dihasilkan oleh lag
            filtered_data = filtered_data.dropna()

            # Input untuk jumlah langkah prediksi masa depan
            future_steps = st.sidebar.number_input(
                "Number of Future Steps to Predict", 
                min_value=1, 
                max_value=720, 
                value=24, 
                step=1
            )

            if st.sidebar.button("Start Predict"):
                # Catat waktu mulai
                start_time = time.time()
                progress = st.progress(0)
                with st.spinner(f"Starting Deep Learning Prediction with {algorithm}..."):
                    # Update progress to 20%
                    progress.progress(20)

                    # Set predictor and target columns
                    X = filtered_data[feature_columns]
                    y = filtered_data[target_column]

                    progress.progress(40)  # Update progress to 60%

                    # LSTM Algorithm
                    if algorithm == "LSTM":
                        # Normalisasi data menggunakan StandardScaler
                        scaler_X = MinMaxScaler(feature_range=(0, 1))
                        scaler_y = MinMaxScaler(feature_range=(0, 1))
 
                        X = scaler_X.fit_transform(X)
                        y = scaler_y.fit_transform(y.values.reshape(-1, 1))

                        # Membagi data menjadi train dan test set tanpa shuffle
                        train_size = int(len(X) * (1 - test_split))

                        X_train, X_test = X[:train_size], X[train_size:]
                        y_train, y_test = y[:train_size], y[train_size:]

                        # Reshape X agar sesuai dengan input yang diperlukan LSTM [samples, timesteps, features]
                        X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
                        X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

                        # Membangun model LSTM
                        model = Sequential()

                        if num_layers == 1:
                            # Jika hanya ada 1 layer, return_sequences harus False
                            model.add(LSTM(units=num_units, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))
                            model.add(Dropout(0.2))  # Menambahkan dropout dengan rate 20%
                        elif num_layers == 2:
                            # Jika ada 2 layer, layer pertama memiliki return_sequences=True
                            model.add(LSTM(units=num_units, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
                            model.add(Dropout(0.2))  # Menambahkan dropout setelah layer pertama
                            model.add(LSTM(units=num_units, return_sequences=False))
                            model.add(Dropout(0.2))  # Menambahkan dropout setelah layer kedua
                        elif num_layers == 3:
                            # Jika ada 3 layer, dua layer pertama memiliki return_sequences=True
                            model.add(LSTM(units=num_units, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
                            model.add(Dropout(0.2))  # Menambahkan dropout setelah layer pertama
                            model.add(LSTM(units=num_units, return_sequences=True))
                            model.add(Dropout(0.2))  # Menambahkan dropout setelah layer kedua
                            model.add(LSTM(units=num_units, return_sequences=False))
                            model.add(Dropout(0.2))  # Menambahkan dropout setelah layer ketiga

                        # Output layer
                        model.add(Dense(1))

                        # Kompilasi model
                        model.compile(optimizer='adam', loss='mean_squared_error')

                        progress.progress(60)  # Update progress to 100%

                        # Menggunakan EarlyStopping untuk mencegah overfitting
                        early_stop = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

                        # Melatih model
                        history = model.fit(
                            X_train, y_train, 
                            epochs=max_epochs, 
                            batch_size=batch_size, 
                            validation_split=0.2, 
                            callbacks=[early_stop], 
                            verbose=1
                        )

                        progress.progress(80)  # Update progress to 100%

                        # Prediksi ke masa depan
                        last_known_values = X_test[-1].reshape(1, 1, -1)  # Pastikan input memiliki dimensi [1, 1, features]
                        predictions = []

                        for _ in range(future_steps):
                            # Prediksi nilai baru
                            prediction = model.predict(last_known_values, verbose=0)[0][0]
                            predictions.append(prediction)

                            # Update input dengan nilai prediksi untuk iterasi berikutnya
                            new_input = np.append(last_known_values[0, 0, 1:], prediction).reshape(1, 1, -1)
                            last_known_values = new_input

                        # Membalikkan normalisasi pada prediksi
                        predictions = scaler_y.inverse_transform(np.array(predictions).reshape(-1, 1))
 
                        # Membuat DataFrame untuk prediksi
                        future_dates = pd.date_range(start=data.index[-1], periods=future_steps + 1, freq='H')[1:]
                        future_df = pd.DataFrame({
                            'Datetime': future_dates,
                            'Predicted 4G Total Traffic (GB)': predictions.flatten()
})

                        # Prediksi menggunakan model
                        y_pred_train = model.predict(X_train)
                        y_pred_test = model.predict(X_test)

                        # Membalikkan normalisasi pada prediksi dan data sebenarnya
                        y_train = scaler_y.inverse_transform(y_train.reshape(-1, 1))
                        y_test = scaler_y.inverse_transform(y_test.reshape(-1, 1))
                        y_pred_train = scaler_y.inverse_transform(y_pred_train.reshape(-1, 1))
                        y_pred_test = scaler_y.inverse_transform(y_pred_test.reshape(-1, 1))
                    
                        progress.progress(100)  # Update progress to 100%

                    # GRU Algorithm
                    if algorithm == "GRU":
                        # Normalisasi data menggunakan StandardScaler
                        scaler_X = StandardScaler()
                        scaler_y = StandardScaler()
                        
                        X = scaler_X.fit_transform(X)
                        y = scaler_y.fit_transform(y.values.reshape(-1, 1))

                        # Membagi data menjadi train dan test set tanpa shuffle
                        train_size = int(len(X) * (1 - test_split))

                        X_train, X_test = X[:train_size], X[train_size:]
                        y_train, y_test = y[:train_size], y[train_size:]

                        # Reshape X agar sesuai dengan input yang diperlukan GRU [samples, timesteps, features]
                        X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
                        X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

                        # Membangun model GRU
                        model = Sequential()
                        model.add(GRU(units=num_units, return_sequences=(num_layers == 2), input_shape=(X_train.shape[1], X_train.shape[2])))
                        if num_layers == 2:
                            model.add(GRU(units=num_units))
                        model.add(Dense(1))
                        
                        # Kompilasi model
                        model.compile(optimizer='adam', loss='mean_squared_error')

                        progress.progress(60)  # Update progress to 60%

                        # Menggunakan EarlyStopping untuk mencegah overfitting
                        early_stop = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

                        # Melatih model
                        history = model.fit(
                            X_train, y_train, 
                            epochs=max_epochs, 
                            batch_size=batch_size, 
                            validation_split=0.2, 
                            callbacks=[early_stop], 
                            verbose=1
                        )

                        progress.progress(80)  # Update progress to 100%

                        # Prediksi ke masa depan
                        last_known_values = X_test[-1].reshape(1, 1, -1)  # Pastikan input memiliki dimensi [1, 1, features]
                        predictions = []

                        for _ in range(future_steps):
                            # Normalisasi nilai input sebelum prediksi
                            normalized_input = scaler_X.transform(last_known_values.reshape(1, -1)).reshape(1, 1, -1)
    
                            # Prediksi nilai baru
                            prediction = model.predict(last_known_values, verbose=0)[0][0]
                            predictions.append(prediction)

                            # Update input dengan nilai prediksi untuk iterasi berikutnya
                            new_input = np.append(last_known_values[0, 0, 1:], prediction).reshape(1, 1, -1)
                            last_known_values = new_input

                        # Membalikkan normalisasi pada prediksi
                        predictions = scaler_y.inverse_transform(np.array(predictions).reshape(-1, 1))

                        # Membuat DataFrame untuk prediksi
                        future_dates = pd.date_range(start=data.index[-1], periods=future_steps + 1, freq='H')[1:]
                        future_df = pd.DataFrame({
                            'Datetime': future_dates,
                            'Predicted 4G Total Traffic (GB)': predictions.flatten()
})

                        # Prediksi menggunakan model
                        y_pred_train = model.predict(X_train)
                        y_pred_test = model.predict(X_test)

                        # Membalikkan normalisasi pada prediksi dan data sebenarnya
                        y_train = scaler_y.inverse_transform(y_train)
                        y_test = scaler_y.inverse_transform(y_test)
                        y_pred_train = scaler_y.inverse_transform(y_pred_train)
                        y_pred_test = scaler_y.inverse_transform(y_pred_test)
                        
                
                progress.progress(100)  # Update progress to 100%
                
                # Catat waktu selesai
                end_time = time.time()

                # Hitung durasi
                duration = end_time - start_time

                st.success(f"Prediction complete in {duration:.2f} seconds!")
                    
                # Evaluasi model pada data uji
                mse = mean_squared_error(y_test, y_pred_test)
                mae = mean_absolute_error(y_test, y_pred_test)
                r2 = r2_score(y_test, y_pred_test)
                msle = mean_squared_log_error(y_test, y_pred_test)
                mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100

                # Display metrics
                st.markdown(f"### Evaluation metrics for {target_column} using {algorithm} in {selected_cell}:")
                col1, col2, col3, col4, col5 = st.columns(5)  # Menambahkan satu kolom lagi
                with col1:
                    st.metric(label="Mean Squared Error (MSE)", value=f"{mse:.4f}")
                with col2:
                    st.metric(label="Mean Absolute Error (MAE)", value=f"{mae:.4f}")
                with col3:
                    st.metric(label="R² Score", value=f"{r2:.4f}")
                with col4:
                    st.metric(label="MSLE", value=f"{msle:.4f}")
                with col5:
                    st.metric(label="MAPE", value=f"{mape:.2f}%")
                
                # Pastikan akses ke index asli sebelum split
                original_index = filtered_data.index[-len(y_test):]  # Ambil index data uji (y_test)

                # Kolom untuk Actual vs Predicted dan Prediksi ke Depan
                col1, col2 = st.columns(2)

                with col1:
                    plt.figure(figsize=(12, 6))  # Adjusted size for column layout
                    plt.plot(original_index, y_test, label='Actual', color='blue')
                    plt.plot(original_index, y_pred_test, label='Predicted', color='red', linestyle='--')
                    plt.title(f"Actual vs Predicted {target_column} using {algorithm}")
                    plt.xlabel("Datetime")
                    plt.ylabel(target_column)
                    plt.legend()
                    st.pyplot(plt)

                with col2:
                    plt.figure(figsize=(12, 6))  # Adjusted size for column layout
                    plt.plot(original_index[-120:], y_test[-120:], label='Data Sebenarnya', color='blue')
                    plt.plot(original_index[-120:], y_pred_test[-120:], label='Data Prediksi', color='red', linestyle='--')
                    plt.title(f"Actual vs Predicted {target_column} (5 Hari Terakhir) using {algorithm}")
                    plt.xlabel("Datetime")
                    plt.ylabel(target_column)
                    plt.legend()
                    st.pyplot(plt)

                # Kolom untuk Actual vs Predicted (5 Hari Terakhir) dan tabel komparasi
                col3, col4 = st.columns(2)

                with col3:                
                    st.write(f"### Future Prediction in {selected_cell}")
                    plt.figure(figsize=(12, 6))  # Adjusted size for column layout
                    plt.plot(future_df['Datetime'], future_df[f'Predicted {target_column}'], label='Prediksi', color='green')
                    plt.title("Future Prediction")
                    plt.xlabel("Datetime")
                    plt.ylabel(target_column)
                    plt.legend()
                    st.pyplot(plt)

                with col4:
                    # Tambahkan kembali identitas data ke dalam DataFrame
                    comparison_df = pd.DataFrame({
                        "Datetime": original_index,  # Gunakan index asli dari data uji
                        "Cell Name": selected_cell,  # Cell yang dipilih oleh pengguna
                        "Actual": y_test.flatten(),  # Data aktual
                        "Predicted": y_pred_test.flatten(),  # Prediksi
                        "Difference": y_test.flatten() - y_pred_test.flatten()  # Selisih antara Actual dan Predicted
                    })

                    # Filter tabel berdasarkan cell
                    comparison_df_filtered = comparison_df[comparison_df["Cell Name"] == selected_cell]

                    # Tampilkan tabel
                    st.write(f"### Actual vs Predicted Data for Cell: {selected_cell}")
                    st.dataframe(comparison_df_filtered)

                    # Tambahkan tombol untuk mengunduh tabel
                    csv = comparison_df_filtered.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label=f"Download Actual vs Predicted Data for {selected_cell} as CSV",
                        data=csv,
                        file_name=f'actual_vs_predicted_{selected_cell}.csv',
                        mime='text/csv',
                    )

                # Plot Actual vs Predicted
                plt.figure(figsize=(12, 6))
                plt.plot(original_index, y_test, label='Actual', color='blue')  # Data Aktual
                plt.plot(original_index, y_pred_test, label='Predicted', color='red', linestyle='--')  # Data Prediksi
                plt.plot(future_df['Datetime'], future_df['Predicted 4G Total Traffic (GB)'], 
                        label='Future Predictions LSTM', color='green', linestyle=':')  # Prediksi Masa Depan
                plt.title(f"Actual vs Predicted vs Future Predictions {target_column} using {algorithm}")
                plt.xlabel("Datetime")
                plt.ylabel(target_column)
                plt.legend()
                st.pyplot(plt)

                # Plot training and validation loss
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(history.history['loss'], label='Train Loss', color='blue')
                ax.plot(history.history['val_loss'], label='Validation Loss', color='orange')
                ax.set_title('Model Loss During Training')
                ax.set_xlabel('Epochs')
                ax.set_ylabel('Loss')
                ax.legend()
                ax.grid(True)

                # Display the plot in Streamlit
                st.pyplot(fig)

        if prediction_type == "Machine Learning":
            # Machine Learning Prediction Configuration
            algorithm = st.sidebar.selectbox("Choose Model", ["Random Forest", "Decision Tree", "KNN", "XGBoost"])

            feature_columns = [col for col in filtered_data.columns if col != target_column]

            # Parameter selection for each algorithm
            if algorithm == "Random Forest":
                st.sidebar.markdown("#### Random Forest Parameters")
                param_selection_mode = st.sidebar.radio("Parameter Selection Mode", ["Grid Search", "Manual Input"])

                if param_selection_mode == "Manual Input":
                    n_estimators = st.sidebar.number_input("Number of Trees (n_estimators)", min_value=1, max_value=500, value=100, step=10)
                    
                    # Checkbox untuk opsi None pada max_depth
                    no_max_depth = st.sidebar.checkbox("No maximum depth (None)")
                    if no_max_depth:
                        max_depth = None
                    else:
                        max_depth = st.sidebar.number_input("Max Depth", min_value=1, max_value=50, value=10, step=1)
                    
                    min_samples_split = st.sidebar.slider("Min Samples Split", min_value=2, max_value=10, value=2, step=1)

                    # Tambahkan max_features
                    max_features_option = st.sidebar.radio(
                        "Max Features",
                        options=["All Features", "sqrt", "log2", "Custom"],
                        index=0
                    )
                    if max_features_option == "All Features":
                        max_features = None
                    elif max_features_option == "Custom":
                        max_features = st.sidebar.number_input(
                            "Custom Max Features (int)",
                            min_value=1,
                            max_value=40,
                            value=10,
                            step=1
                        )
                    else:
                        max_features = max_features_option

                else:
                    n_estimators = None
                    max_depth = None
                    min_samples_split = None
                    max_features = None

            elif algorithm == "Decision Tree":
                st.sidebar.markdown("#### Decision Tree Parameters")
                param_selection_mode = st.sidebar.radio("Parameter Selection Mode", ["Grid Search", "Manual Input"])

                if param_selection_mode == "Manual Input":
                    max_depth = st.sidebar.slider("Max Depth", min_value=1, max_value=8, value=2, step=1)
                    min_samples_split = st.sidebar.slider("Min Samples Split", min_value=2, max_value=4, value=2, step=2)
                    min_samples_leaf = st.sidebar.slider("Min Samples Leaf", min_value=1, max_value=10, value=1, step=1)
                else:
                    max_depth = None
                    min_samples_split = None
                    min_samples_leaf = None

            elif algorithm == "KNN":
                st.sidebar.markdown("#### KNN Parameters")
                param_selection_mode = st.sidebar.radio("Parameter Selection Mode", ["Grid Search", "Manual Input"])

                if param_selection_mode == "Manual Input":
                    n_neighbors = st.sidebar.slider("Number of Neighbors (n_neighbors)", min_value=1, max_value=20, value=3, step=1)
                    weights = st.sidebar.selectbox("Weights", options=["uniform", "distance"])
                    metric = st.sidebar.selectbox("Distance Metric", options=["minkowski", "manhattan"])
                else:
                    n_neighbors = None
                    weights = None
                    metric = None

            elif algorithm == "XGBoost":
                st.sidebar.markdown("#### XGBoost Parameters")
                param_selection_mode = st.sidebar.radio("Parameter Selection Mode", ["Grid Search", "Manual Input"])

                if param_selection_mode == "Manual Input":
                    n_estimators = st.sidebar.slider("Number of Trees (n_estimators)", min_value=50, max_value=500, value=100, step=5)
                    learning_rate = st.sidebar.slider("Learning Rate", min_value=0.01, max_value=0.5, value=0.3, step=0.1)
                    max_depth = st.sidebar.slider("Max Depth", min_value=3, max_value=9, value=6, step=3)
                else:
                    n_estimators = None
                    learning_rate = None
                    max_depth = None
                    subsample = None

            # Training/Test Split
            test_split = st.sidebar.slider("Split for test/training", 0.1, 0.9, 0.3)

            # Menghapus nilai NaN yang dihasilkan oleh lag
            filtered_data = filtered_data.dropna()

            # Input untuk jumlah langkah prediksi masa depan
            future_steps = st.sidebar.number_input(
                "Number of Future Steps to Predict", 
                min_value=1, 
                max_value=240, 
                value=24, 
                step=1
            )

            # Start Predict Button for Machine Learning
            if st.sidebar.button("Start Predict"):
                # Catat waktu mulai
                start_time = time.time()
                progress = st.progress(0)
                with st.spinner(f"Starting Machine Learning Prediction with {algorithm}..."):
                    # Update progress to 20%
                    progress.progress(20)

                    # Set predictor and target columns
                    X = filtered_data[feature_columns]
                    y = filtered_data[target_column]

                    if algorithm in ["KNN", "XGBoost"]:
                        scaler_X = StandardScaler()
                        X_scaled = scaler_X.fit_transform(X)
                    else:
                        X_scaled = X

                    # Train/Test split
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, shuffle=False)

                    progress.progress(40)

                    if algorithm == "Random Forest":
                        if param_selection_mode == "Manual Input":
                            model = RandomForestRegressor(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                max_features=max_features,
                                random_state=42
                            )
                        elif param_selection_mode == "Grid Search":
                            pipeline = Pipeline([
                                ('scaler', StandardScaler()),
                                ('random_forest', RandomForestRegressor(random_state=42))
                            ])
                            param_grid = {
                                'random_forest__n_estimators': [50, 100, 200],
                                'random_forest__max_depth': [5, 10, 20, None],
                                'random_forest__min_samples_split': [2, 5, 10],
                                'random_forest__max_features': ['sqrt', 'log2', None, 0.5]
                            }
                            tscv = TimeSeriesSplit(n_splits=5)
                            grid_search = GridSearchCV(pipeline, param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
                            grid_search.fit(X_train, y_train)
                            model = grid_search.best_estimator_
                            best_params = grid_search.best_params_

                            progress.progress(80)

                            st.write("Best Parameters (Random Forest):", best_params)

                    elif algorithm == "Decision Tree":
                        if param_selection_mode == "Manual Input":
                            model = DecisionTreeRegressor(
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf,
                                random_state=42
                            )
                        elif param_selection_mode == "Grid Search":
                            pipeline = Pipeline([
                                ('scaler', StandardScaler()),
                                ('dtr', DecisionTreeRegressor(random_state=42))
                            ])
                            param_grid = {
                                'dtr__max_depth': [1, 2, 3, 4, 5, 6, 7, 8],
                                'dtr__min_samples_split': [2, 4],
                                'dtr__min_samples_leaf': [1, 2]
                            }
                            tscv = TimeSeriesSplit(n_splits=5)
                            grid_search = GridSearchCV(pipeline, param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
                            grid_search.fit(X_train, y_train)
                            model = grid_search.best_estimator_
                            best_params = grid_search.best_params_

                            progress.progress(80)

                            st.write("Best Parameters (Decision Tree):", best_params)

                    elif algorithm == "KNN":
                        if param_selection_mode == "Manual Input":
                            pipeline = Pipeline([
                                ('scaler', StandardScaler()),
                                ('knn', KNeighborsRegressor(
                                    n_neighbors=n_neighbors,
                                    weights=weights,
                                    metric=metric
                                ))
                            ])

                            # Fit the pipeline to the training data
                            pipeline.fit(X_train, y_train)
                            model = pipeline  # Assign the pipeline to model

                        elif param_selection_mode == "Grid Search":
                            pipeline = Pipeline([
                                ('scaler', StandardScaler()),
                                ('knn', KNeighborsRegressor())
                            ])
                            param_grid = {
                                'knn__n_neighbors': [3, 5, 10],
                                'knn__weights': ['uniform', 'distance'],
                                'knn__metric': ['minkowski', 'manhattan']
                            }
                            tscv = TimeSeriesSplit(n_splits=5)
                            grid_search = GridSearchCV(pipeline, param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
                            grid_search.fit(X_train, y_train)
                            model = grid_search.best_estimator_
                            best_params = grid_search.best_params_

                            progress.progress(80)

                            st.write("Best Parameters (KNN):", best_params)

                    elif algorithm == "XGBoost":
                        if param_selection_mode == "Manual Input":
                            model = XGBRegressor(
                                n_estimators=n_estimators,
                                learning_rate=learning_rate,
                                max_depth=max_depth,
                                subsample=subsample,
                                objective='reg:squarederror',
                                random_state=42
                            )
                        elif param_selection_mode == "Grid Search":
                            pipeline = Pipeline([
                                ('scaler', StandardScaler()),
                                ('xgbr', XGBRegressor(objective='reg:squarederror', random_state=42))
                            ])
                            param_grid = {
                                'xgbr__n_estimators': [5, 10, 50, 100],
                                'xgbr__learning_rate': [0.1, 0.3, 0.5],
                                'xgbr__max_depth': [3, 6, 9],
                            }
                            tscv = TimeSeriesSplit(n_splits=5)
                            grid_search = GridSearchCV(pipeline, param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
                            grid_search.fit(X_train, y_train)
                            model = grid_search.best_estimator_
                            best_params = grid_search.best_params_

                            progress.progress(80)

                            st.write("Best Parameters (XGBoost):", best_params)

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Prediksi Masa Depan
                last_known_values = X_test.values[-1].reshape(1, -1)
                future_predictions = []

                for _ in range(future_steps):
                    next_prediction = model.predict(last_known_values)[0]
                    future_predictions.append(next_prediction)
                    new_input = np.append(last_known_values[0, 1:], next_prediction).reshape(1, -1)
                    last_known_values = new_input

                future_dates = pd.date_range(start=y_test.index[-1], periods=future_steps + 1, freq='H')[1:]
                future_df = pd.DataFrame({
                    'Datetime': future_dates,
                    'Predicted': future_predictions
                })

                end_time = time.time()
                duration = end_time - start_time

                st.success(f"Prediction complete in {duration:.2f} seconds!")

                progress.progress(100)

                # Model Evaluation (For RF, DT,KNN, XGBoost)
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                msle = mean_squared_log_error(y_test, y_pred)
                mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

                # Display metrics
                st.markdown(f"### Evaluation metrics for {target_column} using {algorithm} in {selected_cell}:")
                col1, col2, col3, col4, col5 = st.columns(5)  # Menambahkan satu kolom lagi
                with col1:
                    st.metric(label="Mean Squared Error (MSE)", value=f"{mse:.4f}")
                with col2:
                    st.metric(label="Mean Absolute Error (MAE)", value=f"{mae:.4f}")
                with col3:
                    st.metric(label="R² Score", value=f"{r2:.4f}")
                with col4:
                    st.metric(label="MSLE", value=f"{msle:.4f}")
                with col5:
                    st.metric(label="MAPE", value=f"{mape:.2f}%")

                # Pastikan akses ke index asli sebelum split
                original_index = filtered_data.index[-len(y_test):]  # Ambil index data uji (y_test)

                # Kolom untuk Actual vs Predicted dan Prediksi ke Depan
                col1, col2 = st.columns(2)

                with col1:
                    # Plotting Actual vs Predicted (For RF, DT, KNN and XGBoost)
                    plt.figure(figsize=(12, 6))
                    plt.plot(y_test.index, y_test, label='Actual Data', color='blue')
                    plt.plot(y_test.index, y_pred, label='Predicted Data', color='red', linestyle='--')
                    plt.title(f"Actual vs Predicted {target_column} using {algorithm} in {selected_cell}")
                    plt.xlabel("Datetime")
                    plt.ylabel(target_column)
                    plt.legend()
                    st.pyplot(plt)

                with col2:
                    # Plot perbandingan antara data sebenarnya dan prediksi (default 120 jam terakhir)
                    plt.figure(figsize=(12, 6))
                    plt.plot(original_index[-120:], y_test[-120:], label='Data Sebenarnya', color='blue')
                    plt.plot(original_index[-120:], y_pred[-120:], label='Data Prediksi', color='red', linestyle='--')
                    plt.title(f"Actual vs Predicted {target_column} (120 Jam Terakhir) using {algorithm} in {selected_cell}")
                    plt.xlabel("Datetime")
                    plt.ylabel(target_column)
                    plt.legend()
                    st.pyplot(plt)

                # Kolom untuk Actual vs Predicted (5 Hari Terakhir) dan tabel komparasi
                col3, col4 = st.columns(2)

                with col3:                
                    st.write(f"### Future Prediction in {selected_cell}")
                    plt.figure(figsize=(12, 6))  # Adjusted size for column layout
                    plt.plot(future_df['Datetime'], future_df['Predicted'], label='Prediksi', color='green')
                    plt.title("Future Prediction")
                    plt.xlabel("Datetime")
                    plt.ylabel(target_column)
                    plt.legend()
                    st.pyplot(plt)

                with col4:
                    # Tambahkan kembali identitas data ke dalam DataFrame
                    comparison_df = pd.DataFrame({
                        "Datetime": filtered_data.index[-len(y_test):],  # Ambil tanggal dari data asli untuk subset data uji
                        "Cell Name": selected_cell,  # Cell yang dipilih
                        "Actual": y_test.values,  # Data aktual
                        "Predicted": y_pred,  # Data prediksi
                        "Difference": y_test.values - y_pred  # Selisih antara actual dan predicted
                    })

                    # Tampilkan tabel hasil
                    st.write(f"### Actual vs Predicted Data for Cell: {selected_cell}")
                    st.dataframe(comparison_df)

                    # Tambahkan tombol untuk mengunduh tabel
                    csv = comparison_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label=f"Download Actual vs Predicted Data for {selected_cell} as CSV",
                        data=csv,
                        file_name=f'actual_vs_predicted_{selected_cell}.csv',
                        mime='text/csv',
                    )
 
                # Plot Actual, Predicted, and Future Predictions
                plt.figure(figsize=(12, 6))
                plt.plot(y_test.index, y_test.values, label='Actual', color='blue')  # Plot data aktual
                plt.plot(y_test.index, y_pred, label='Predicted', color='red', linestyle='--')  # Plot prediksi
                plt.plot(future_df['Datetime'], future_df['Predicted'], label='Future Predictions', color='green', linestyle=':')  # Plot prediksi masa depan
                plt.title('Actual vs Predicted vs Future Predictions')  # Judul grafik
                plt.xlabel('Time')  # Label sumbu X
                plt.ylabel('Values')  # Label sumbu Y
                plt.legend()  # Tambahkan legenda
                st.pyplot(plt)  # Tampilkan plot pada Streamlit

                if algorithm == "Random Forest":
                    if param_selection_mode == "Grid Search":
                        # Feature Importance
                        feature_importances = model.named_steps['random_forest'].feature_importances_
                        importance_df = pd.DataFrame({
                            'Feature': feature_columns,
                            'Importance': feature_importances
                        }).sort_values(by='Importance', ascending=False)

                        st.write("Feature Importance (Random Forest):")
                        st.dataframe(importance_df)

                        plt.figure(figsize=(10, 6))
                        plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
                        plt.xlabel('Feature Importance')
                        plt.ylabel('Feature')
                        plt.title('Feature Importance for Random Forest')
                        plt.gca().invert_yaxis()
                        st.pyplot(plt)

                    if param_selection_mode == "Manual Input":
                        # Feature Importance
                        feature_importances = model.feature_importances_
                        importance_df = pd.DataFrame({
                            'Feature': feature_columns,
                            'Importance': feature_importances
                        }).sort_values(by='Importance', ascending=False)

                        st.write("Feature Importance (Random Forest):")
                        st.dataframe(importance_df)

                        # Plot Feature Importance
                        plt.figure(figsize=(10, 6))
                        plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
                        plt.xlabel('Feature Importance')
                        plt.ylabel('Feature')
                        plt.title('Feature Importance for Random Forest')
                        plt.gca().invert_yaxis()
                        st.pyplot(plt)
                
                if algorithm == "KNN":
                    if param_selection_mode == "Grid Search":
                        # Menghitung permutation importance pada model terbaik
                        result = permutation_importance(grid_search.best_estimator_, X_test, y_test, n_repeats=10, random_state=42, scoring='neg_mean_squared_error')

                        # Menyimpan hasil importance ke dalam DataFrame
                        importance_df = pd.DataFrame({
                            'Feature': feature_columns,
                            'Importance': result.importances_mean,
                            'Std Dev': result.importances_std
                        }).sort_values(by='Importance', ascending=False)

                        # Menampilkan hasil feature importance
                        st.write("Feature Importance (KNN):")
                        st.dataframe(importance_df)

                        # Plot feature importance
                        plt.figure(figsize=(12, 6))
                        plt.barh(importance_df['Feature'], importance_df['Importance'], xerr=importance_df['Std Dev'], color='skyblue')
                        plt.title('Feature Importance using Permutation Importance for KNN')
                        plt.xlabel('Importance')
                        plt.ylabel('Features')
                        plt.gca().invert_yaxis()
                        st.pyplot(plt)

                if algorithm == "Decision Tree":
                    if param_selection_mode == "Grid Search":
                        # Feature Importance
                        feature_importances = model.named_steps['dtr'].feature_importances_
                        importance_df = pd.DataFrame({
                            'Feature': feature_columns,
                            'Importance': feature_importances
                        }).sort_values(by='Importance', ascending=False)

                        st.write("Feature Importance (Decision Tree):")
                        st.dataframe(importance_df)

                        plt.figure(figsize=(10, 6))
                        plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
                        plt.xlabel('Feature Importance')
                        plt.ylabel('Feature')
                        plt.title('Feature Importance for Decision Tree')
                        plt.gca().invert_yaxis()
                        st.pyplot(plt)

                if algorithm == "XGBoost":
                    if param_selection_mode == "Grid Search":
                        # Mengambil feature importances dari model terbaik
                        importances = grid_search.best_estimator_.named_steps['xgbr'].feature_importances_
                        
                        # Membuat DataFrame untuk menampilkan feature importances
                        importance_df = pd.DataFrame({
                            'Feature': feature_columns,
                            'Importance': importances
                        }).sort_values(by='Importance', ascending=False)

                        # Menampilkan feature importances
                        st.write("Feature Importance (XGBoost):")
                        st.dataframe(importance_df)

                        # Plot feature importance
                        plt.figure(figsize=(10, 6))
                        plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
                        plt.gca().invert_yaxis()  # Membalik urutan fitur agar terpenting di atas
                        plt.title('Feature Importance for XGBoost Regressor')
                        plt.xlabel('Importance')
                        plt.ylabel('Feature')
                        st.pyplot(plt)

        # SARIMAX Prediction Code
        if prediction_type == "Statistic":
            # Statistic Prediction Configuration
            algorithm = st.sidebar.selectbox("Choose Model", ["SARIMA"])

            # Pilih metode input parameter
            param_mode = st.sidebar.radio("Parameter Selection Mode", ["Auto (Auto-ARIMA)", "Manual Input"])

            if param_mode == "Manual Input":
                # Parameter manual (p, d, q, P, D, Q, S)
                p = st.sidebar.number_input("ARIMA Order (p)", min_value=0, max_value=5, value=1, step=1)
                d = st.sidebar.number_input("ARIMA Differencing (d)", min_value=0, max_value=2, value=1, step=1)
                q = st.sidebar.number_input("ARIMA Moving Average (q)", min_value=0, max_value=5, value=1, step=1)
                P = st.sidebar.number_input("Seasonal Order (P)", min_value=0, max_value=3, value=1, step=1)
                D = st.sidebar.number_input("Seasonal Differencing (D)", min_value=0, max_value=2, value=1, step=1)
                Q = st.sidebar.number_input("Seasonal Moving Average (Q)", min_value=0, max_value=3, value=1, step=1)
                S = st.sidebar.number_input("Seasonal Period (S)", min_value=1, max_value=365, value=24, step=1)

            # Training/Test Split
            test_split = st.sidebar.slider("Split for test/training", 0.1, 0.9, 0.3)

            # Input jumlah langkah prediksi masa depan
            future_steps = st.sidebar.number_input(
                "Number of Future Steps to Predict",
                min_value=1,
                max_value=240,
                value=24,
                step=1
            )

            # Start Predict Button
            if st.sidebar.button("Start Predict"):
                # Catat waktu mulai
                start_time = time.time()

                progress = st.progress(0)
                with st.spinner(f"Starting Statistic prediction with {algorithm}..."):

                    # Update progress to 20%
                    progress.progress(20)

                    if algorithm == "SARIMA":
                        y = filtered_data[target_column]

                        # Pisahkan data menjadi train dan test
                        train_size = int(len(y) * (1 - test_split))
                        y_train, y_test = y[:train_size], y[train_size:]

                        if param_mode == "Auto (Auto-ARIMA)":
                            # Auto-ARIMA untuk mencari parameter terbaik
                            sarima_model = auto_arima(
                                y_train,
                                seasonal=True, m=24,  # m=24 untuk data musiman dengan periodisitas harian
                                start_p=0, start_q=0, max_p=5, max_q=5,
                                start_P=0, start_Q=0, max_P=2, max_Q=2,
                                d=None, D=None,  # Auto-ARIMA akan menentukan nilai differencing
                                trace=True,
                                error_action='ignore',
                                suppress_warnings=True,
                                stepwise=True
                            )
                            # Menampilkan parameter terbaik
                            st.write("Best SARIMA Parameters:", sarima_model.order, sarima_model.seasonal_order)

                            # Fit ulang model menggunakan parameter terbaik
                            sarima_fit = SARIMAX(
                                y_train,
                                order=sarima_model.order,
                                seasonal_order=sarima_model.seasonal_order,
                                enforce_stationarity=False,
                                enforce_invertibility=False
                            ).fit(disp=False)
                        else:
                            # Manual Input: Bangun model SARIMA menggunakan input pengguna
                            sarima_model = SARIMAX(
                                y_train,
                                order=(p, d, q),
                                seasonal_order=(P, D, Q, S),
                                enforce_stationarity=False,
                                enforce_invertibility=False
                            )
                            sarima_fit = sarima_model.fit(disp=False)

                        progress.progress(60)  # Update progress to 60%

                        # Prediksi pada data uji
                        y_pred = sarima_fit.predict(start=train_size, end=len(y) - 1)

                        # Prediksi ke masa depan
                        future_forecast = sarima_fit.get_forecast(steps=future_steps)
                        future_mean = future_forecast.predicted_mean
                        future_conf_int = future_forecast.conf_int()

                        # Sinkronkan panjang prediksi dengan y_test
                        if len(y_pred) != len(y_test):
                            st.warning(f"Length mismatch: Adjusting y_pred length from {len(y_pred)} to {len(y_test)}")
                            y_pred = pd.Series(y_pred[:len(y_test)], index=y_test.index)

                        progress.progress(80)  # Update progress to 80%

                        progress.progress(100)
                        st.success("Prediction complete!")
                        # Catat waktu selesai
                        end_time = time.time()
                        duration = end_time - start_time
                        st.write(f"Time taken for prediction: {duration:.2f} seconds")

                        # Evaluasi prediksi
                        mse = mean_squared_error(y_test, y_pred)
                        mae = mean_absolute_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

                        # Display metrics
                        st.markdown(f"### Evaluation metrics for {target_column} using {algorithm} ({param_mode}) in {selected_cell}:")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric(label="Mean Squared Error (MSE)", value=f"{mse:.4f}")
                        with col2:
                            st.metric(label="Mean Absolute Error (MAE)", value=f"{mae:.4f}")
                        with col3:
                            st.metric(label="R² Score", value=f"{r2:.4f}")
                        with col4:
                            st.metric(label="MAPE", value=f"{mape:.2f}%")
                        
                        # Pastikan akses ke index asli sebelum split
                        original_index = filtered_data.index[-len(y_test):]  # Ambil index data uji (y_test)

                        # Kolom untuk Actual vs Predicted dan Prediksi 3 Hari ke Depan
                        col1, col2 = st.columns(2)

                        with col1:
                            # Visualisasi hasil prediksi dan prediksi masa depan
                            plt.figure(figsize=(12, 6))
                            plt.plot(y_test.index, y_test, label="Actual", color="blue")
                            plt.plot(y_test.index, y_pred, label="Predicted", color="red", linestyle="--")
                            plt.plot(
                                pd.date_range(start=y_test.index[-1], periods=future_steps + 1, freq='H')[1:],
                                future_mean, label="Future Predictions", color="green", linestyle=":"
                            )
                            plt.fill_between(
                                pd.date_range(start=y_test.index[-1], periods=future_steps + 1, freq='H')[1:],
                                future_conf_int.iloc[:, 0], future_conf_int.iloc[:, 1],
                                color='green', alpha=0.2, label="Confidence Interval"
                            )
                            plt.title(f"SARIMA Model: Actual vs Predicted vs Future ({param_mode})")
                            plt.xlabel("Datetime")
                            plt.ylabel(target_column)
                            plt.legend()
                            st.pyplot(plt)

                        with col2:
                            # Plot perbandingan antara data sebenarnya dan prediksi (default 120 jam terakhir)
                            plt.figure(figsize=(12, 6))
                            plt.plot(original_index[-120:], y_test[-120:], label='Data Sebenarnya', color='blue')
                            plt.plot(original_index[-120:], y_pred[-120:], label='Data Prediksi', color='red', linestyle='--')
                            plt.title(f"Actual vs Predicted {target_column} (120 Jam Terakhir) using {algorithm} in {selected_cell}")
                            plt.xlabel("Datetime")
                            plt.ylabel(target_column)
                            plt.legend()
                            st.pyplot(plt)

        if prediction_type == "Hybrid":
            algorithm = st.sidebar.selectbox("Choose Model", ["LSTM + SARIMAX"])

             # Deep Learning Prediction Configuration
            feature_columns = [col for col in filtered_data.columns if col != target_column]
            
            # Pilihan parameter default atau custom
            parameter_mode = st.sidebar.radio("Parameter Mode", ["Default", "Custom"])

            if parameter_mode == "Custom":
                num_units = st.sidebar.slider("Jumlah Unit", min_value=10, max_value=256, value=128, step=10)
                batch_size = st.sidebar.selectbox("Batch Size", options=[16, 32, 64, 128], index=1)
                max_epochs = st.sidebar.slider("Epochs Max", min_value=10, max_value=500, value=100, step=10)
                patience = st.sidebar.slider("Patience (Epoch)", min_value=1, max_value=50, value=10, step=1)
                num_layers = st.sidebar.radio("Jumlah Layer", [1, 2, 3], index=1)

                sarimax_order = st.sidebar.text_input("SARIMAX Order (p,d,q)", "(1,1,1)")  # Example: (1,1,1)
                seasonal_order = st.sidebar.text_input("Seasonal Order (P,D,Q,S)", "(1,1,1,24)")  # Example: (1,1,1,24)

                sarimax_order = eval(sarimax_order)
                seasonal_order = eval(seasonal_order)
            else:
                # Default Parameters
                num_units = 128
                batch_size = 16
                max_epochs = 100
                patience = 50
                num_layers = 1

                sarimax_order = ("(1,1,1)")  # Example: (1,1,1)
                seasonal_order = ("(1,1,1,24)")  # Example: (1,1,1,24)

                sarimax_order = eval(sarimax_order)
                seasonal_order = eval(seasonal_order)

            # Training/Test Split
            test_split = st.sidebar.slider("Split for test/training", 0.1, 0.9, 0.3)

            # Menghapus nilai NaN yang dihasilkan oleh lag
            filtered_data = filtered_data.dropna()

            # Input untuk jumlah langkah prediksi masa depan
            future_steps = st.sidebar.number_input(
                "Number of Future Steps to Predict", 
                min_value=1, 
                max_value=720, 
                value=24, 
                step=1
            )
            
            if st.sidebar.button("Start Predict"):
                # Catat waktu mulai
                start_time = time.time()
                progress = st.progress(0)
                with st.spinner(f"Starting Hybrid Model (LSTM + SARIMAX)..."):
                    # Update progress to 20%
                    progress.progress(20)

                    # Prepare data for prediction
                    filtered_data['Hour'] = filtered_data.index.hour
                    filtered_data['Day'] = filtered_data.index.day
                    filtered_data['Month'] = filtered_data.index.month

                    # Set predictor and target columns
                    X = filtered_data[feature_columns]
                    y = filtered_data[target_column]

                    progress.progress(40)  # Update progress to 40%

                    # Train SARIMAX model
                    sarimax_model = SARIMAX(y, order=sarimax_order, seasonal_order=seasonal_order)
                    sarimax_fit = sarimax_model.fit(disp=False)
                    sarimax_predictions = sarimax_fit.predict(start=0, end=len(y)-1)

                    # Add SARIMAX predictions as a feature
                    filtered_data['SARIMAX_Predictions'] = sarimax_predictions

                    # Update features
                    feature_columns.append('SARIMAX_Predictions')
                    X = filtered_data[feature_columns]

                    # **Step 2: LSTM for Short-Term Prediction**
                    # Normalisasi data menggunakan MinMaxScaler
                    scaler_X = MinMaxScaler(feature_range=(0, 1))
                    scaler_y = MinMaxScaler(feature_range=(0, 1))

                    X = scaler_X.fit_transform(X)
                    y = scaler_y.fit_transform(y.values.reshape(-1, 1))

                    # Membagi data menjadi train dan test set
                    train_size = int(len(X) * (1 - test_split))
                    X_train, X_test = X[:train_size], X[train_size:]
                    y_train, y_test = y[:train_size], y[train_size:]

                    # Reshape X agar sesuai dengan input yang diperlukan LSTM
                    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
                    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

                    # Membangun model LSTM
                    model = Sequential()

                    if num_layers == 1:
                        # Jika hanya ada 1 layer, return_sequences harus False
                        model.add(LSTM(units=num_units, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))
                        model.add(Dropout(0.2))  # Menambahkan dropout dengan rate 20%
                    elif num_layers == 2:
                        # Jika ada 2 layer, layer pertama memiliki return_sequences=True
                        model.add(LSTM(units=num_units, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
                        model.add(Dropout(0.2))  # Menambahkan dropout setelah layer pertama
                        model.add(LSTM(units=num_units, return_sequences=False))
                        model.add(Dropout(0.2))  # Menambahkan dropout setelah layer kedua
                    elif num_layers == 3:
                        # Jika ada 3 layer, dua layer pertama memiliki return_sequences=True
                        model.add(LSTM(units=num_units, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
                        model.add(Dropout(0.2))  # Menambahkan dropout setelah layer pertama
                        model.add(LSTM(units=num_units, return_sequences=True))
                        model.add(Dropout(0.2))  # Menambahkan dropout setelah layer kedua
                        model.add(LSTM(units=num_units, return_sequences=False))
                        model.add(Dropout(0.2))  # Menambahkan dropout setelah layer ketiga

                    model.add(Dense(1))

                    # Kompilasi model
                    model.compile(optimizer='adam', loss='mean_squared_error')

                    progress.progress(60)

                    # Menggunakan EarlyStopping untuk mencegah overfitting
                    early_stop = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

                    # Melatih model
                    history = model.fit(
                        X_train, y_train,
                        epochs=max_epochs,
                        batch_size=batch_size,
                        validation_split=0.2,
                        callbacks=[early_stop],
                        verbose=1
                    )

                    progress.progress(80)

                    # **Step 3: Future Prediction**
                    # Prediksi menggunakan LSTM
                    y_pred_test = model.predict(X_test)
                    y_pred_test = scaler_y.inverse_transform(y_pred_test)

                    # Prediksi ke masa depan dengan SARIMAX
                    future_sarimax = sarimax_fit.get_forecast(steps=future_steps)
                    future_sarimax_mean = future_sarimax.predicted_mean

                    # Kombinasikan prediksi SARIMAX dan LSTM dengan skala yang sama
                    future_predictions = []
                    last_known_values = X_test[-1].reshape(1, 1, -1)
                    for i in range(future_steps):
                        # Prediksi LSTM
                        lstm_prediction = model.predict(last_known_values, verbose=0)[0][0]
                        
                        # Prediksi SARIMAX
                        sarimax_prediction = future_sarimax_mean.iloc[i]
                        
                        # Normalisasi SARIMAX agar setara dengan LSTM
                        sarimax_normalized = scaler_y.transform([[sarimax_prediction]])[0][0]
                        
                        # Kombinasikan menggunakan rata-rata berbobot
                        combined_prediction = 0.7 * lstm_prediction + 0.3 * sarimax_normalized
                        
                        # Tambahkan ke daftar prediksi
                        future_predictions.append(combined_prediction)
                        
                        # Update input dengan nilai prediksi
                        new_input = np.append(last_known_values[0, 0, 1:], lstm_prediction).reshape(1, 1, -1)
                        last_known_values = new_input

                    # Kembalikan ke skala asli
                    future_predictions = scaler_y.inverse_transform(np.array(future_predictions).reshape(-1, 1))

                    # Buat DataFrame untuk prediksi hybrid
                    future_dates = pd.date_range(start=filtered_data.index[-1], periods=future_steps + 1, freq='H')[1:]
                    future_df = pd.DataFrame({
                        'Datetime': future_dates,
                        'Predicted': future_predictions.flatten()
                    })

                    progress.progress(100)

                    # Catat waktu selesai
                    end_time = time.time()

                    # Hitung durasi
                    duration = end_time - start_time

                    # Tampilkan hasil prediksi dan waktu proses
                    st.success(f"Prediction complete in {duration:.2f} seconds!")

                    # Evaluasi model pada data uji
                    try:
                        # Kembalikan y_test ke skala asli
                        y_test_original = scaler_y.inverse_transform(y_test)

                        # Hitung metrik evaluasi menggunakan data asli
                        mse = mean_squared_error(y_test_original, y_pred_test)
                        mae = mean_absolute_error(y_test_original, y_pred_test)
                        r2 = r2_score(y_test_original, y_pred_test)
                        msle = mean_squared_log_error(y_test_original, y_pred_test)
                        mape = np.mean(np.abs((y_test_original - y_pred_test) / y_test_original)) * 100

                        # Display metrics
                        st.markdown(f"### Evaluation metrics for {target_column} using Hybrid LSTM + SARIMAX in {selected_cell}:")
                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            st.metric(label="Mean Squared Error (MSE)", value=f"{mse:.4f}")
                        with col2:
                            st.metric(label="Mean Absolute Error (MAE)", value=f"{mae:.4f}")
                        with col3:
                            st.metric(label="R² Score", value=f"{r2:.4f}")
                        with col4:
                            st.metric(label="MSLE", value=f"{msle:.4f}")
                        with col5:
                            st.metric(label="MAPE", value=f"{mape:.2f}%")
                    except Exception as e:
                        st.error(f"Error in evaluation metrics: {e}")

                    # Pastikan akses ke index asli sebelum split
                    original_index = filtered_data.index[-len(y_test):]  # Ambil index data uji (y_test)

                    # Visualisasi hasil perbaikan
                    plt.figure(figsize=(12, 6))
                    plt.plot(filtered_data.index[-len(y_test):], scaler_y.inverse_transform(y_test), label="Actual", color='blue')
                    plt.plot(filtered_data.index[-len(y_test):], y_pred_test, label="LSTM Predictions")
                    plt.plot(future_df['Datetime'], future_df['Predicted'], label="Future Predictions LSTM + SARIMA", linestyle="--", color="green")
                    plt.xlabel("Datetime")
                    plt.ylabel("4G Total Traffic (GB)")
                    plt.legend()
                    st.pyplot(plt)

                    st.write(f"### Future Predictions 4G Total Traffic in {selected_cell}")
                    plt.figure(figsize=(12, 6))
                    plt.plot(future_df['Datetime'], future_df['Predicted'], label='Future Predictions', color='green', linestyle='-.')
                    plt.title("Future Predictions: 4G Total Traffic (GB)")
                    plt.xlabel("Datetime")
                    plt.ylabel(target_column)
                    plt.legend()
                    st.pyplot(plt)

                    # Plot training and validation loss
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(history.history['loss'], label='Train Loss', color='blue')
                    ax.plot(history.history['val_loss'], label='Validation Loss', color='orange')
                    ax.set_title('Model Loss During Training')
                    ax.set_xlabel('Epochs')
                    ax.set_ylabel('Loss')
                    ax.legend()
                    ax.grid(True)

                    # Display the plot in Streamlit
                    st.pyplot(fig)
                                           
    elif menu == "Data Visualization":
        st.subheader("Data Visualization")

        # Sidebar: Visualization Configuration
        if 'Cell Name' in data.columns:
            unique_cell_names = data['Cell Name'].unique()
            selected_cell_vis = st.sidebar.selectbox("Select Cell Name for Visualization", unique_cell_names)
            data_vis = data[data['Cell Name'] == selected_cell_vis]
        else:
            st.warning("Column 'Cell Name' not found in the dataset.")
            data_vis = data  

        # Display Filtered Data
        st.write(f"### Filtered Data for Selected Cell")

        # Show table with all data but initial display is limited to a scrollable view
        st.dataframe(data_vis, height=200)  # Adjust the height to limit visible rows

        # Display number of rows and columns
        num_rows, num_cols = data_vis.shape
        st.caption(f"Jumlah baris: {num_rows}, Jumlah kolom: {num_cols}")

        target_column_vis = st.sidebar.selectbox("Select Column to Visualize", data_vis.select_dtypes(include=[np.number]).columns)

        # Visualisasi 4G Total Traffic Per Cell Name
        st.write(f"### {target_column_vis} Per Cell Name")
        traffic_per_cell = (
            data.groupby([data.index, 'Cell Name'])[target_column_vis]
            .sum()
            .reset_index()
        )

        # Pivot data agar tiap Cell Name menjadi kolom
        pivot_data = traffic_per_cell.pivot(index='Datetime', columns='Cell Name', values=target_column_vis)
        pivot_data = pivot_data.fillna(0)

        # Membuat figure plotly
        fig = go.Figure()

        # Menambahkan line untuk setiap 'Cell Name'
        for cell in pivot_data.columns:
            fig.add_trace(go.Scatter(
                x=pivot_data.index,
                y=pivot_data[cell],
                mode='lines',
                name=cell
            ))

        # Update layout
        fig.update_layout(
            title=(f'{target_column_vis} Per Cell Name'),
            xaxis_title='Waktu',
            yaxis_title=target_column_vis,
            legend_title='Cell Name',
            hovermode="x unified",
            template="plotly_dark",
            margin=dict(l=0, r=0, t=40, b=40),  # Mengurangi margin agar plot lebih besar
            width=1200  # Tentukan lebar plot, bisa disesuaikan dengan kebutuhan
        )

        # Menampilkan plot
        st.plotly_chart(fig, use_container_width=True)  # Menyesuaikan dengan lebar container

        # Kolom untuk 4G Total Traffic dan 4G Active User
        col_4g_total_traffic = "4G Total Traffic (GB)"
        col_4g_active_user = "4G Active User"
       
        # Menampilkan Top 5 Cell Name untuk Total Traffic dan Active User
        st.write("### Ranking All Cells by Total Traffic and Active Users")

        # Menghitung top 5 cell dengan total traffic tertinggi
        top_5_traffic = data.groupby('Cell Name')[col_4g_total_traffic].sum().nlargest(19)

        # Menghitung top 5 cell dengan active user tertinggi
        top_5_users = data.groupby('Cell Name')[col_4g_active_user].sum().nlargest(19)

        # Membuat dua kolom sejajar
        col1, col2 = st.columns(2)

        # Plot untuk Top 5 Total Traffic
        with col1:
            plt.figure(figsize=(6, 6))
            top_5_traffic.plot(kind='bar', color='mediumseagreen', edgecolor='black')
            plt.title(f"Ranking of Cells by {col_4g_total_traffic}")
            plt.xlabel("Cell Name")
            plt.ylabel(col_4g_total_traffic)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(plt)

        # Plot untuk Top 5 Active User
        with col2:
            plt.figure(figsize=(6, 6))
            top_5_users.plot(kind='bar', color='steelblue', edgecolor='black')
            plt.title(f"Ranking of Cells {col_4g_active_user}")
            plt.xlabel("Cell Name")
            plt.ylabel(col_4g_active_user)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(plt)

        if st.sidebar.button("Start Visualization"):
            progress = st.progress(0)
            with st.spinner(f"Starting Visualization for {target_column_vis}..."):
                progress.progress(0)  # Update progress to 0%
                st.write("###", target_column_vis, "Visualization")
                plt.figure(figsize=(20, 5))
                plt.plot(data_vis.index, data_vis[target_column_vis])
                plt.title(f"{target_column_vis} for {selected_cell_vis}")
                plt.xlabel("Datetime")
                plt.ylabel(target_column_vis)
                st.pyplot(plt)
                progress.progress(30)  # Update progress to 30%

                # Weekly
                start_weekly = '2024-02-02 00:00:00'
                end_weekly = '2024-02-09 23:59:59'
                data_mingguan = data_vis.loc[start_weekly:end_weekly]
                date = data_mingguan.index  
                st.write("### Weekly", target_column_vis, "Visualization")
                plt.figure(figsize=(20, 5))
                plt.plot(data_mingguan.index, data_mingguan[target_column_vis])
                plt.title(f"Daily {target_column_vis} for {selected_cell_vis}")
                plt.xlabel("Datetime")
                plt.ylabel(target_column_vis)
                st.pyplot(plt)
                progress.progress(60)  # Update progress to 60%

                # Daily
                start_daily = '2024-02-02 00:00:00'
                end_daily = '2024-02-02 23:59:59'
                data_harian = data_vis.loc[start_daily:end_daily]
                date = data_harian.index  
                st.write("### Daily", target_column_vis, "Visualization")
                plt.figure(figsize=(20, 5))
                plt.plot(data_harian.index, data_harian[target_column_vis])
                plt.title(f"Daily {target_column_vis} for {selected_cell_vis}")
                plt.xlabel("Datetime")
                plt.ylabel(target_column_vis)
                st.pyplot(plt)
                progress.progress(100)  # Update progress to 100%

else:
    st.write("Please upload a CSV file to get started.")
