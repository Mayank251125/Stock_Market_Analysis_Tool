# Part 1 - Data Preprocessing

import numpy as np
import pandas as pd
import os
import pickle

# Importing the training set
dataset_train = pd.read_csv('data/Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
required_timesteps = 60 # Define required_timesteps for clarity

# Ensure loop range is valid
if len(training_set_scaled) >= required_timesteps:
    for i in range(required_timesteps, len(training_set_scaled)):
        X_train.append(training_set_scaled[i-required_timesteps:i, 0])
        y_train.append(training_set_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
else:
    print(f"Error: Training set has only {len(training_set_scaled)} samples, but requires at least {required_timesteps} for timesteps.")
    X_train, y_train = np.array([]), np.array([]) # Initialize as empty arrays to prevent errors later

# Reshaping
if X_train.size > 0: # Only reshape if X_train is not empty
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
else:
    print("X_train is empty, skipping reshape.")

# Part 2 - Building the RNN

# Correct Keras imports for TensorFlow 2.16+
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import load_model # For saving/loading models directly from tf.keras

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
# Use required_timesteps for input_shape
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (required_timesteps, 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
if X_train.size > 0 and y_train.size > 0:
    print("Starting model training...")
    regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
    print("Model training finished.")

    # --- Saving the trained model and scaler ---
    model_path = 'models/google_stock_lstm_model.keras'
    scaler_path = 'models/min_max_scaler.pkl'

    print(f"Model will be saved to: {os.path.abspath(model_path)}")
    print(f"Scaler will be saved to: {os.path.abspath(scaler_path)}")

    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
        print("Directory 'models' created.")
    else:
        print("Directory 'models' already exists.")

    try:
        # Save the trained model in Keras's native format (implicitly uses SavedModel)
        regressor.save(model_path)
        print("LSTM model saved successfully!")
    except Exception as e:
        print(f"ERROR: Failed to save LSTM model: {e}")

    try:
        # Save the scaler
        with open(scaler_path, 'wb') as f:
            pickle.dump(sc, f)
        print("MinMaxScaler saved successfully!")
    except Exception as e:
        print(f"ERROR: Failed to save MinMaxScaler: {e}")
else:
    print("Skipping model training and saving because X_train or y_train is empty.")


print("rnn.py script finished.")

# Note: The original 'Part 3' for visualization from the notebook should ideally
# not be part of this script as it's run for model saving/training, not for deployment.
# If you have visualization code here, ensure it's commented out or removed.