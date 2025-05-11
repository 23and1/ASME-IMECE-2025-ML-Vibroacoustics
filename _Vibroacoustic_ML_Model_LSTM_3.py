import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt

# Function to load labels from external file (multi-label)
def load_labels(label_file):
    labels_df = pd.read_csv(label_file)  # Assumes columns: "filename", "fault", "anomaly"
    labels_df.set_index("filename", inplace=True)
    return labels_df

# Function to load and preprocess CSV files
def load_and_preprocess_datasets(folder_path, label_file, max_time_steps=200):
    """
    Loads all CSV files from a folder, normalizes timestamps, scales features,
    and pads sequences for LSTM input.
    """
    file_paths = glob.glob(os.path.join(folder_path, "*.csv"))
    label_df = load_labels(label_file)
    datasets = []
    labels = []
    scaler = MinMaxScaler()

    for file in file_paths:
        filename = os.path.basename(file).strip().lower()
        label_df.index = label_df.index.str.strip().str.lower()

        if filename not in label_df.index:
            print(f"Skipping {filename} (no label found)")
            continue

        data = pd.read_csv(file)

        if data.shape[0] == 0:
            print(f"Skipping {filename} (empty file)")
            continue

        # Normalize timestamps to relative time (seconds from start)
        data.iloc[:, 0] = data.iloc[:, 0] - data.iloc[0, 0]
        
        # Normalize timestamps to range 0-1
        max_time = data.iloc[:, 0].max()
        if max_time > 0:
            data.iloc[:, 0] = data.iloc[:, 0] / max_time

        # Extract numerical features (including normalized timestamps)
        features = data.values  # Keep all columns, including timestamps
        features = scaler.fit_transform(features)
        
        # Pad sequences to max_time_steps
        padded_features = pad_sequences([features], maxlen=max_time_steps, padding='post', dtype='float32')[0]

        datasets.append(padded_features)
        labels.append(label_df.loc[filename].values.astype(np.float32))

    if len(datasets) == 0:
        raise ValueError("No valid datasets loaded. Check file paths and label matching.")

    # Convert to NumPy arrays
    X = np.array(datasets)
    y = np.array(labels) # Ensure labels are numerical
    
    print(f"Loaded {len(X)} datasets")
    return X, y

# Function to build and tune LSTM model for multi-label classification
def build_lstm_model(hp):
    model = Sequential([
        LSTM(hp.Int('units_1', 32, 128, step=32), return_sequences=True, activation=None, recurrent_activation=None, recurrent_dropout=hp.Float('recurrent_dropout_1', 0.0, 0.4, step=0.1)),
        Dropout(hp.Float('dropout_1', 0.2, 0.5, step=0.1)),
        LSTM(hp.Int('units_2', 16, 64, step=16), return_sequences=False, activation=None, recurrent_activation=None),
        Dropout(hp.Float('dropout_2', 0.2, 0.5, step=0.1)),
        Dense(hp.Int('dense_units', 8, 32, step=8), activation='relu'),
        Dense(2, activation='sigmoid')  # Multi-label binary output for Fault and Anomaly
    ])

    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=hp.Choice('learning_rate', [0.001, 0.0005, 0.0001])),
        metrics=['accuracy']
    )
    return model

# Define the folder path and label filepath for datasets
folder_path = r'C:\Users\23and\OneDrive - Kennesaw State University\ENGR 9900 - Dissertation Research\Dissertation Research\Vibroacoustic Data\Training Data'
label_file = r'C:\Users\23and\OneDrive - Kennesaw State University\ENGR 9900 - Dissertation Research\Dissertation Research\Vibroacoustic Data\labels_fwd_clean_no_outliers.csv'
X, y = load_and_preprocess_datasets(folder_path, label_file)

# Split data into training, validation, and test sets
if len(X) > 0:
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Hyperparameter tuning
    tuner = kt.Hyperband(
        build_lstm_model,
        objective='val_accuracy',
        max_epochs=20,
        factor=3,
        directory='tuner_results',
        project_name='lstm_multilabel_tuning_v3_fwd_clean_no_outliers_relu'
    )

    tuner.search(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32)

    # Get the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(1)[0]
    
    # Build and train the best model
    model = tuner.hypermodel.build(best_hps)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32)

    # Evaluate and print confusion matrices for each label
    y_pred = model.predict(X_test) > 0.5
    y_pred = y_pred.astype(int)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['fault', 'anomaly']))

    # Save the trained model
    model.save("lstm_model_600_multi_fwd_clean_no_outliers_none.h5")
    print("Model saved successfully as lstm_model_600_multi_fwd_clean_no_outliers_none.h5")

    for i, label in enumerate(['fault', 'anomaly']):
        cm = confusion_matrix(y_test[:, i], y_pred[:, i])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {label}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
else:
    print("No data available for training. Fix errors above.")
