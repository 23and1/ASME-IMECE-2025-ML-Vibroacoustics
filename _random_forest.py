# Author: Gershom Richards
# Revision: 0
# Original Date: April 27, 2025
# Last Revision Date: April 27, 2025

# Objective: This code creates a random forest 

import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Define filepath locations for the frequency-domain features, training labels, and training model results
features_dir = r"C:\\Users\\23and\\OneDrive - Kennesaw State University\\ENGR 9900 - Dissertation Research\\Dissertation Research\\Vibroacoustic Data\\channel_feature_outputs\\processed_data"
label_file = r"C:\\Users\\23and\\OneDrive - Kennesaw State University\\ENGR 9900 - Dissertation Research\\Dissertation Research\\Vibroacoustic Data\\labels_random_forest_600.csv"
output_model_results = r"C:\\Users\\23and\\OneDrive - Kennesaw State University\\ENGR 9900 - Dissertation Research\\Dissertation Research\\Vibroacoustic Data\\channel_feature_outputs\\_random_forest_Model_Predictions.csv"
label_columns = ['Operation', 'Anomaly', 'Fault']  # <-- Modify with your actual label columns

# Function to remove the hidden characters in the filename CSVs
def clean_filename(raw_name):
    cleaned = str(raw_name).strip()
    cleaned = cleaned.replace('\u202a', '').replace('\ufeff', '').replace('\xa0', '')
    cleaned = re.sub(r'[\r\n\t]+', '', cleaned)
    return cleaned

# Load & combine all the feature CSVs
all_features = []
for file in os.listdir(features_dir):
    if file.endswith("_features.csv"):
        df = pd.read_csv(os.path.join(features_dir, file))
        all_features.append(df)

if not all_features:
    raise ValueError("❌ No feature CSVs found. Check the directory path.")

features_df = pd.concat(all_features, ignore_index=True)
features_df['Source File'] = features_df['Source File'].apply(clean_filename)
features_df.to_csv('features_concat.csv')

# === Load Label CSV and Clean ===
labels_df = pd.read_csv(label_file)
labels_df['Source File'] = labels_df['Source File'].apply(clean_filename)

# === Merge Features + Labels ===
merged_df = pd.merge(features_df, labels_df, on='Source File', how='inner')

# === Prepare Data ===
feature_cols = [
    'Mean', 'Median', 'Variance', 'Std Dev', 'RMS', 'Skewness', 'Kurtosis',
    'Dominant Frequency [Hz]', 'Instantaneous Frequency [Hz]', 
    'Spectral Entropy', 'Max Amplitude'
]

X = merged_df[feature_cols]
Y = merged_df[label_columns]

# === Train-Test Split ===
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y[label_columns[0]], random_state=42
)

# === Train Random Forest Model ===
base_model = RandomForestClassifier(n_estimators=100, random_state=42)
multi_model = MultiOutputClassifier(base_model)
multi_model.fit(X_train, Y_train)

Y_pred = multi_model.predict(X_test)
Y_pred_df = pd.DataFrame(Y_pred, columns=label_columns)

# === Evaluate Model and Plot Feature Importances ===
importances_per_label = {}
for i, col in enumerate(label_columns):
    print(f"\n📊 Classification Report for '{col}'")
    print(classification_report(Y_test[col], Y_pred_df[col]))

    # Feature Importances per label
    importances = pd.Series(multi_model.estimators_[i].feature_importances_, index=feature_cols)
    importances_per_label[col] = importances

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances.values, y=importances.index)
    plt.title(f"Feature Importance for '{col}'")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

    # Confusion Matrix
    cm = confusion_matrix(Y_test[col], Y_pred_df[col])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(Y_test[col]))
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title(f"Confusion Matrix for '{col}'")
    plt.tight_layout()
    plt.show()

# === Average Feature Importance ===
importances_df = pd.DataFrame(importances_per_label)
importances_df['Average'] = importances_df.mean(axis=1)
importances_df_sorted = importances_df.sort_values(by='Average', ascending=False)

# Plot averaged importances
plt.figure(figsize=(10, 6))
sns.barplot(x=importances_df_sorted['Average'].values, y=importances_df_sorted.index)
plt.title("Average Feature Importance Across All Labels")
plt.xlabel("Average Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# === Save Predictions ===
pred_df = X_test.copy()
pred_df['Source File'] = merged_df.loc[Y_test.index, 'Source File'].values

for col in label_columns:
    pred_df[f"{col}_True"] = Y_test[col].values
    pred_df[f"{col}_Pred"] = Y_pred_df[col].values

pred_df.to_csv(output_model_results, index=False)
print(f"\n✅ Saved predictions and results to: {output_model_results}")
