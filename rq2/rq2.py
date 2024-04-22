import pandas as pd
import numpy as np
import os
import warnings
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from scipy.stats import spearmanr
from math import sqrt
from sklearn.model_selection import TimeSeriesSplit

# Created a combined excel files for RQ2 (Only specific files for corresponding years are used)
base_path = 'C:/Users/richa/Desktop/ESPNAnalytics/England-premier-league/england-premier-league-'
file_paths = [f"{base_path}{year}-to-{year+1}.csv" for year in range(2000, 2019)]

# Create an empty DataFrame to store all data
all_premier_league = pd.DataFrame()

# Define all possible columns needed for RQ2 involving some matches
all_columns = ['Date', 'HomeTeam', 'AwayTeam', 'Attendance', 'Referee', 'HY', 'AY', 'HR', 'AR', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR', 'B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'GBH', 'GBD', 'GBA', 'IWH', 'IWD', 'IWA', 'LBH', 'LBD', 'LBA', 'SBH', 'SBD', 'SBA', 'WHH', 'WHD', 'WHA', 'SJH', 'SJD', 'SJA', 'VCH', 'VCD', 'VCA', 'BSH', 'BSD', 'BSA', 'Bb1X2', 'BbMxH', 'BbAvH', 'BbMxD', 'BbAvD', 'BbMxA', 'BbAvA', 'BbOU', 'BbAH', 'BbAHh', 'BbMxAHH', 'BbAvAHH', 'BbMxAHA', 'BbAvAHA', 'PSCH', 'PSCD', 'PSCA']

# Read all files and concatenate them into a single DataFrame
for file in file_paths:
    filename = os.path.splitext(os.path.basename(file))[0]
    try:
        df = pd.read_csv(file, encoding='latin1')
    except pd.errors.ParserError as e:
        print(f"Error parsing file {file}: {e}")
        continue

     # Add missing columns with value 0
    missing_columns = set(all_columns) - set(df.columns)
    for col in missing_columns:
        df[col] = 'N/A'

    # Reorder columns
    df = df.reindex(columns=all_columns)

    # Adjust all_columns based on the columns in the current file
    if all_premier_league.empty:
        all_columns = df.columns.tolist()
    else:
        all_columns = all_premier_league.columns.tolist()
    
    # Fill missing data with 0
    df = df.fillna('N/A')

    for col in df.columns:
        if df[col].dtype == 'float64' or df[col].dtype == 'int64':
            df[col] = df[col].replace('N/A', df[col].astype(float).mean())

    # Convert back to 'N/A' strings
    df = df.replace({np.nan: 'N/A', None: 'N/A'})

    df['File'] = filename
    all_premier_league = pd.concat([all_premier_league, df])

# Save under rq2 folder
output_dir = 'rq2'
os.makedirs(output_dir, exist_ok=True)

# Save DataFrame to CSV file
csv_filename = os.path.join(output_dir, 'rq2_premier_league.csv')
all_premier_league.to_csv(csv_filename, index=False)

# print(f"CSV file saved to: {csv_filename}\n")

#=========================================
# RQ2: Predicting Match Outcomes Table

warnings.filterwarnings('ignore', category=FutureWarning)

# Load the data
df = pd.read_csv('rq2/rq2_premier_league.csv')

# Convert 'HY', 'AY', 'HR', 'AR' etc. to numeric, coercing errors to NaN
numeric_columns = ['HY', 'AY', 'HR', 'AR', 'B365H', 'B365D', 'B365A']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Data Preparation: Normalizing features
scaler = MinMaxScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Drop rows with NaN values in the specified columns
df.dropna(subset=numeric_columns, inplace=True)

# Create a binary outcome column for home team win
df['HomeWin'] = df['FTR'].apply(lambda x: 1 if x == 'H' else 0)

# Statistical Analysis: Spearman’s Rank Correlation
spearman_correlation = df[numeric_columns + ['HomeWin']].corr(method='spearman')

# Predicting match outcomes using Random Forest
X = df[numeric_columns]  # Features
y = df['HomeWin']  # Target variable

# Time-series cross-validation setup
tscv = TimeSeriesSplit(n_splits=5)

# Placeholder for cross-validated accuracy scores
accuracy_scores = []

# Perform time-series cross-validation
for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy_scores.append(accuracy_score(y_test, y_pred))

# Print cross-validated accuracies
print(f'Cross-validated accuracies: {accuracy_scores}')

# Logistic Regression for hypothesis testing
logit_model = sm.Logit(y, sm.add_constant(X)).fit()
print(logit_model.summary())

# Sensitivity Analysis
# Change 'HY' (Home Yellow Cards) to see effect on prediction
X_test_sensitivity = X_test.copy()
X_test_sensitivity['HY'] = X_test_sensitivity['HY'] * 1.1  # Increase yellow cards by 10%
y_pred_sensitivity = model.predict(X_test_sensitivity)

# Compare predictions
changes = np.mean(y_pred_sensitivity != y_pred)
print(f'Changes in predictions after increasing home yellow cards by 10%: {changes:.2%}')

# Print Spearman correlation results
print(spearman_correlation)