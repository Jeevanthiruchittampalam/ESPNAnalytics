import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Corrected directory path
directory_path = 'rq3\data'

seasons = range(2000, 2020)  # Seasons from 2000 to 2019
files = [f'{directory_path}/england-premier-league-{season}-to-{season+1}.csv' for season in seasons]

# Load and concatenate data
data_frames = []
for file in files:
    try:
        df = pd.read_csv(file, on_bad_lines='skip', encoding='ISO-8859-1')
        data_frames.append(df)
    except Exception as e:
        print(f"Error reading file: {file} - {e}")
data = pd.concat(data_frames, ignore_index=True)

# Check for required columns and handle missing values
required_columns = ['Attendance', 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']
if not all(col in data.columns for col in required_columns):
    raise ValueError("Not all required columns are present in the data")
data.dropna(subset=required_columns, inplace=True)

# Feature Engineering
data = data.assign(
    TotalAwayCards=data['AY'] + data['AR'],
    HomeAdvantage=(data['FTHG'] > data['FTAG']).astype(int),
    AttendanceScaled=np.log1p(data['Attendance'])
)

# EDA: Descriptive statistics and visualizations
print(data[['TotalAwayCards', 'AttendanceScaled']].describe())

# Histogram of Total Away Cards
plt.figure(figsize=(10, 5))
data['TotalAwayCards'].plot(kind='hist', title='Distribution of Total Away Cards')
plt.xlabel('Total Away Cards')
plt.ylabel('Frequency')
plt.show()

# Box plot for outliers in Attendance
plt.figure(figsize=(10, 5))
plt.boxplot(data['Attendance'])
plt.title('Box Plot of Attendance')
plt.ylabel('Attendance')
plt.show()

# Model setup and training
X = data[['HomeAdvantage', 'AttendanceScaled']]
y = data['TotalAwayCards']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Cross-validation with Negative Binomial Regression
kf = KFold(n_splits=5, shuffle=True, random_state=42)
nb_results_list = []
model_scores = []
mae_scores = []

for train_index, test_index in kf.split(X):
    X_train, X_test = y.iloc[train_index], y.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    nb_model = sm.GLM(y_train, sm.add_constant(X_train), family=sm.families.NegativeBinomial())
    nb_results = nb_model.fit()
    nb_results_list.append(nb_results)
    y_pred = nb_results.predict(sm.add_constant(X_test))
    
    # Evaluation metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    model_scores.append(rmse)
    mae_scores.append(mae)

# Average RMSE and MAE
print(f"Average RMSE across all folds: {np.mean(model_scores)}")
print(f"Average MAE across all folds: {np.mean(mae_scores)}")

# Visualizations of Predicted vs Actual
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, alpha=0.2)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Actual Total Away Cards')
plt.ylabel('Predicted Total Away Cards')
plt.title('Predicted vs Actual')
plt.show()

# Residuals Plot
residuals = y_test - y_pred
plt.figure(figsize=(10, 5))
plt.scatter(y_pred, residuals, alpha=0.2)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Total Away Cards')
plt.ylabel('Residuals')
plt.title('Residuals Plot')
plt.show()

# Print summaries of the fitted models
for result in nb_results_list:
    print(result.summary())
