import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error

# Define directory path assuming the script and data folder are at the same level
directory_path = '../England-premier-league'
seasons = range(2014, 2019)  # Adjusted for your sample data years
files = [f'{directory_path}/england-premier-league-{season}-to-{season+1}.csv' for season in seasons]

# Load and concatenate data
data = pd.concat([pd.read_csv(file) for file in files], ignore_index=True)

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

# Exploratory Data Analysis
print(data[['TotalAwayCards', 'AttendanceScaled']].describe())
data['TotalAwayCards'].plot(kind='hist', title='Distribution of Total Away Cards')

# Model setup and training
X = data[['HomeAdvantage', 'AttendanceScaled']]
y = data['TotalAwayCards']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Cross-validation with Negative Binomial Regression
kf = KFold(n_splits=5, shuffle=True, random_state=42)
model_scores = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    nb_model = sm.GLM(y_train, sm.add_constant(X_train), family=sm.families.NegativeBinomial())
    nb_results = nb_model.fit()
    y_pred = nb_results.predict(sm.add_constant(X_test))
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    model_scores.append(rmse)

# Print average RMSE
print(f"Average RMSE across all folds: {np.mean(model_scores)}")

# Visualizations
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, alpha=0.2)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Negative Binomial Regression Predicted vs Actual')
plt.show()

residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.2)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Residuals Plot')
plt.show()
