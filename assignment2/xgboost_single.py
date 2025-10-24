import time
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

#https://www.kaggle.com/competitions/home-data-for-ml-course/data?select=data_description.txt

# Modules to load in the ULHPC cluster
#module load data/scikit-learn
#module load vis/matplotlib
#module load bio/Seaborn/0.13.2-gfbf-2023b

# Load your virtual environment where XGboost is installed
#source xgboost_env/bin/activate

# Read the data
X = pd.read_csv('train.csv.gz', index_col='Id')

# Remove rows with missing target, separate target from predictors
X.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X.SalePrice
X.drop(['SalePrice'], axis=1, inplace=True)

# Break off validation set from training data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                random_state=0)

# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 
                        X_train_full[cname].dtype == "object"]

# Select numeric columns
numeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

# Keep selected columns only
my_cols = low_cardinality_cols + numeric_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()

# One-hot encode the data (to shorten the code, we use pandas)
X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)
X_train, X_valid = X_train.align(X_valid, join='left', axis=1)

start_time = time.time()
# Train the model
model = XGBRegressor(
    # Basic parameters (vanilla setup)
    objective='reg:squarederror',  # Regression with squared loss
    n_estimators=100,              # Number of boosting rounds
    max_depth=6,                   # Maximum tree depth
    learning_rate=0.1,             # Step size shrinkage
    random_state=42,               # For reproducibility
    subsample=1.0,               # Use all data for each tree
    colsample_bytree=1.0,        # Use all features for each tree
    reg_lambda=1,                # L2 regularization
    reg_alpha=0,                 # L1 regularization
)

print("Training XGBoost model...")
model.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],  # Optional: for early stopping
    verbose=False                   # Set to True to see training progress
)

# Make predictions on validation set
print("Making predictions...")
y_pred = model.predict(X_valid)

# Calculate evaluation metrics
mse = mean_squared_error(y_valid, y_pred)
mae = mean_absolute_error(y_valid, y_pred)
rmse = np.sqrt(mse)

# Log progress
elapsed = time.time() - start_time

print("\n" + "="*50)
print("VALIDATION SET PERFORMANCE")
print("="*50)
print(f"Elapsed time train and validation: {elapsed} seconds")
print(f"Mean Squared Error (MSE):  {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print("="*50)
