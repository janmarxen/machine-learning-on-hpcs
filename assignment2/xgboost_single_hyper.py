import time
import itertools
import argparse
import random
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from mpi4py import MPI

#https://www.kaggle.com/competitions/home-data-for-ml-course/data?select=data_description.txt

# Modules to load in the ULHPC cluster
#module load data/scikit-learn
#module load vis/matplotlib
#module load bio/Seaborn/0.13.2-gfbf-2023b
#module load mpi/OpenMPI
#module load lib/mpi4py/3.1.5-gompi-2023b

# Load your virtual environment where XGboost is installed
#source xgboost_env/bin/activate

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Parse command-line arguments (rank 0 parses, then broadcast)
parser = argparse.ArgumentParser(description='XGBoost hyperparameter search with MPI')
parser.add_argument('--distribution', choices=['contiguous', 'roundrobin', 'shuffle'], default='roundrobin',
                    help='Work distribution strategy: contiguous (naive), roundrobin (A), or shuffle (B)')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for shuffle distribution')
args = parser.parse_args() if rank == 0 else None
args = comm.bcast(args, root=0)

# Start overall timing
start_total = time.time()

if rank == 0:
    print(f"Running with {size} MPI processes")
    print(f"Distribution strategy: {args.distribution}")
    if args.distribution == 'shuffle':
        print(f"Shuffle seed: {args.seed}")

# Read the data (all processes read the data)
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


# Your parameter grid
param_grid = {
    'n_estimators': [100, 200, 500, 750, 1000, 1500],
    'learning_rate': [0.001, 0.01, 0.05, 0.1],
    'max_depth': [6, 8, 10],
    'subsample': [0.6, 0.8],
    'colsample_bytree': [0.6, 0.8],
    'reg_lambda': [0, 0.5, 1, 2],
    'reg_alpha': [0, 0.5, 1, 2],
    
}

# Generate all combinations
param_combinations = list(itertools.product(
    param_grid['n_estimators'],
    param_grid['learning_rate'],
    param_grid['max_depth'],
    param_grid['subsample'],
    param_grid['colsample_bytree'],
    param_grid['reg_lambda'],
    param_grid['reg_alpha'],
))

total_combinations = len(param_combinations)
if rank == 0:
    print(f"Total parameter combinations to test: {total_combinations}")

# Distribute work across MPI processes
if args.distribution == 'shuffle':
    # Option B: Shuffle then chunk
    if rank == 0:
        rng = random.Random(args.seed)
        rng.shuffle(param_combinations)
    param_combinations = comm.bcast(param_combinations, root=0)
    
    combinations_per_rank = total_combinations // size
    start_idx = rank * combinations_per_rank
    end_idx = total_combinations if rank == size - 1 else start_idx + combinations_per_rank
    my_combinations = param_combinations[start_idx:end_idx]
elif args.distribution == 'contiguous':
    # Naive: Contiguous block assignment (original approach)
    combinations_per_rank = total_combinations // size
    start_idx = rank * combinations_per_rank
    end_idx = total_combinations if rank == size - 1 else start_idx + combinations_per_rank
    my_combinations = param_combinations[start_idx:end_idx]
else:
    # Option A: Round-robin (default)
    my_combinations = param_combinations[rank::size]

if rank == 0:
    print(f"Each process will handle approximately {len(my_combinations)} combinations")

# Store results for this process
local_results = []

for i, (n_estimators, learning_rate, max_depth, subsample, colsample_bytree, reg_lambda, reg_alpha) in enumerate(my_combinations):
    # Calculate global index based on distribution strategy
    if args.distribution == 'roundrobin':
        global_idx = rank + i * size
    else:
        # Both 'contiguous' and 'shuffle' use start_idx
        global_idx = start_idx + i
    
    print(f"[Rank {rank}] Testing combination {i+1}/{len(my_combinations)} (global: {global_idx+1}/{total_combinations}):")
    print(f"  n_estimators: {n_estimators}, lr: {learning_rate}, max_depth: {max_depth}, "
          f"subsample: {subsample}, colsample_bytree: {colsample_bytree}, "
          f"reg_lambda: {reg_lambda}, reg_alpha: {reg_alpha}")

    start_time = time.time()
    model = XGBRegressor(random_state=42)

    params = {
        'n_estimators': int(n_estimators),
        'learning_rate': learning_rate,
        'max_depth': int(max_depth),
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'reg_lambda': reg_lambda,
        'reg_alpha': reg_alpha,
    }
    model.set_params(**params)
    model.fit(X_train, y_train)

    # Make predictions on validation set
    y_pred = model.predict(X_valid)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_valid, y_pred)
    mae = mean_absolute_error(y_valid, y_pred)
    rmse = np.sqrt(mse)

    elapsed = time.time() - start_time

    print(f"[Rank {rank}] MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, Time: {elapsed:.2f}s")
    
    # Store result
    local_results.append({
        'n_estimators': n_estimators,
        'learning_rate': learning_rate,
        'max_depth': max_depth,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'reg_lambda': reg_lambda,
        'reg_alpha': reg_alpha,
        'mse': mse,
        'mae': mae
    })

# Gather all results to rank 0
all_results = comm.gather(local_results, root=0)

# Rank 0 processes and saves the results
if rank == 0:
    print("\n" + "="*50)
    print("GATHERING AND SAVING RESULTS")
    print("="*50)
    
    # Flatten the list of lists
    results_flat = [item for sublist in all_results for item in sublist]
    
    # Create DataFrame
    df_results = pd.DataFrame(results_flat)
    
    # Sort by MSE
    df_results = df_results.sort_values('mse')
    
    # Save to CSV
    output_filename = 'hyperparameter_search_results.csv'
    df_results.to_csv(output_filename, index=False)
    
    print(f"\nResults saved to: {output_filename}")
    print(f"Total experiments: {len(df_results)}")
    print("\nTop 5 results by MSE:")
    print(df_results.head().to_string(index=False))
    print("="*50)
    
    # Print total execution time
    total_time = time.time() - start_total
    print(f"\nTotal execution time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print("="*50)

