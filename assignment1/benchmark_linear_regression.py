from linear_regression_complete import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
import csv
import math
import gc


def run_benchmarks(out_csv='benchmark_results.csv', out_png='benchmark_elapsed_time.png'):
	"""Run timing benchmarks for permutations of n_samples and n_features.

	Saves a CSV with columns: n_samples,n_features,elapsed_seconds,success,exception
	and a heatmap PNG of elapsed times (rows: n_samples, cols: n_features).
	"""

	n_samples_list = [10000, 100000, 500000, 1000000]
	n_features_list = [10, 50, 100, 1000]

	results = []

	print("Preparing grid for heatmap...")
	# Prepare grid for heatmap (rows = samples, cols = features)
	elapsed_grid = np.full((len(n_samples_list), len(n_features_list)), np.nan, dtype=float)

	print("Starting benchmarks...\n")
	for i, n_samples in enumerate(n_samples_list):
		for j, n_features in enumerate(n_features_list):
			print(f"Running benchmark: n_samples={n_samples}, n_features={n_features}")
			gc.collect()
			start_time = None
			elapsed = None
			# Create synthetic data (use float32 to reduce memory)
			X = (np.random.randn(n_samples, n_features) * 5).astype(np.float32)
			# create coefficients and target
			true_coeffs = np.random.randn(n_features).astype(np.float32)
			true_intercept = np.float32(2.0)
			y = (true_intercept + X.dot(true_coeffs) + (np.random.randn(n_samples) * 1.5).astype(np.float32)).astype(np.float32)

			model = LinearRegression()

			start_time = time.perf_counter()
			model.fit(X, y)
			elapsed = time.perf_counter() - start_time

			# free memory used by X,y,model components
			del X, y, model, true_coeffs
			gc.collect()

			results.append({
				'n_samples': n_samples,
				'n_features': n_features,
				'elapsed_seconds': elapsed if elapsed is not None else math.nan
			})
			print(f" n_samples: {n_samples}, n_features: {n_features}: {elapsed:.4f} seconds \n")

			# store in grid for plotting
			elapsed_grid[i, j] = float(elapsed)

			# small pause
			time.sleep(0.5)

	# Write CSV
	with open(out_csv, 'w', newline='') as csvfile:
		fieldnames = ['n_samples', 'n_features', 'elapsed_seconds']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()
		for r in results:
			writer.writerow(r)

	# Plot heatmap
	try:
		fig, ax = plt.subplots(figsize=(10, 6))
		ma = np.ma
		masked = ma.masked_invalid(elapsed_grid)
		sns.heatmap(masked, annot=True, fmt='.4f', cmap='viridis', ax=ax,
					xticklabels=n_features_list, yticklabels=n_samples_list)
		ax.set_xlabel('n_features')
		ax.set_ylabel('n_samples')
		ax.set_title('Elapsed time (seconds) for model.fit')
		plt.tight_layout()
		plt.savefig(out_png, dpi=300, bbox_inches='tight')
		plt.close()
	except Exception as e:
		print(f"Failed to create heatmap: {e}")



if __name__ == '__main__':
	run_benchmarks()
