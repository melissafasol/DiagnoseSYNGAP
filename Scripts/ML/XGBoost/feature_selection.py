import os
import numpy as np
import pandas as pd
from sklearn.utils import resample
from xgboost import XGBClassifier
from BorutaShap import BorutaShap
import matplotlib.pyplot as plt

# Load the filtered dataset
file_path = '/home/melissa/PROJECT_DIRECTORIES/EEGFeatureExtraction/Results/'
filtered_dataset = pd.read_csv(os.path.join(file_path, 'correlation_matrix_filtered.csv'))

# Extract directory for saving filtered datasets
output_dir = os.path.dirname(file_path)

# Prepare dataset for BorutaShap
boruta_train = filtered_dataset.drop(columns=['Epoch', 'Animal_ID', 'Br_State'])
X = boruta_train.iloc[:, 1:]
y = boruta_train.iloc[:, 0]
animal_ids = filtered_dataset['Animal_ID']

# Bootstrapping parameters
n_bootstraps = 10  # Number of bootstrap iterations
feature_importance_threshold = 0.75  # Threshold to retain features (e.g., 75% of iterations)

# Store selected features across bootstraps
feature_counts = pd.Series(dtype=int)

# Perform bootstrapping
for i in range(n_bootstraps):
    print(f"Bootstrap Iteration {i+1}/{n_bootstraps}")
    
    # Resample the data
    X_resampled, y_resampled = resample(X, y, random_state=i)
    
    # Fit BorutaShap
    estimator_borutashap = XGBClassifier(n_jobs=-1, random_state=42, max_depth=4)
    feature_selector = BorutaShap(
        model=estimator_borutashap,
        importance_measure='shap',
        classification=True
    )
    
    feature_selector.fit(X=X_resampled, y=y_resampled, n_trials=200, random_state=0)
    
    # Failsafe for plotting
    try:
        fig = feature_selector.plot(which_features='all', figsize=(24, 12))
        
        # Save the figure if plotting is successful
        plot_path = os.path.join(output_dir, f"bootstrap_{i+1}_feature_importance.png")
        fig.savefig(plot_path)
        plt.close(fig)
        print(f"Feature importance plot saved for bootstrap {i+1} at: {plot_path}")
    except Exception as e:
        print(f"Failed to plot feature importance for bootstrap {i+1}: {e}")
    
    # Get selected features for this iteration
    selected_features = feature_selector.Subset().columns
    print(f"Selected features in bootstrap {i+1}: {list(selected_features)}")
    
    # Update feature counts
    feature_counts = feature_counts.add(pd.Series(1, index=selected_features), fill_value=0)

# Aggregate results
total_iterations = n_bootstraps
feature_counts = feature_counts.sort_values(ascending=False)
important_features = feature_counts[feature_counts / total_iterations >= feature_importance_threshold]
print(f"Features retained (≥{feature_importance_threshold * 100}% importance):")
print(important_features)

# Save the aggregated important features
important_features_path = os.path.join(output_dir, "important_features_boruta_shap.csv")
important_features.to_csv(important_features_path, header=["Frequency"], index=True)
print(f"Important features saved at: {important_features_path}")

# Optional: Save feature counts across all iterations
feature_counts_path = os.path.join(output_dir, "all_iterations_feature_counts_boruta.csv")
feature_counts.to_csv(feature_counts_path, header=["Frequency"], index=True)
print(f"Feature counts across all iterations saved at: {feature_counts_path}")