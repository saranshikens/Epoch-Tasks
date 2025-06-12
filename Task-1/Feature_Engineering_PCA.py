# With Principal Component Analysis on the Unlabelled Data, we will extract the  
# before mentioned 100 features.

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib

large_data = pd.read_csv(r'Task-1\large_data.csv')

scaler = StandardScaler()
scaled_large_data = scaler.fit_transform(large_data)

pca = PCA(n_components=100)
large_pca_data = pca.fit_transform(scaled_large_data)

large_pca_data_df = pd.DataFrame(large_pca_data, columns=[f'PC{i+1}' for i in range(large_pca_data.shape[1])])

# After fitting PCA
joblib.dump(pca, r'Task-1\pca_model.pkl')
joblib.dump(scaler, r'Task-1\scaler.pkl')
