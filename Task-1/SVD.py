# Both of our data has 2304 features. Using all of them will be computationally  
# expensive. Instead we can try to use the most significant features only.

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

large_data = pd.read_csv('Task-1\large_data.csv')

X = large_data - large_data.mean()
U, S, VT = np.linalg.svd(X, full_matrices=False)

# S is a diagonal matrix. The diagonal elements in S give us the proportion in
# which each component explains the variance of the original data. For convenience
# and interpretability we transform the proportions into ratios.
S = (S/np.sum(S))*100

S_df = pd.DataFrame({
    'Component': [f'S{i+1}' for i in range(len(S))],
    'Percent Explained Variance': np.round(S, 2)
})

# PLOTTING S_df
# We will find out the distribution of the variance explained by each component in S_df.  
# This will help us in deducing which "features" are the most important in describing the images in large_data.  
# From here, we can reduce the dimensionality from 2304 to a significantly smaller number.


sns.lineplot(data=S_df, x='Component', y='Percent Explained Variance')
plt.title("Percentage of Variance explained by each Component")
plt.xticks(ticks=range(0, 2304, 250))
plt.ylabel("Explained Variance (in %)")
plt.show()


# The elbow point of the distribution lies somewhere between 80 and 100.  
# Instead of using all of the 2304 features, we will use only 100, reducing our  
# load by more than 95%.