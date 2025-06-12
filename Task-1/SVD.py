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



sns.lineplot(data=S_df, x='Component', y='Percent Explained Variance')
plt.title("Percentage of Variance explained by each Component")
plt.xticks(ticks=range(0, 2304, 250))
plt.ylabel("Explained Variance (in %)")
plt.show()