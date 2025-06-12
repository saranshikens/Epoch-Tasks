import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

pca = joblib.load(r'Task-1\pca_model.pkl')
small_data = pd.read_csv(r'Task-1\small_data.csv')


# remove the entries for which emotion=6
small_data_1 = small_data[small_data['emotion']!=6]

small_data_1 = small_data_1.drop(columns=['emotion'])

# we use our pre-trained pca model to transform small_data_1 into a low dimensionality space
small_data_1_pca = pca.transform(small_data_1)

# we name our features PC1, PC2, and so on...
small_data_1_pca_df = pd.DataFrame(small_data_1_pca, columns=[f'PC{i+1}' for i in range(small_data_1_pca.shape[1])])



X_pca_1 = small_data_1_pca_df

# in accordance with approach-1, we remove the entries where emotion=6
y_pca_1 = small_data.loc[small_data['emotion'] != 6, 'emotion']

X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_pca_1, y_pca_1, test_size=0.3, random_state=0, stratify=y_pca_1)

svm = SVC()

# we use different kernels for SVM, and use the best one out of them
hyper_param_grid = {
    'kernel': ['linear', 'rbf', 'poly'],
    'C': [0.001, 0.1, 0.1],
    'gamma': ['scale', 0.01, 0.001]
}

# cross-validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

# search through the grid of hyper parameters, and use cross validation to return the best set out of them
grid_search = GridSearchCV(svm, hyper_param_grid, cv=cv, scoring='accuracy', n_jobs=-1)

grid_search.fit(X_train_1, y_train_1)

print("Best parameters:", grid_search.best_params_)





# we choose the best choices for the hyperparameters, and use them in our SVM model
best_svm = grid_search.best_estimator_

y_pred_1 = best_svm.predict(X_test_1)

print("Classification Report:")
print(classification_report(y_test_1, y_pred_1))



cm = confusion_matrix(y_test_1, y_pred_1)
display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_pred_1))
display.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()