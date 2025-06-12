import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

pca = joblib.load(r'Task-1\pca_model.pkl')
small_data = pd.read_csv(r'Task-1\small_data.csv')


# we separate the entries with emotion 6
emotion_6_data = small_data[small_data['emotion'] == 6]

# selecting 47 random entries from emotion 6 data
random_emotion_6 = emotion_6_data.sample(n=47, random_state=0)

# getting the data where emotion is not 6
rest_of_data = small_data[small_data['emotion'] != 6]

# we combine the randomly selected emotion 6 data with the rest of the data
small_data_2 = pd.concat([random_emotion_6, rest_of_data])



small_data_2_dropped = small_data_2.drop(columns=['emotion'])

# we use our pre-trained pca model to transform small_data_2 into a low dimensionality space
small_data_2_pca = pca.transform(small_data_2_dropped)

# we name our features PC1, PC2, and so on...
small_data_2_pca_df = pd.DataFrame(small_data_2_pca, columns=[f'PC{i+1}' for i in range(small_data_2_pca.shape[1])])




X_pca_2 = small_data_2_pca_df
y_pca_2 = small_data_2['emotion']

X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_pca_2, y_pca_2, test_size=0.3, random_state=0, stratify=y_pca_2)

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

grid_search.fit(X_train_2, y_train_2)

print("Best parameters:", grid_search.best_params_)




# we choose the best choices for the hyperparameters, and use them in our SVM model
best_svm = grid_search.best_estimator_

y_pred_2 = best_svm.predict(X_test_2)

print("Classification Report:")
print(classification_report(y_test_2, y_pred_2))





cm = confusion_matrix(y_test_2, y_pred_2)
display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_pred_2))
display.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()