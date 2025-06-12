import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

pca = joblib.load(r'Task-1\pca_model.pkl')
small_data = pd.read_csv(r'Task-1\small_data.csv')
scaler = joblib.load(r'Task-1\scaler.pkl')


# INITIALIZING THE EVALUATION METRICS

# we will store results from each iteration in these lists and dictionaries
accuracy_scores = []
confusion_matrices = []

# since we have to output the metrics for each emotion class, we initialize the keys
# as the emotions themselves
precision_scores = {str(i): [] for i in range(7)} 
recall_scores = {str(i): [] for i in range(7)}
f1_scores = {str(i): [] for i in range(7)}

n_iterations = 100


# RUNNING THE ITERATIONS


for i in range(n_iterations):
    if (i+1)%10==0:
        print(f"Iteration {i+1}/{n_iterations}")

    # we separate the entries with emotion 6
    emotion_6_data = small_data[small_data['emotion'] == 6]

    # selecting 47 random entries from emotion 6 data
    # we will use a different random state for each iteration
    random_emotion_6 = emotion_6_data.sample(n=47, random_state=i)

    # geting the data where emotion is not 6
    rest_of_data = small_data[small_data['emotion'] != 6]

    # we combine the randomly selected emotion 6 data with the rest of the data
    small_data_2 = pd.concat([random_emotion_6, rest_of_data])

    # Apply PCA transformation
    small_data_2_dropped = small_data_2.drop(columns=['emotion'])
    
    # we use our pre-trained pca model to transform small_data_2 into a low dimensionality space
    small_data_2_pca = pca.transform(scaler.transform(small_data_2_dropped))

    # we name our features PC1, PC2, and so on...
    small_data_2_pca_df = pd.DataFrame(small_data_2_pca, columns=[f'PC{j+1}' for j in range(small_data_2_pca.shape[1])])

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

    best_svm = grid_search.best_estimator_

    y_pred_2 = best_svm.predict(X_test_2)

    # we will calculate all the evaluation metrics and store them in the lists
    report = classification_report(y_test_2, y_pred_2, output_dict=True)
    accuracy_scores.append(report['accuracy'])
    for label in precision_scores.keys():
        precision_scores[label].append(report[label]['precision'])
        

    for label in recall_scores.keys():
         recall_scores[label].append(report[label]['recall'])
        

    for label in f1_scores.keys():
        f1_scores[label].append(report[label]['f1-score'])
         
    # to determining the overall confusion matrix, we sum up all the confusion matrices
    if i == 0:
        overall_confusion_matrix = confusion_matrix(y_test_2, y_pred_2, labels=np.unique(y_pca_2))
    else:
        overall_confusion_matrix += confusion_matrix(y_test_2, y_pred_2, labels=np.unique(y_pca_2))


# AVERAGING OUT THE METRICS


average_confusion_matrix = overall_confusion_matrix / n_iterations
average_accuracy = np.mean(accuracy_scores)
average_precision = {label: np.mean(precision_scores[label]) for label in precision_scores.keys()}
average_recall = {label: np.mean(recall_scores[label]) for label in recall_scores.keys()}
average_f1 = {label: np.mean(f1_scores[label]) for label in f1_scores.keys()}



# AVERAGE CLASSIFICATION REPORT


print(f"\nAverage Classification Report over {n_iterations} iterations:")
print(f"Average Accuracy: {average_accuracy:.4f}")
print("Average Precision:")
for label, avg_prec in average_precision.items():
    print(f"  {label}: {avg_prec:.4f}")
print("Average Recall:")
for label, avg_rec in average_recall.items():
    print(f"  {label}: {avg_rec:.4f}")
print("Average F1-score:")
for label, avg_f1_score in average_f1.items():
    print(f"  {label}: {avg_f1_score:.4f}")



# AVERAGE CONFUSION MATRIX


display = ConfusionMatrixDisplay(confusion_matrix=average_confusion_matrix, display_labels=np.unique(y_pca_2))
display.plot(cmap='Blues')
plt.title(f"Average Confusion Matrix over {n_iterations} Iterations")
plt.show()





