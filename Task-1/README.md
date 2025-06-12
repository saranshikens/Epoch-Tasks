# Task-1

Introduction:

Problem Statement :
Design, implement, and evaluate a machine learning pipeline that leverages SVM's, PCA, and SVD to classify facial expressions, utilizing both the small labeled dataset and the large unlabeled dataset to maximize the model's performance.

My Approach :
The labelled data is highly unbalanced with more than 95% of the entries belonging to emotion-6. 
To account for this disbalance, I have used two approaches:
Approach-1 : Simply remove all of the emotion-6 entries from the labelled data.
Approach-2 : Ignoring emotion-6, each emotion has on average 47 entries. Out of the 593 entries for emotion-6, I have randomly selected 47 entries.


Methodoloy :

Data Preprocessing :
In the labelled data, all the pixels had been stored in a single string. So I split each string, and extracted the pixels, storing them as individual features. At the end, each image was stored as a vector of size 2304.
The images in unlabelled data were stored in separate folders, so I had to exhaustively search for all .jpg files. On encountering such a file, I grayscaled , resized to 48x48 (size of each image in labelled data), and reshaped them to a vector of size 2304 (same as labelled data).

PCA Feature Engineering :
Using all of the 2304 features would be computationally expensive, so I used SVD to deduce which features are the most important. I plotted the explained variance of each component, and by roughly judging the elbow point in the graph, I deduced that 100 features contain most of the information.
To determine those 100 features, I used PCA with n_components=100.

SVM Classification :
To choose the optimal svm kernel, and hyperparameters, I stored them in a grid (dictionary), and using cross validation, determined the accuracy of each set of kernel and hyperparameters. The one with the highest accuracy was chosen. I tried out different values for 'C' and 'gamma'. At the end, linear kernel with C = 0.001 was chosen.


Results :
Approach-1 :

Classification Report:
              precision    recall  f1-score   support

           0       0.53      0.57      0.55        14
           1       0.80      0.89      0.84        18
           2       0.62      0.62      0.62         8
           3       1.00      1.00      1.00        21
           4       0.22      0.25      0.24         8
           5       0.95      0.84      0.89        25
           7       0.50      0.40      0.44         5

    accuracy                           0.76        99
   macro avg       0.66      0.65      0.66        99
weighted avg       0.77      0.76      0.76        99

![Confusion Matrix of Approach-1](Task-1\Images\Approach-1.png)

The model performs well for emotion-1,3, and 5, and performs very poorly for emotion-4.


Approach-2 (One Iteration):

Classification Report:
              precision    recall  f1-score   support

           0       0.54      0.50      0.52        14
           1       0.74      0.94      0.83        18
           2       0.57      0.50      0.53         8
           3       0.95      1.00      0.98        21
           4       0.14      0.12      0.13         8
           5       0.95      0.80      0.87        25
           6       0.57      0.57      0.57        14
           7       0.50      0.60      0.55         5

    accuracy                           0.72       113
   macro avg       0.62      0.63      0.62       113
weighted avg       0.72      0.72      0.71       113

![Confusion Matrix of Approach-2 with One Iteration](Task-1\Images\Approach-2_One_Itr.png)

The model improves its recall and f-1 score in emotion-7, and continues to show the same trends as Approach-1.
Including emotion-6, improved the results for emotion-7, this could mean that both emotions have some correlation.
Removing emotion-6 from our data lead to loss of important information about emotion-7. As emotion-6 denotes a 'neutral'
emotion, it could be possible that some of them are similar to 'contempt', i.e. emotion-7.

Approach-2 (100 Iterations)

Classification Report:
              precision    recall  f1-score   support

           0       0.53      0.54      0.53        14
           1       0.86      0.99      0.92        18
           2       0.66      0.53      0.58         8
           3       0.95      1.00      0.98        21
           4       0.24      0.21      0.23         8
           5       0.96      0.87      0.91        25
           6       0.60      0.53      0.55        14
           7       0.40      0.55      0.46         5

Average Accuracy: 0.7462
![Confusion Matrix of Approach-2 with One Iteration](Task-1\Images\Approach-2_100_Itr.png)

The model performs better than the one in 1 iteration in almost all fields, except emotion-7. Still, the model's performance remains poor in emotion 4.


Discussion and Insights
Only 100 pixels were significant out of the 2304 pixels. This means that to determine the emotion of a face, we need not analyze the entire face. Analyzing a certain small portion (my guess - eyes, eyebrows and lips) is enough to deduce emotions.
The strategy worked great for emotions 1,3 and 5, while it was quite poor for emotion-4. After emotion-4, it is the emotion of 'contempt' that performs poor. This could be due to the complexity of this emotion. Contempt is the emotion of feeling superior to someone, its signs are not as pronounced as other emotions.
The main challenge was to account for the disbalance in the labelled data, that I tackled using the previously mentioned approaches.
Since my approach relies on iterations, main limitation is computation power and time. As we keep on increasing the iterations are results will keep on getting better. Since PCA retains directions of maximum variance, if an emotion has low variance, it could get under represented in the principal components.
The performance of this model on emotion-4 is worrisome. My next steps will be to deeply analysize the distribution of the data for emotion-4. Images belonging to emotion-4 can be augmented so that the model captures this emotion more clearly. Increasing the number of iterations can be also be tried.
