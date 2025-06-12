import numpy as np
import pandas as pd
import os 


small_data = pd.read_csv(r"Task-1\Data Sets\ckextended.csv")
print(small_data.head())


# The way the pixels have been stored is problematic.  
# Each image has been reshaped to a vector, with all pixels stored in a single string.  
# For our purpose, each pixel has to be a separate feature, i.e. we need to store  
# each pixel as an individual feature and as an integer, not a string.


# 'Usage' serves no "use" for the task at hand
small_data = small_data.drop(columns=['Usage'])

# splits each string entry in small_data['pixels'], and returns a list of individual pixel values for each entry
small_data['pixels'] = small_data['pixels'].apply(lambda x: list(map(int, x.split())))

# each "pixel" in small_data['pixels'] needs to be stored as a separate feature, since we want to describe an image as a vector
pixels_df = pd.DataFrame(small_data['pixels'].tolist())

# the original string of pixels is no longer needed
small_data = small_data.drop(columns=['pixels'])
small_data = pd.concat([small_data, pixels_df], axis=1)
print(small_data.head())


# To gain more insights about the distribution of the emotions, we count the frequency of each emotion.
emotion_counts = small_data['emotion'].value_counts()
print(emotion_counts)


# Above table reveals a problem with the dataset.  
# It is very imbalanced, with emotion '6' having more than 60% of the entries.  
# This will give an unfair weightage to emotion '6' when we try to fit a model to  
# this data, and try to access its accuracy and other metrics.  
# Basically, if our model works only for emotion '6', even then its accuracy will be higher than it realistically should be.


# We can take care of this using two approaches:

# Approach-1: Dropping Emotion'6'  
# We can ignore the entries that are causing trouble.  

# Approach-2: Randomly selecting entries for emotion '6'  
# Ignoring emotion '6', each emotion has on an average 47 entries.  
# We may randomly select 47 entries for emotion '6', fit our model, repeat the process a certain number of times, and average out the results. 

small_data.to_csv('small_data.csv', index=False)