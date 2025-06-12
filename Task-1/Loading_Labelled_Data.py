import numpy as np
import pandas as pd
import os 

small_data = pd.read_csv(r"Task-1\Data Sets\ckextended.csv")
small_data.head()

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

emotion_counts = small_data['emotion'].value_counts()
print(emotion_counts)

small_data.to_csv('small_data.csv', index=False)