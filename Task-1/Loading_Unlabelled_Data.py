import numpy as np
import pandas as pd
import os
from PIL import Image

lfw_dir = r"Task-1\Data Sets\lfw-deepfunneled"

image_vectors = [] # we will store each image here, and later use it to make a pandas dataframe for the unlabelled data

for person_name in os.listdir(lfw_dir): # os.listdir(lfw_dir) returns a list of all files and folders in lfw_dir
    person_dir = os.path.join(lfw_dir, person_name) # we obtain the path of each folder by joining its name with lfw_dir
    if os.path.isdir(person_dir):
        for image_filename in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_filename)

            # Open the image
            img = Image.open(image_path)

            # Convert to grayscale
            img_gray = img.convert('L')

            # Resize to 48x48
            img_resized = img_gray.resize((48, 48))

            # Convert to numpy array and flatten into a 1x2304 vector
            img_array = np.array(img_resized)
            img_vector = img_array.flatten().tolist()

            image_vectors.append(img_vector)


large_data = pd.DataFrame(image_vectors)
print(large_data.head())

large_data.to_csv('large_data.csv', index=False)