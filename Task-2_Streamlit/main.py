import numpy as np
import pandas as pd
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import joblib

df = pd.read_csv(r"Task-2_Streamlit\Files\UpdatedResumeDataSet.csv")



def clean_resume(resume):
    # Remove '\r' and '\n'
    cleaned_resume = resume.replace('\r', '').replace('\n', ' ')
    # Remove characters not in the allowed set (alphanumeric and whitespace)
    cleaned_resume = re.sub(r'[^a-zA-Z\s]', '', cleaned_resume)
    return cleaned_resume

df['Cleaned_Resume'] = df['Resume'].apply(clean_resume)
df['Resume'] = df['Cleaned_Resume']
df = df.drop('Cleaned_Resume', axis=1)



nlp = spacy.load('en_core_web_sm')
df['Tokenized_Resume'] = df['Resume'].apply(lambda x: [token.text for token in nlp(x)])

def extract_sections(resume_tokens):
    sections = {"Skills": [], "Work Experience": [], "Projects": []}
    current_section = None
    section_keywords = {
        ("skill", "skills"): "Skills", # we account for both lower case and camel case
        ("experience", "work experience"): "Work Experience",
        ("project", "projects"): "Projects"
    }

    # Join tokens back into a string for easier pattern matching
    resume_text = " ".join(resume_tokens).lower()

    #  we use regular expressions to find sections based on keywords
    for keywords, section_name in section_keywords.items():
        pattern = r"({})".format("|".join(keywords))
        matches = list(re.finditer(pattern, resume_text))

        if matches:
            # assuming the text after the first match of a keyword belongs to that section
            start_index = matches[0].end()
            # finding the start of the next potential section to define the end of the current one
            next_section_starts = [re.search(other_pattern, resume_text[start_index:]) for other_keywords, other_pattern in section_keywords.items() for other_pattern in [r"({})".format("|".join(other_keywords))]]
            next_section_starts = [match.start() + start_index for match in next_section_starts if match]

            end_index = min(next_section_starts) if next_section_starts else len(resume_text)

            sections[section_name] = [resume_text[start_index:end_index].strip()]

    return sections

df['Extracted_Features'] = df['Tokenized_Resume'].apply(extract_sections)



# Concatenate extracted sections for each resume
df['Concatenated_Features'] = df['Extracted_Features'].apply(lambda x: ' '.join([' '.join(section) for section in x.values()]))

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

# Fit and transform the concatenated features
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Concatenated_Features'])

# Convert to DataFrame for easier inspection (optional)
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())



X = tfidf_df - tfidf_df.mean()
U, S, VT = np.linalg.svd(X, full_matrices=False)

# S is a diagonal matrix. The diagonal elements in S give us the proportion in
# which each component explains the variance of the original data. For convenience
# and interpretability we transform the proportions into ratios.
S = (S/np.sum(S))*100



scaler = StandardScaler()
scaled_tfidf_df = scaler.fit_transform(tfidf_df)

pca = PCA(n_components=175)
tfidf_pca = pca.fit_transform(scaled_tfidf_df)

tfidf_pca_df = pd.DataFrame(tfidf_pca, columns=[f'PC{i+1}' for i in range(tfidf_pca.shape[1])])



X_temp, X_test, y_temp, y_test = train_test_split(tfidf_pca_df, df['Category'], test_size=0.3, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42) 



model = keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=[tfidf_pca_df.shape[1]]),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(df['Category'].nunique(), activation='softmax'),
])



# One-hot encode the target variable
encoder = OneHotEncoder(sparse_output=False)
y_train_encoded = encoder.fit_transform(y_train.values.reshape(-1, 1))
y_val_encoded = encoder.transform(y_val.values.reshape(-1, 1))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# if the accuract drops continuously for more than 10 epochs, we will stop training
# accuracy is said to change if it differs by atleast 0.001
early_stopping = keras.callbacks.EarlyStopping(
    patience=10,
    min_delta=0.001,
    restore_best_weights=True,
)

history = model.fit(
    X_train, y_train_encoded,
    validation_data=(X_val, y_val_encoded),
    batch_size=256,
    epochs=100,
    callbacks=[early_stopping],
    verbose=0,
)




model.save("model.keras")
joblib.dump(encoder, 'encoder.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(pca, 'pca.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')


