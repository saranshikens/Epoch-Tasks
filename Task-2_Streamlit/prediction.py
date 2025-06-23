import joblib
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

########################
pca = joblib.load(r"Task-2_Streamlit\Model_Files\pca.pkl")
scaler = joblib.load(r"Task-2_Streamlit\Model_Files\scaler.pkl")
tfidf_vectorizer = joblib.load(r"Task-2_Streamlit\Model_Files\tfidf_vectorizer.pkl")
encoder = joblib.load(r"Task-2_Streamlit\Model_Files\encoder.pkl")
model = keras.models.load_model(r"Task-2_Streamlit\Model_Files\model.keras")
########################

class Prediction:
    def __init__(self, resume, category):
        self.resume = resume
        self.category = category

    @staticmethod
    def clean_resume(resume):
        # Remove '\r' and '\n'
        cleaned_resume = resume.replace('\r', '').replace('\n', ' ')
        # Remove characters that are not allowed
        cleaned_resume = re.sub(r'[^a-zA-Z\s]', '', cleaned_resume)
        return cleaned_resume

    def extract_sections(self, resume_tokens):
        sections = {"Skills": [], "Work Experience": [], "Projects": []}
        current_section = None
        section_keywords = {
            ("skill", "skills"): "Skills",
            ("experience", "work experience"): "Work Experience",
            ("project", "projects"): "Projects"
        }

        resume_text = " ".join(resume_tokens).lower()

        for keywords, section_name in section_keywords.items():
            pattern = r"({})".format("|".join(keywords))
            matches = list(re.finditer(pattern, resume_text))

            if matches:
                start_index = matches[0].end()
                next_section_starts = [
                    re.search(other_pattern, resume_text[start_index:])
                    for other_keywords in section_keywords.keys()
                    for other_pattern in [r"({})".format("|".join(other_keywords))]
                ]
                next_section_starts = [match.start() + start_index for match in next_section_starts if match]
                end_index = min(next_section_starts) if next_section_starts else len(resume_text)
                sections[section_name] = [resume_text[start_index:end_index].strip()]

        return sections


    def tokenization(self, df):
        tfidf_matrix = tfidf_vectorizer.fit_transform(df['Concatenated_Features'])

        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
        return tfidf_df

    def prediction(self):
        df_user = pd.DataFrame({
            'Category': [self.category],
            'Resume': [self.resume]
        })
        
        nlp = spacy.load('en_core_web_sm')

        df_user['Cleaned_Resume'] = df_user['Resume'].apply(self.clean_resume)
        df_user['Resume'] = df_user['Cleaned_Resume']
        df_user = df_user.drop('Cleaned_Resume', axis=1)
        df_user['Tokenized_Resume'] = df_user['Resume'].apply(lambda x: [token.text for token in nlp(x)])
        df_user['Extracted_Features'] = df_user['Tokenized_Resume'].apply(self.extract_sections)
        df_user['Concatenated_Features'] = df_user['Extracted_Features'].apply(lambda x: ' '.join([' '.join(section) for section in x.values()]))

        tfidf_matrix_user = tfidf_vectorizer.transform(df_user['Concatenated_Features'])
        features_seen = tfidf_vectorizer.get_feature_names_out()
        tfidf_df_user = pd.DataFrame(tfidf_matrix_user.toarray(), columns=features_seen)

        tfidf_df_user = tfidf_df_user.reindex(columns=features_seen, fill_value=0)
        scaled_tfidf_df_user = scaler.transform(tfidf_df_user)
        tfidf_pca_user = pca.transform(scaled_tfidf_df_user)
        tfidf_pca_df_user = pd.DataFrame(tfidf_pca_user, columns=[f'PC{i+1}' for i in range(tfidf_pca_user.shape[1])])
        category_pred_encoded = model.predict(tfidf_pca_df_user)
        category_pred_decoded = encoder.inverse_transform(category_pred_encoded)

        return category_pred_decoded[0][0]
