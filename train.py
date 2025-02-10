import re
import os
import kagglehub
import pickle

import numpy as np
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime

class DropNaBodyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column="body"):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Ensure it's a DataFrame for easy manipulation
        if isinstance(X, pd.DataFrame):
            return X.dropna(subset=[self.column])
        else:
            raise ValueError("Input must be a pandas DataFrame")

class MissingValueIndicatorFiller(BaseEstimator, TransformerMixin):
    def __init__(self, fill_value="unknown"):
        self.fill_value = fill_value

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Transform the data by filling missing values and adding indicator columns.

        Args:
            X: Input data (Pandas DataFrame).

        Returns:
            Transformed DataFrame.
        """
        X = X.copy()
        
        for feature in X.columns:
            # Create the indicator column
            indicator_col = f"{feature}_is_known"
            X[indicator_col] = X[feature].notna().astype(int)  # 1 if known, 0 if unknown
            
            # Fill missing values
            X[feature] = X[feature].fillna(self.fill_value)
        
        return X
    
class URLImputer(BaseEstimator, TransformerMixin):
    def __init__(self, url_regex=None):
        self.url_regex = url_regex
        if self.url_regex is None:
            self.url_regex = r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)'
        self.url_finder = re.compile(self.url_regex)
        
    def fit(self, X, y=None):
        return self

    def _identify_url(self, row):
        if pd.isna(row['urls']):
            body = str(row['body'])
            is_url = bool(self.url_finder.search(body))
            return int(is_url)
        else:
            return row['urls']

    def transform(self, X, y=None):
        impute_X = X.copy()
        impute_X['urls'] = X.apply(self._identify_url, axis=1)
        # impute_X.drop("body", inplace=True, axis=1)

        return impute_X

class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, placeholder="unknown", date_format="%a, %d %b %Y %H:%M:%S %z"):
        self.placeholder = placeholder
        self.date_format = date_format

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # X is assumed to be a NumPy array (with columns: 'subject', 'sender', 'date', 'body', 'urls')
        
        # Extract the 'date' column (index of 'date' can vary, here it's assumed to be index 2)
        date_column_idx = 0
        date_column = X[:, date_column_idx]

        # Handle missing or placeholder values
        parsed_dates = np.vectorize(self._safe_parse_date)(date_column)

        # Extract features from parsed dates
        day_of_week = np.vectorize(lambda x: x.weekday() if x else 0)(parsed_dates)
        hour_of_day = np.vectorize(lambda x: x.hour if x else 0)(parsed_dates)
        is_weekend = (day_of_week >= 5).astype(int)  # 1 for weekend, 0 for weekday
        year = np.vectorize(lambda x: x.year if x else 1900)(parsed_dates)
        month = np.vectorize(lambda x: x.month if x else 1)(parsed_dates)
        day_of_month = np.vectorize(lambda x: x.day if x else 1)(parsed_dates)

        # Stack all new features together as columns (along with the rest of the data)
        X_new = np.column_stack([day_of_week, hour_of_day, is_weekend, year, month, day_of_month])
        
        # Return the transformed data (now with additional features)
        return X_new

    def _safe_parse_date(self, date_str):
        try:
            return datetime.strptime(date_str, self.date_format)
        except (ValueError, TypeError):
            return None  # Return None for invalid dates



def main():
    input_paths = []

    path = kagglehub.dataset_download("naserabdullahalam/phishing-email-dataset")

    stripped_datasets = ["Enron.csv", "Ling.csv", "phishing_email.csv"]
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            if filename in stripped_datasets:
                continue
            input_paths.append(os.path.join(dirname, filename))


    dataframes = []
    for path in input_paths:
        df = pd.read_csv(path)
        dataframes.append(df)

    df = pd.concat(dataframes, axis=0, ignore_index=True)
    df = df.dropna(subset=['body'])
    df.reset_index(drop=True, inplace=True)
    X = df.drop(["label"], axis=1)
    y = df["label"]

    data_preprocessing = ColumnTransformer([
        ("fill_na", MissingValueIndicatorFiller(), ["subject", "sender", "date"]),
        ("url", URLImputer(), ["urls", "body"])
    ])

    datapreprocessing_pipeline = Pipeline([
        ("handleNA", data_preprocessing)
    ])

    feature_extraction_pipeline = ColumnTransformer([
        ("subject_tfidf", TfidfVectorizer(), 0),  # Column index for 'subject'

        ("sender_tfidf", TfidfVectorizer(), 1),  # Column index for 'sender'

        ("body_tfidf", TfidfVectorizer(), 2)  # Column index for 'body'
    ], remainder="passthrough")  # Keep other columns unchanged

    feature_extraction_pipeline = ColumnTransformer([
        ("date_extraction", DateFeatureExtractor(), [2]),
        ("text_extraction", feature_extraction_pipeline, [0, 1, 7])
    ], remainder='passthrough')


    full_pipeline = Pipeline([
        ("preprocessing", datapreprocessing_pipeline),
        ("feature_extraction", feature_extraction_pipeline),
        ('predictor', XGBClassifier(random_state=42))
    ])

    full_pipeline.fit(X, y)
    pickle.dump(full_pipeline, open("full_pipeline.pkl", "wb"))
    print("Pipeline fitted and saved.")

if __name__ == "__main__":
    main()