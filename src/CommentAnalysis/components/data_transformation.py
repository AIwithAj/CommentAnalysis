import nltk
import sys
from src.CommentAnalysis import logger
from src.CommentAnalysis.utils.common import read_yaml,preprocess_comment,read_data
from src.CommentAnalysis.constants import SCHEMA_FILE_PATH
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

import numpy as np
import scipy.sparse
nltk.download('wordnet')
import pandas as pd
nltk.download('stopwords')
class DataTransformation:
    def __init__(self,data_transformation_config):
        try:
            self.data_transformation_config=data_transformation_config
            self._schema_config = read_yaml(SCHEMA_FILE_PATH)
        except Exception as e:
            raise Exception(e, sys)
    
 
    def normalize_text(self,df):
        """Apply preprocessing to the text data in the dataframe."""
        try:
            df['clean_comment'] = df['clean_comment'].apply(preprocess_comment)
            df.dropna(inplace=True)
            logger.debug('Text normalization completed')
            return df
        except Exception as e:
            logger.error(f"Error during text normalization: {e}")
            raise
    def apply_tfidf(self,train_data: pd.DataFrame, max_features: int, ngram_range: tuple) -> tuple:
        """Apply TF-IDF with ngrams to the data."""
        try:
            vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)

            X_train = train_data['clean_comment'].values
            y_train = train_data['category'].values

            # Perform TF-IDF transformation
            X_train_tfidf = vectorizer.fit_transform(X_train)
            
            logger.debug(f"TF-IDF transformation complete. Train shape: {X_train_tfidf.shape}")

            # Save the vectorizer in the root directory
            with open(self.data_transformation_config.transformer, 'wb') as f:
                pickle.dump(vectorizer, f)

            logger.debug('TF-IDF applied with trigrams and data transformed')
            return X_train_tfidf, y_train
        except Exception as e:
            logger.error('Error during TF-IDF transformation: %s', e)
            raise


    def initiate_data_Transformation(self):
        logger.debug("Starting data preprocessing...")
        train_data = read_data(self.data_transformation_config.train_file_path)
        test_data = read_data(self.data_transformation_config.test_file_path)
        logger.debug('Data loaded successfully')

        train_processed_data = self.normalize_text(train_data)
        test_processed_data = self.normalize_text(test_data)

        transform_train_path = self.data_transformation_config.transform_train_file
        transform_test_path = self.data_transformation_config.transform_test_file

        train_processed_data.to_csv(transform_train_path, index=False)
        test_processed_data.to_csv(transform_test_path, index=False)

        X_train_tfidf, y_train = self.apply_tfidf(
            train_processed_data,
            max_features=self.data_transformation_config.max_features,
            ngram_range=self.data_transformation_config.ngram_range
        )

        # save X_train_tfidf (csr_matrix) properly
        scipy.sparse.save_npz(self.data_transformation_config.x_train_file_path.replace(".csv", ".npz"), X_train_tfidf)
        logger.info(f"X_train TF-IDF matrix saved as sparse npz at: {self.data_transformation_config.x_train_file_path.replace('.csv', '.npz')}")

        # save y_train as csv
        pd.Series(y_train).to_csv(self.data_transformation_config.y_train_file_path, index=False)

        logger.info(f"Transformed Train file saved to: {transform_train_path}")
        logger.info(f"Transform Test file saved to: {transform_test_path}")
        logger.info(f"y Train file saved to: {self.data_transformation_config.y_train_file_path}")
        logger.info("****************************************************")


            


