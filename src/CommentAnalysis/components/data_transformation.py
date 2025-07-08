import nltk
import sys
from src.CommentAnalysis import logger
from src.CommentAnalysis.utils.common import preprocess_comment,read_data
nltk.download('wordnet')
nltk.download('stopwords')
class DataTransformation:
    def __init__(self,data_transformation_config):
        try:
            self.data_transformation_config=data_transformation_config
            # self._schema_config = read_yaml(SCHEMA_FILE_PATH)
        except Exception as e:
            raise Exception(e, sys)
    
 
    def normalize_text(self,df):
        """Apply preprocessing to the text data in the dataframe."""
        try:
            df['clean_comment'] = df['clean_comment'].apply(preprocess_comment)
            logger.debug('Text normalization completed')
            return df
        except Exception as e:
            logger.error(f"Error during text normalization: {e}")
            raise
    def initiate_data_Transformation(self):
        logger.debug("Starting data preprocessing...")
        train_data=read_data(self.data_transformation_config.train_file_path)
        test_data=read_data(self.data_transformation_config.test_file_path)
        logger.debug('Data loaded successfully')

        train_processed_data = self.normalize_text(train_data)
        test_processed_data = self.normalize_text(test_data)

        transform_train_path = self.data_transformation_config.transform_train_file
        transform_test_path = self.data_transformation_config.transform_test_file

        train_processed_data.to_csv(transform_train_path, index=False)
        test_processed_data.to_csv(transform_test_path, index=False)

        logger.info(f"Transformed Train file saved to: {transform_train_path}")
        logger.info(f"Transform Test file saved to: {transform_test_path}")
          


