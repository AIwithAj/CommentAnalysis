from src.CommentAnalysis.constants import *
from src.CommentAnalysis.utils.common import read_yaml, create_directories
from  src.CommentAnalysis.entity.config_entity import DataIngestionConfig,DataValidationConfig,DataTransformationConfig
class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema=read_yaml(SCHEMA_FILE_PATH)
        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            
        )

        return data_ingestion_config
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        config2=self.config.data_ingestion


        create_directories([config.DATA_VALIDATION_DIR])

        data_validation_config = DataValidationConfig(
            DATA_VALIDATION_DIR=config.DATA_VALIDATION_DIR,
            train_file_path=config.train_file_path,
            test_file_path=config.test_file_path,
            drift_report_file_path=config.drift_report_file_path,
            validation_status_file=config.validation_status_file,
            test_size=self.params.test_size,
            local_file_path=config2.local_data_file
            
        )

        return data_validation_config
      
    def get_data_transformation_config(self) -> DataTransformationConfig:

        config = self.config.data_Transformation
        config2=self.config.data_validation


        create_directories([config.DATA_transformation_DIR])
        

        data_transformation_config = DataTransformationConfig(
            DATA_transformation_DIR=config.DATA_transformation_DIR,
            train_file_path=config2.train_file_path,
            test_file_path=config2.test_file_path,
            transform_train_file=config.transform_train_file,
            transform_test_file=config.transform_test_file
            
            
        )

        return data_transformation_config
      