from src.CommentAnalysis.constants import *
from src.CommentAnalysis.utils.common import read_yaml, create_directories
import os
from  src.CommentAnalysis.entity.config_entity import DataIngestionConfig,DataValidationConfig,DataTransformationConfig,ModelTrainerConfig,EvaluationConfig
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

        trainfilepath = Path(config.transform_train_file)
        trainfiledir,train_filename = os.path.split(trainfilepath)

        testfilepath = Path(config.transform_test_file)
        testfiledir, filename = os.path.split(testfilepath)
        
        create_directories([config.DATA_transformation_DIR,trainfiledir,testfiledir])
        
        max_features = self.params['model_building']['max_features']
        ngram_range = tuple(self.params['model_building']['ngram_range'])
        data_transformation_config = DataTransformationConfig(
            DATA_transformation_DIR=config.DATA_transformation_DIR,
            train_file_path=config2.train_file_path,
            test_file_path=config2.test_file_path,
            transform_train_file=config.transform_train_file,
            transform_test_file=config.transform_test_file,
            x_train_file_path=config.x_train_file_path,
            y_train_file_path=config.y_train_file_path,
            transformer=config.transformer,
            max_features=max_features,
            ngram_range=ngram_range)

        return data_transformation_config
    def get_model_config(self) -> ModelTrainerConfig:

        config = self.config.prepare_model
        config2=self.config.data_Transformation

        learning_rate = self.params['model_building']['learning_rate']
        max_depth = self.params['model_building']['max_depth']
        n_estimators = self.params['model_building']['n_estimators']


        create_directories([config.root_dir])
        

        model_trainer_config = ModelTrainerConfig(
            trained_model_path=config.trained_model_path,
            x_train_file_path=config2.x_train_file_path,
            y_train_file_path=config2.y_train_file_path,
            root_dir=config.root_dir,
            transformer_obj=config2.transformer,
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators
            
        )
        

        return model_trainer_config
    def get_evaluation_config(self) -> EvaluationConfig:
        config=self.config.Model_Evaluation
        Modelconfig = self.config.prepare_model
        dataconfg=self.config.data_Transformation
        create_directories([config.root_dir])

        eval_config = EvaluationConfig(
            path_of_model=Modelconfig.trained_model_path,
            x_train_file_path=dataconfg.x_train_file_path,
            y_train_file_path=dataconfg.y_train_file_path,
            testing_data=dataconfg.transform_test_file,
            mlflow_uri="https://dagshub.com/AIwithAj/CommentAnalysis.mlflow",
            params_file_path=PARAMS_FILE_PATH,
            root_dir=config.root_dir,
            transformer=dataconfg.transformer,
            file_path=config.file_path
        )
        return eval_config