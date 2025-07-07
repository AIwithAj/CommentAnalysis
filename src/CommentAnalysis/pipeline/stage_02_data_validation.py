from src.CommentAnalysis import logger
from src.CommentAnalysis.components.data_validation import DataValidation
from src.CommentAnalysis.config.configuration import ConfigurationManager


STAGE_NAME = "Data validation stage"

class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            data_validation_config = config.get_data_validation_config()
            data_validation = DataValidation(data_validation_config=data_validation_config)
            data_validation.initiate_data_validation() 
        
        except Exception as e:
            raise e
        




if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataValidationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e