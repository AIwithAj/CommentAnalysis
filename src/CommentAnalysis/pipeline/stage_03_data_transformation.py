from src.CommentAnalysis import logger
from src.CommentAnalysis.components.data_transformation import DataTransformation
from src.CommentAnalysis.config.configuration import ConfigurationManager


STAGE_NAME = "Data Transformation stage"


class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            data_transformation_config = config.get_data_transformation_config()
            data_validation = DataTransformation(data_transformation_config=data_transformation_config)
            data_validation.initiate_data_Transformation() 
        
        except Exception as e:
            raise e
        



STAGE_NAME = "Data Transformation stage"

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformationTrainingPipeline
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e