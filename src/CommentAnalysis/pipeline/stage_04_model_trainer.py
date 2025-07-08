from src.CommentAnalysis import logger
from src.CommentAnalysis.components.model_trainer import ModelTrainer
from src.CommentAnalysis.config.configuration import ConfigurationManager


STAGE_NAME = "Data Model Training stage"


class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            model_config = config.get_model_config()
            modeltrainer = ModelTrainer(model_config)
            modeltrainer.initiate_model_training() 
        
        except Exception as e:
            raise e
        



STAGE_NAME = "Data Transformation stage"

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e