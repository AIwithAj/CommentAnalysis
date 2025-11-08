from src.CommentAnalysis import logger
from src.CommentAnalysis.pipeline.register_model import main as register_model_main
from src.CommentAnalysis.config.configuration import ConfigurationManager


STAGE_NAME = "Model Registration stage"


class ModelRegistrationPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            # Register model to MLflow
            register_model_main()
        except Exception as e:
            raise e


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelRegistrationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e

