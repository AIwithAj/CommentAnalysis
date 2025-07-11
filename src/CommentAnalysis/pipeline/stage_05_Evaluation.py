from src.CommentAnalysis import logger

from src.CommentAnalysis.components.data_evaluation import Evaluation
from src.CommentAnalysis.config.configuration import ConfigurationManager


STAGE_NAME = "Data Model Evaluatin stage"


class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            eval_config = config.get_evaluation_config()
            evaltrainer = Evaluation(eval_config)
            evaltrainer.initiate_model_evaluation()
        
        except Exception as e:
            raise e
        



if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e