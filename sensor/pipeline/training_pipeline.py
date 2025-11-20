from sensor.logger import logging
from sensor.expection import SensorException
from sensor.utils import get_collection_as_dataframe
import os,sys
from sensor.entity import config_entity
from sensor.components.data_ingestion import DataIngestion
# from sensor.components.data_validation import DataValidation
# from sensor.components.data_transformation import DataTransformation
# from sensor.components.model_trainer import ModelTrainer
# from sensor.components.model_evaluation import ModelEvaluation
# from sensor.components.model_pusher import ModelPusher

def start_training_pipeline():
    try:
        training_pipeline_config=config_entity.TrainingPipelineConfig()
        
        #data ingestion
        data_ingestion_config=config_entity.DataIngestionConfig(training_pipeline_config=training_pipeline_config)
        print(data_ingestion_config.to_dict())
        data_ingestion=DataIngestion(data_ingestion_config=data_ingestion_config)
        data_ingestion_artifact=data_ingestion.initiate_data_ingestion()
    
    except Exception as e:
        raise SensorException(e,sys)