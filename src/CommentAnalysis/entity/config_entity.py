from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    
@dataclass(frozen=True)
class DataValidationConfig:
    DATA_VALIDATION_DIR: Path
    train_file_path: Path
    test_file_path: Path
    drift_report_file_path: Path
    validation_status_file: Path
    test_size:float
    local_file_path:Path


@dataclass(frozen=True)
class DataTransformationConfig:
    train_file_path: Path
    test_file_path: Path
    DATA_transformation_DIR: Path
    transform_train_file: Path
    transform_test_file: Path

