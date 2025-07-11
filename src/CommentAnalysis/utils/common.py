import os
import json
import yaml
import joblib
from box import ConfigBox
from box.exceptions import BoxValueError
from pathlib import Path
from typing import Any, Iterable, Union
from typeguard import typechecked
from src.CommentAnalysis import logger
import pandas as pd
from pandas import DataFrame
import sys
# Define he preprocessing function
import os 
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer


def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        # Convert to lowercase
        comment = comment.lower()

        # Remove trailing and leading whitespaces
        comment = comment.strip()

        # Remove newline characters
        comment = re.sub(r'\n', ' ', comment)

        # Remove non-alphanumeric characters, except punctuation
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        logger.error(f"Error in preprocessing comment: {e}")
        return comment
def load_vectorizer(vectorizer_path: str) -> TfidfVectorizer:
    """Load the saved TF-IDF vectorizer."""
    try:
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)
        logger.debug('TF-IDF vectorizer loaded from %s', vectorizer_path)
        return vectorizer
    except Exception as e:
        logger.error('Error loading vectorizer from %s: %s', vectorizer_path, e)
        raise
def read_data(file_path) -> DataFrame:
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        raise Exception(e, sys)
    
@typechecked
def read_yaml(path_to_yaml: Union[str, Path]) -> ConfigBox:
    """
    Read a YAML file and return its contents as a ConfigBox.

    Args:
        path_to_yaml (str | Path): Path to the YAML file.

    Raises:
        ValueError: If YAML file is empty.
        Exception: For any other error.

    Returns:
        ConfigBox: Parsed YAML content.
    """
    try:
        with open(path_to_yaml, "r", encoding="utf-8") as yaml_file:
            content = yaml.safe_load(yaml_file)
            if content is None:
                raise ValueError("YAML file is empty")
            logger.info(f"YAML file loaded successfully: {path_to_yaml}")
            return ConfigBox(content)
    except BoxValueError as e:
        raise ValueError("Invalid YAML content") from e
    except Exception as e:
        raise RuntimeError(f"Failed to read YAML file: {path_to_yaml}") from e


@typechecked
def create_directories(paths: Iterable[Union[str, Path]], verbose: bool = True) -> None:
    """
    Create directories if they don't exist.

    Args:
        paths (Iterable[str | Path]): List or set of directories to create.
        verbose (bool, optional): Log the creation. Defaults to True.
    """
    for path in paths:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory: {path}")


@typechecked
def save_json(path: Union[str, Path], data: dict) -> None:
    """
    Save data to a JSON file.

    Args:
        path (str | Path): Path to save the JSON file.
        data (dict): Data to save.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    logger.info(f"JSON file saved: {path}")


@typechecked
def load_json(path: Union[str, Path]) -> ConfigBox:
    """
    Load data from a JSON file.

    Args:
        path (str | Path): Path to the JSON file.

    Returns:
        ConfigBox: Loaded data.
    """
    with open(path, "r", encoding="utf-8") as f:
        content = json.load(f)
    logger.info(f"JSON file loaded: {path}")
    return ConfigBox(content)


@typechecked
def save_bin(data: Any, path: Union[str, Path]) -> None:
    """
    Save an object to a binary file using joblib.

    Args:
        data (Any): Data to save.
        path (str | Path): Path to save the binary file.
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"Binary file saved: {path}")


@typechecked
def load_bin(path: Union[str, Path]) -> Any:
    """
    Load an object from a binary file.

    Args:
        path (str | Path): Path to the binary file.

    Returns:
        Any: Loaded object.
    """
    data = joblib.load(path)
    logger.info(f"Binary file loaded: {path}")
    return data


@typechecked
def get_size(path: Union[str, Path]) -> str:
    """
    Get size of file in KB.

    Args:
        path (str | Path): Path to the file.

    Returns:
        str: Size as a string.
    """
    size_in_kb = round(os.path.getsize(path) / 1024)
    logger.info(f"Size of {path}: ~{size_in_kb} KB")
    return f"~{size_in_kb} KB"

