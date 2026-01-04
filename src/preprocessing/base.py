# src/preprocessing/base.py

"""
Base class for data preprocessors.

Defines how processing classes must behave
Guarantees that every preprocessor has:
    load()
    validate()
    process()
    output()

"""

from abc import ABC, abstractmethod
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class BasePreprocessor(ABC):
    """
    Abstract base class for all preprocessors.
    """

    def __init__(self, input_path: str):
        self.input_path = input_path

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def process(self):
        pass

    @abstractmethod
    def validate(self):
        pass

    @abstractmethod
    def output(self):
        pass
