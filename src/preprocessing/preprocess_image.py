# src/preprocessing/preprocess_image.py
from src.preprocessing.base import BasePreprocessor


class ImageSignalPreprocessor(BasePreprocessor):
    """
    Todo: for future image/signal preprocessing.
    """

    def load(self):
        pass

    def validate(self):
        pass

    def process(self):
        # Future work: feature extraction (CNN, signal transforms)
        pass

    def output(self):
        return None
