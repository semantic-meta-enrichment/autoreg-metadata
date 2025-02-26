from abc import ABC, abstractmethod

from wrench.log import logger
from wrench.models import CommonMetadata


class BaseCatalogger(ABC):
    def __init__(self, endpoint: str, api_key: str):
        """
        Initializes the base class with the given endpoint and API key.

        Args:
            endpoint (str): The API endpoint URL.
            api_key (str): The API key for authentication.
        """
        self.endpoint = endpoint
        self.api_key = api_key

        self.logger = logger.getChild(self.__class__.__name__)

    @abstractmethod
    def register(self, service: CommonMetadata, groups: list[CommonMetadata]):
        pass
