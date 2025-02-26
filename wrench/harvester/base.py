from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from pydantic import BaseModel

from wrench.grouper.base import Group
from wrench.models import CommonMetadata

T = TypeVar("T", bound=BaseModel)  # For input
T_co = TypeVar("T_co", bound=BaseModel, covariant=True)  # For output


class BaseHarvester(ABC):
    @abstractmethod
    def return_items(self) -> list:
        pass

    @abstractmethod
    def get_service_metadata(self) -> CommonMetadata:
        """
        Retrieve metadata information from the sensor service.

        Returns:
            CommonMetadata: The metadata information.
        """
        pass

    @abstractmethod
    def get_device_group_metadata(self, group: Group) -> CommonMetadata:
        """
        Retrieve metadata information from the device groups.

        Args:
            group(Group): The grouped devices returned from BaseGrouper
        Returns:
            CommonMetadata: The metadata of the device group
        """
        pass


class TranslationService(ABC, Generic[T]):
    url: str

    @abstractmethod
    def translate[T: BaseModel](self, obj: T) -> T:
        """
        Translates the given object.

        This function takes an object of type T,which must be a
        subclass of BaseModel, and returns an object of the same type.

        Args:
            obj (T): The object to be translated.

        Returns:
            T: The translated object.
        """
        pass
