from datetime import datetime
from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ConfigDict

# Define a generic type for source-specific data
T = TypeVar("T")


# define a protocol for location
@runtime_checkable
class Location(Protocol):
    def get_coordinates(self):
        """
        Returns the coordinates as a tuple of two float values.

        Returns:
            tuple: A tuple containing two float values representing the coordinates.
        """
        tuple[float, float]


class TimeFrame(BaseModel):
    start_time: datetime
    latest_time: datetime


class Item(BaseModel):
    """
    Item model representing an entity with an ID.

    Attributes:
        id (str): The unique identifier for the item.
    """

    id: str

    model_config = ConfigDict(arbitrary_types_allowed=True)


class CommonMetadata(BaseModel):
    """
    Extensible common metadata format.

    Attributes:
        identifier (str): A unique identifier for the metadata.
        title (str): The title of the metadata.
        description (str): A brief description of the metadata.
        endpoint_url (str): The URL endpoint where the metadata can be accessed.

        spatial_extent (str, optional): The spatial extent of the data.
                                        Defaults to an empty string.
        temporal_extent (TimeFrame, optional): The temporal extent of the data.
                                               Defaults to None.
        tags (list[str], optional): A list of tags associated with the metadata.
                                    Defaults to an empty list.
        keywords (list[str], optional): A list of keywords associated with
                                        the metadata. Defaults to an empty list.

        source_type (str): The type of source from which the data originates.
        last_updated (datetime, optional): The date and time when the metadata
                                           was last updated. Defaults to None.
        update_frequency (str, optional): The frequency at which the data is
                                          updated. Defaults to None.
        owner (str, optional): The owner of the metadata. Defaults to None.

        license (str, optional): The license under which the data is provided.
                                 Defaults to None.
    """

    # required fields
    identifier: str
    title: str
    description: str
    endpoint_url: str

    # standard, but optional fields
    spatial_extent: str = ""
    temporal_extent: TimeFrame | None = None
    tags: list[str] = []
    keywords: list[str] = []
    thematic_groups: list[str] = []

    # data quality and provenance
    source_type: str
    last_updated: datetime | None = None
    update_frequency: str | None = None
    owner: str | None = None

    # license and access information
    license: str | None = None
