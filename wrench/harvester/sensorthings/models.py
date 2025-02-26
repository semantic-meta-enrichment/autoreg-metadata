from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

model_config = ConfigDict(
    alias_generator=to_camel,
    populate_by_name=True,
    from_attributes=True,
    coerce_numbers_to_str=True,
)


class SensorThingsBase(BaseModel):
    """Base mixin for common fields for relevant SensorThings API Entities."""

    model_config = model_config
    id: str = Field(validation_alias="@iot.id", serialization_alias="@iot.id")
    name: str
    description: str
    properties: dict[str, Any] | None = None


class Sensor(SensorThingsBase):
    encoding_type: str


class ObservedProperty(SensorThingsBase):
    pass


class Datastream(SensorThingsBase):
    unit_of_measurement: dict
    observed_area: dict | None = None
    phenomenon_time: str | None = None
    result_time: str | None = None
    sensor: Sensor = Field(alias="Sensor")
    observed_property: ObservedProperty | None = Field(
        default=None, alias="ObservedProperty"
    )


class GeoPoint(BaseModel):
    """Represents a GeoJSON Point Geometry."""

    type: str = "Point"
    coordinates: tuple[float, float]  # longitude, latitude


class GenericLocation(ABC, SensorThingsBase):
    encoding_type: str

    @abstractmethod
    def get_coordinates(self) -> tuple[float, float]:
        """
        Retrieves the coordinates of the location.

        Returns:
            tuple[float, float]: Tuple with latitude and longitude
            of the location.
        """
        pass


class Location(GenericLocation):
    location: GeoPoint

    def get_coordinates(self) -> tuple[float, float]:
        """
        Retrieves the coordinates of the location.

        Returns:
            tuple[float, float]: Tuple with latitude and longitude
            of the location.
        """
        return self.location.coordinates


class Thing(SensorThingsBase):
    datastreams: list[Datastream] = Field(
        default=[],
        alias="Datastreams",
    )
    location: list[Location] = Field(
        default=[],
        alias="Locations",
    )

    def __str__(self):
        """
        Returns a JSON string representation of the model.

        The JSON string is generated using the `model_dump_json` method with
        `by_alias` set to True and `exclude_none` set to True.

        Returns:
            str: A JSON string representation of the model.
        """
        return self.model_dump_json(by_alias=True, exclude_none=True)
