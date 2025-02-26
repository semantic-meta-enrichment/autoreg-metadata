from enum import Enum

from pydantic import BaseModel, Field, computed_field


class SDDICategory(Enum):
    online_service = "online-service"
    device = "device"


class SDDIDataset(BaseModel):
    # required
    name: str = Field(serialization_alias="name")  # use "name" for the JSON payload
    title: str = Field(serialization_alias="title")
    notes: str = Field(serialization_alias="notes")
    owner_org: str
    # optional with predefined defaults
    url: str | None = None
    author: str = ""
    author_email: str = ""
    end_collection_date: str = ""  # necessary to properly register entries
    groups: list[dict] = []
    language: str = ""
    licence_agreement: list[str] = ["licence_agreement_check"]
    license_id: str = "Apache-2.0"
    license_title: str = "Apache License 2.0"
    license_url: str = "https://www.apache.org/licenses/LICENSE-2.0"
    private: bool = False
    relationships_as_object: list[dict] = []
    relationships_as_subject: list[dict] = []
    spatial: str = ""
    state: str = "active"
    tags: list[dict] = []
    type: str = "dataset"

    def model_dump(self, **kwargs):
        """Override to ensure serialization aliases are used by default."""
        kwargs.setdefault("by_alias", True)
        return super().model_dump(**kwargs)


class OnlineService(SDDIDataset):
    groups: list[dict] = [{"name": SDDICategory.online_service.value}]

    @computed_field  # type: ignore[misc]
    @property
    def resources(self) -> list[dict]:
        return [
            {
                "name": "API Service URL",
                "description": "URL for the API service",
                "format": "api",
                "url": self.url,
            }
        ]


class DeviceGroup(SDDIDataset):
    groups: list[dict] = [{"name": SDDICategory.device.value}]
    resources: list[dict] = []

    @classmethod
    def from_api_service(
        cls,
        online_service: OnlineService,
        name: str,
        description: str,
        tags: list[dict[str, str]],
        resources: list = [],
    ) -> "DeviceGroup":
        # Get all fields from api_service except 'groups' and 'resources'
        ckan_name = name.lower().strip().replace(" ", "_")
        data = online_service.model_dump(exclude={"groups", "resources"})
        data.update(
            {
                "id": ckan_name,
                "name": name,
                "description": description,
                "tags": tags,
                "resources": resources or [],
            }
        )
        return cls(**data)
