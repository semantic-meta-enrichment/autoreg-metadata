from pathlib import Path

from geojson import MultiPoint

from wrench.adapter.base import AdapterConfig, BaseCatalogAdapter
from wrench.catalogger.sddi.models import DeviceGroup, OnlineService
from wrench.catalogger.sddi.register import SDDICatalogger
from wrench.grouper.base import Group
from wrench.harvester.sensorthings.harvester import SensorThingsHarvester
from wrench.harvester.sensorthings.models import Thing
from wrench.harvester.sensorthings.querybuilder import ThingQuery
from wrench.models import CommonMetadata


class SensorThingsSDDIAdapter(
    BaseCatalogAdapter[SensorThingsHarvester, SDDICatalogger]
):
    def __init__(self, config: AdapterConfig | str | Path):
        if isinstance(config, (str, Path)):
            config = AdapterConfig.from_yaml(config)

        self.config = config
        super().__init__(llm_host=self.config.llm_host, model=self.config.llm_model)

    def create_service_entry(self, metadata: CommonMetadata) -> OnlineService:

        # set a default owner for now HANDLE THIS LATER
        owner = metadata.owner or "lehrstuhl-fur-geoinformatik"

        service = OnlineService(
            url=metadata.endpoint_url,
            id=metadata.identifier,
            description=metadata.description,
            owner_org=owner,
            name=metadata.title,
            tags=[{"name": tag} for tag in metadata.tags],
            spatial=metadata.spatial_extent,
        )

        print(service)

        return service

    def create_group_entry(
        self, api_service: OnlineService, group: Group
    ) -> DeviceGroup:

        self.logger.info("Creating device group")

        domain_groups = [
            "administration",
            "mobility",
            "environment",
            "agriculture",
            "urban-planning",
            "health",
            "energy",
            "information-technology",
            "tourism",
            "living",
            "education",
            "construction",
            "culture",
            "trade",
            "craft",
            "work",
        ]

        catalog_entry = self._generate_catalog_data(
            service_entry=api_service, group=group
        )

        coord = []
        ids = []

        for item in group.items:
            thing_with_location = Thing.model_validate_json(item)
            ids.append(thing_with_location.id)
            if not thing_with_location.location:
                continue
            for loc in thing_with_location.location:
                lon, lat = loc.get_coordinates()
                coord.append((lon, lat))

        self.logger.info("Finished getting things with locations")

        query = ThingQuery()
        filter_expression = None

        for id in ids:
            current_filter = ThingQuery.property("@iot.id").eq(id)

            if filter_expression is None:
                filter_expression = current_filter
            else:
                filter_expression = filter_expression | current_filter

        param_url = query.filter(filter_expression).build()

        device_group = DeviceGroup.from_api_service(
            online_service=api_service,
            name=catalog_entry.name,
            tags=[{"name": tag} for tag in group.parent_classes],
            description=catalog_entry.description,
            resources=[
                {
                    "name": f"URL for {catalog_entry.name}",
                    "description": f"URL provides a list of all data associated with the category {group.name}",
                    "format": "JSON",
                    "url": f"{api_service.url}/{param_url}",
                }
            ],
        )
        # extend group with any of the domain names from classifier (e.g. mobility)
        device_group.groups.extend(
            [{"name": dom} for dom in group.parent_classes if dom in domain_groups]
        )

        device_group.spatial = str(MultiPoint(coord))

        return device_group
