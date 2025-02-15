from pathlib import Path

from ckanapi import RemoteCKAN
from ollama import Client

from autoreg_metadata.catalogger.base import BaseCatalogger
from autoreg_metadata.catalogger.sddi.utils import CatalogGenerator
from autoreg_metadata.common.models import CommonMetadata
from autoreg_metadata.grouper.base import Group
from autoreg_metadata.log import logger

from .config import SDDIConfig
from .models import DeviceGroup, OnlineService


class SDDICatalogger(BaseCatalogger):
    """
    SDDICatalogger is a class responsible for interacting with a SDDI CKAN server to register and manage datasets.

    :param url: The URL of the SDDI CKAN server.
    :param api_key: The API key for authenticating with the SDDI CKAN server.
    """

    def __init__(self, config: SDDIConfig | str | Path):
        # Load config if path is provided
        if isinstance(config, (str, Path)):
            config = SDDIConfig.from_yaml(config)

        self.config = config

        super().__init__(endpoint=self.config.base_url, api_key=self.config.api_key)

        self.ckan_server = RemoteCKAN(address=self.endpoint, apikey=self.api_key)
        self.generator = CatalogGenerator(
            llm_client=Client(host=self.config.llm_host),
            model=self.config.llm_model,
        )

        self.logger = logger.getChild(self.__class__.__name__)

    def register(self, metadata: CommonMetadata, groups: list[Group]):
        try:
            api_service = self.generator.create_api_service(metadata)
            self._register_api_service(api_service)

            self.logger.info("Successfully registered API Service")

            if groups:
                device_groups = self.generator.create_device_groups(api_service, groups)
                self._register_device_groups(device_groups)
                for d in device_groups:
                    self.logger.info(
                        "Creating relationships for device_group %s", d.name
                    )
                    self._register_relationship(
                        api_service_name=api_service.name,
                        device_group_name=d.name,
                    )

        except Exception as e:
            self.logger.error("Failed to register: %s", e)
            raise

    def _register_api_service(self, api_service: OnlineService):
        pkg = self.ckan_server.call_action(
            action="package_create", data_dict=api_service.model_dump()
        )
        return pkg

    def _register_device_groups(self, device_groups: list[DeviceGroup]):

        for device_group in device_groups:
            pkg = self.ckan_server.call_action(
                action="package_create", data_dict=device_group.model_dump()
            )

        return pkg

    def _register_relationship(self, api_service_name: str, device_group_name: str):
        rel = self.ckan_server.call_action(
            action="package_relationship_create",
            data_dict={
                "subject": device_group_name,
                "object": api_service_name,
                "type": "links_to",
            },
        )
        return rel

    def delete_resource(self, dataset_name: str):
        self.ckan_server.call_action(
            action="dataset_purge", data_dict={"id": dataset_name}
        )
        self.logger.info("successfully deleted resource")

    def get_owner_orgs(self) -> list[str]:
        return self.ckan_server.call_action(
            action="organization_list",
        )
