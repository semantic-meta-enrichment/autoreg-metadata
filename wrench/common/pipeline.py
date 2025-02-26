from typing import Optional

# Use TYPE_CHECKING for imports needed only for type hints
from wrench.catalogger.base import BaseCatalogger
from wrench.grouper.base import BaseGrouper
from wrench.harvester.base import BaseHarvester
from wrench.log import logger


class Pipeline:
    """
    A composable pipeline for processing sensor data.

    Components can be added or omitted based on requirements.
    """

    def __init__(
        self,
        harvester: BaseHarvester,
        catalogger: BaseCatalogger,
        grouper: Optional[BaseGrouper] = None,
    ):
        """
        Initialize the pipeline with the given components.

        Args:
            harvester (H): The harvester component responsible for data collection.
            catalogger (C): The catalogger component responsible for cataloging data.
            adapter (BaseCatalogAdapter): The adapter for catalog operations.
            grouper (Optional[G], optional): The grouper component for grouping data. Defaults to None.
        """
        self.harvester = harvester
        self.catalogger = catalogger
        self.grouper = grouper
        self.logger = logger.getChild(self.__class__.__name__)

    def run(self):
        """
        Execute the pipeline with available components.

        Returns PipelineResult containing execution results or None if failed.
        """
        self.logger.info(
            "Running pipeline with %s harvester, %s classifier, and %s catalogger...",
            self.harvester.__class__.__name__,
            self.grouper.__class__.__name__,
            self.catalogger.__class__.__name__,
        )
        try:
            # Step 1: Harvest data
            self.logger.debug("Retrieving data with harvester")
            service_metadata = self.harvester.get_service_metadata()
            documents = self.harvester.return_items()
            if not service_metadata or not documents:
                self.logger.warning("No data retrieved from harvester")
                return None

            # Step 2: Optional classification
            grouped_docs = []
            if self.grouper is not None:
                self.logger.debug("Starting classification")
                try:
                    grouped_docs.extend(self.grouper.group_items(documents))
                except Exception as e:
                    self.logger.error("Classification failed: %s", e)
                    # Continue pipeline even if classification fails

            # Step 3: Run results through adapter
            # service_entry, group_entries = self.adapter.create_entries(
            #     service_metadata, grouped_docs
            # )

            group_metadata = []

            if grouped_docs:
                group_metadata.extend(
                    [
                        self.harvester.get_device_group_metadata(group)
                        for group in grouped_docs
                    ]
                )

            # Step 4: Catalog results
            try:
                # If classification was performed and successful, use classified documents
                # Otherwise, use raw documents in a default structure

                self.logger.debug("Registering data into catalog")
                self.catalogger.register(service_metadata, group_metadata)
            except Exception as e:
                self.logger.error("Cataloging failed: %s", e)
                # Still return results even if cataloging fails

        except Exception as e:
            self.logger.error("Pipeline execution failed: %s", e)
            raise
