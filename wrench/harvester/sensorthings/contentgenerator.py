from pathlib import Path

import yaml
from ollama import Client
from pydantic import BaseModel, Field

from wrench.grouper.base import Group
from wrench.models import CommonMetadata


class Content(BaseModel):
    name: str
    description: str


class GeneratorConfig(BaseModel):
    """Configuration for SDDI Catalogger."""

    @classmethod
    def from_yaml(cls, config: str | Path) -> "GeneratorConfig":
        with open(config, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls.model_validate(config_dict)

    llm_host: str = Field(description="URL for the LLM host")

    model: str = Field(description="LLM model to use")


class ContentGenerator:
    def __init__(self, config: GeneratorConfig | str | Path):
        """
        Initialize the LLM based content generator for entry metadata.

        Args:
            config (GeneratorConfig | str | Path): LLM config for content generation
            provided directly or via path to YAML file

        """
        if isinstance(config, (str, Path)):
            config = GeneratorConfig.from_yaml(config)

        self.config = config
        self.client = Client(host=self.config.llm_host)
        self.model = self.config.model

    def generate_content(
        self, service_metadata: CommonMetadata, group: Group
    ) -> tuple[str, str]:
        system_prompt = """You are an agent generating name and description for a urban sensor metadata catalog entry
              based on solely the information given by the user, do not add extra information which is not given by the user.
              Here are some examples of name and descriptions:
              Request:
                Data:
                  Measured Parameter: Bicycle Count
                  Server Source: City X FROST Server
                  Sample Data: {
                    "@iot.selfLink": "https://city-x.com/v1.1/Things(72)",
                    "@iot.id": 72,
                    "description": "Bicycle Count (Location: XYZ)",
                    "name": "Bicycle Count XYZ",
                    "properties": {
                        "operation_status": "inactive",
                        "deviceIDniota": 4174,
                        "keywords": ["Bicycle", "Count"],
                        "language": "en",
                        "owner": "Transportation Minister of City X",
                        "ownerThing": "Transportation Minister of City X",
                        "topic": "Bicycle count"
                    },
                  }
              Response:
              {{
                "name": "Bicycle Counts in City X",
                "description": "Bicycle counts conducted by Transportation Minister of City X around City X at key locations."
              }}"""

        prompt = """
            Data:
                Measured Parameter: {measured_param}
                Server Source: {title}
                Sample Data: {data}
            """

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": prompt.format(
                    measured_param=group.name,
                    title=service_metadata.title,
                    data=[group.items[0]],
                ),
            },
        ]
        response = self.client.chat(
            model=self.model,
            messages=messages,
            format=Content.model_json_schema(),
        )
        if not response.message.content:
            raise RuntimeError("LLM returned no messages")

        content = Content.model_validate_json(response.message.content)

        return content.name, content.description
