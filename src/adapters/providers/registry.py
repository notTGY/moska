"""Model-provider registry. Independent of local worker runtimes."""

from collections.abc import Callable
from dataclasses import dataclass

from .openai import OpenAIConnection
from .openai_model import ChatGPTModel


@dataclass(frozen=True)
class ProviderSpec:
    provider_id: str
    name: str
    account_name: str
    connect_label: str
    connect_instructions: str
    access_label: str
    connection_factory: Callable
    model_factory: Callable

    def create_connection(self):
        return self.connection_factory()

    def create_model(self, connection, model_name, reasoning_effort):
        return self.model_factory(
            connection=connection, model_name=model_name, reasoning_effort=reasoning_effort,
        )


def openai_connection():
    return OpenAIConnection()


MODEL_PROVIDERS = {
    'openai': ProviderSpec(
        provider_id='openai', name='OpenAI', account_name='ChatGPT',
        connect_label='Continue with ChatGPT',
        connect_instructions='Sign in and allow Moska to use your ChatGPT plan.',
        access_label='ChatGPT plan usage',
        connection_factory=openai_connection, model_factory=ChatGPTModel,
    ),
}
