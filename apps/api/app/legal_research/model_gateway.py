"""Stable provider-neutral gateway exports for the legal research package."""

from app.model_gateway import (
    AnthropicModelGateway,
    ModelGateway,
    ModelGatewayConfig,
    ModelGatewayError,
    ModelRateLimitError,
    ModelSchemaError,
    ModelTechnicalError,
    ModelTransportError,
    OpenAIModelGateway,
    create_model_gateway,
    get_model_gateway_config,
)

__all__ = [
    "AnthropicModelGateway", "ModelGateway", "ModelGatewayConfig",
    "ModelGatewayError", "ModelRateLimitError", "ModelSchemaError",
    "ModelTechnicalError", "ModelTransportError", "OpenAIModelGateway",
    "create_model_gateway", "get_model_gateway_config",
]
