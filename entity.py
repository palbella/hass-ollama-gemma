"""Base entity for the Ollama integration."""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator, Callable
import json
import logging
from typing import Any

import ollama
import voluptuous as vol
from voluptuous_openapi import convert

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr, llm
from homeassistant.helpers.entity import Entity

from . import OllamaConfigEntry
from .const import (
    CONF_KEEP_ALIVE,
    CONF_MAX_HISTORY,
    CONF_MODEL,
    CONF_FUNCTION_MODEL,
    CONF_NUM_CTX,
    CONF_THINK,
    DEFAULT_KEEP_ALIVE,
    DEFAULT_MAX_HISTORY,
    DEFAULT_NUM_CTX,
    DOMAIN,
)
from .models import MessageHistory, MessageRole

# Max number of back and forth with the LLM to generate a response
MAX_TOOL_ITERATIONS = 10

_LOGGER = logging.getLogger(__name__)


def _format_tool(
    tool: llm.Tool, custom_serializer: Callable[[Any], Any] | None
) -> dict[str, Any]:
    """Format tool specification."""
    tool_spec = {
        "name": tool.name,
        "parameters": convert(tool.parameters, custom_serializer=custom_serializer),
    }
    if tool.description:
        tool_spec["description"] = tool.description
    return {"type": "function", "function": tool_spec}


def _fix_invalid_arguments(value: Any) -> Any:
    """Attempt to repair incorrectly formatted json function arguments.

    Small models (for example llama3.1 8B) may produce invalid argument values
    which we attempt to repair here.
    """
    if not isinstance(value, str):
        return value
    if (value.startswith("[") and value.endswith("]")) or (
        value.startswith("{") and value.endswith("}")
    ):
        try:
            return json.loads(value)
        except json.decoder.JSONDecodeError:
            pass
    return value


def _parse_tool_args(arguments: dict[str, Any]) -> dict[str, Any]:
    """Rewrite ollama tool arguments.

    This function improves tool use quality by fixing common mistakes made by
    small local tool use models. This will repair invalid json arguments and
    omit unnecessary arguments with empty values that will fail intent parsing.
    """
    return {
        k: _fix_invalid_arguments(v)
        for k, v in arguments.items()
        if v is not None and v != ""
    }


def _convert_content(
    chat_content: (
        conversation.Content
        | conversation.ToolResultContent
        | conversation.AssistantContent
    ),
) -> ollama.Message:
    """Create tool response content."""
    if isinstance(chat_content, conversation.ToolResultContent):
        return ollama.Message(
            role=MessageRole.TOOL.value,
            content=json.dumps(chat_content.tool_result),
        )
    if isinstance(chat_content, conversation.AssistantContent):
        return ollama.Message(
            role=MessageRole.ASSISTANT.value,
            content=chat_content.content,
            thinking=chat_content.thinking_content,
            tool_calls=[
                ollama.Message.ToolCall(
                    function=ollama.Message.ToolCall.Function(
                        name=tool_call.tool_name,
                        arguments=tool_call.tool_args,
                    )
                )
                for tool_call in chat_content.tool_calls or ()
            ]
            or None,
        )
    if isinstance(chat_content, conversation.UserContent):
        images: list[ollama.Image] = []
        for attachment in chat_content.attachments or ():
            if not attachment.mime_type.startswith("image/"):
                raise HomeAssistantError(
                    translation_domain=DOMAIN,
                    translation_key="unsupported_attachment_type",
                )
            images.append(ollama.Image(value=attachment.path))
        return ollama.Message(
            role=MessageRole.USER.value,
            content=chat_content.content,
            images=images or None,
        )
    if isinstance(chat_content, conversation.SystemContent):
        return ollama.Message(
            role=MessageRole.SYSTEM.value,
            content=chat_content.content,
        )
    raise TypeError(f"Unexpected content type: {type(chat_content)}")


async def _transform_stream(
    result: AsyncIterator[ollama.ChatResponse],
) -> AsyncGenerator[conversation.AssistantContentDeltaDict]:
    """Transform the response stream into HA format.

    An Ollama streaming response may come in chunks like this:

    response: message=Message(role="assistant", content="Paris")
    response: message=Message(role="assistant", content=".")
    response: message=Message(role="assistant", content=""), done: True, done_reason: "stop"
    response: message=Message(role="assistant", tool_calls=[...])
    response: message=Message(role="assistant", content=""), done: True, done_reason: "stop"

    This generator conforms to the chatlog delta stream expectations in that it
    yields deltas, then the role only once the response is done.
    """

    new_msg = True
    async for response in result:
        _LOGGER.debug("Pablo: Received response: %s", response)
        response_message = response["message"]
        chunk: conversation.AssistantContentDeltaDict = {}
        if new_msg:
            new_msg = False
            chunk["role"] = "assistant"
        if (tool_calls := response_message.get("tool_calls")) is not None:
            chunk["tool_calls"] = [
                llm.ToolInput(
                    tool_name=tool_call["function"]["name"],
                    tool_args=_parse_tool_args(tool_call["function"]["arguments"]),
                )
                for tool_call in tool_calls
            ]
        if (content := response_message.get("content")) is not None:
            chunk["content"] = content
        if (thinking := response_message.get("thinking")) is not None:
            chunk["thinking_content"] = thinking
        if response_message.get("done"):
            new_msg = True
        yield chunk


async def _buffered_json_transform_stream(
    result: AsyncIterator[ollama.ChatResponse],
) -> AsyncGenerator[conversation.AssistantContentDeltaDict]:
    """Buffer the stream, parse JSON, and yield tool calls if applicable."""
    full_content = ""
    async for response in result:
        if (content := response["message"].get("content")) is not None:
            full_content += content

    _LOGGER.debug("Pablo: Buffered content from function model: %s", full_content)

    # Attempt to parse JSON
    try:
        # cleanup markdown code blocks if any
        clean_content = full_content.strip()
        if clean_content.startswith("```json"):
            clean_content = clean_content[7:]
        if clean_content.startswith("```"):
            clean_content = clean_content[3:]
        if clean_content.endswith("```"):
            clean_content = clean_content[:-3]
        clean_content = clean_content.strip()

        data = json.loads(clean_content)
        
        # Check if it looks like a tool call (name and parameters/arguments)
        if isinstance(data, dict) and "name" in data:
            tool_name = data["name"]
            tool_args = data.get("parameters", data.get("arguments", {}))
            
            _LOGGER.info("Parsed tool call from JSON: %s(%s)", tool_name, tool_args)
            
            yield {
                "role": "assistant",
                "tool_calls": [
                    llm.ToolInput(
                        tool_name=tool_name,
                        tool_args=_fix_invalid_arguments(tool_args),
                    )
                ],
            }
            return

    except json.JSONDecodeError:
        _LOGGER.warning("Failed to parse JSON from function model response")

    # If parsing failed or not a tool call, yield as text
    yield {
        "role": "assistant",
        "content": full_content,
    }


class OllamaBaseLLMEntity(Entity):
    """Ollama base LLM entity."""

    _attr_has_entity_name = True
    _attr_name = None

    def __init__(self, entry: OllamaConfigEntry, subentry: ConfigSubentry) -> None:
        """Initialize the entity."""
        self.entry = entry
        self.subentry = subentry
        self._attr_unique_id = subentry.subentry_id

        model, _, version = subentry.data[CONF_MODEL].partition(":")
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, subentry.subentry_id)},
            name=subentry.title,
            manufacturer="Ollama",
            model=model,
            sw_version=version or "latest",
            entry_type=dr.DeviceEntryType.SERVICE,
        )

    async def _async_handle_chat_log(
        self,
        chat_log: conversation.ChatLog,
        structure: vol.Schema | None = None,
    ) -> None:
        """Generate an answer for the chat log."""
        settings = {**self.entry.data, **self.subentry.data}

        client = self.entry.runtime_data
        
        tools: list[dict[str, Any]] | None = None
        tool_schemas = []
        if chat_log.llm_api:
            tools = []
            for tool in chat_log.llm_api.tools:
                t = _format_tool(tool, chat_log.llm_api.custom_serializer)
                tools.append(t)
                tool_schemas.append(t)

        # Select model: Use function_model if tools are available/active, else use standard model
        model = settings[CONF_MODEL]
        use_function_model = False
        if tools and settings.get(CONF_FUNCTION_MODEL):
             model = settings[CONF_FUNCTION_MODEL]
             use_function_model = True

        message_history: MessageHistory = MessageHistory(
            [_convert_content(content) for content in chat_log.content]
        )
        max_messages = int(settings.get(CONF_MAX_HISTORY, DEFAULT_MAX_HISTORY))
        self._trim_history(message_history, max_messages)

        _LOGGER.debug("Pablo: Variable tool_schemas tiene el valor: %s", tool_schemas)
        # Inject system prompt for function model (Force JSON)
        if use_function_model:
            system_prompt = (
                "You are an AI assistant capable of calling tools. "
                "The available tools are defined below in JSON schema format:\n\n"
                f"{json.dumps(tool_schemas, indent=2)}\n\n"
                "To call a tool, verify the tool name and arguments usage, simply respond with a JSON object containing the `name` of the tool and its `parameters`. "
                "Do not add any explanation, markdown formatting, or extra text. Just the JSON object.\n"
                "Example: {\"name\": \"tool_name\", \"parameters\": {\"arg1\": \"value1\"}}"
            )
            
            if message_history.messages and message_history.messages[0]["role"] == "system":
                 message_history.messages[0]["content"] += f"\n\n{system_prompt}"
            else:
                message_history.messages.insert(0, ollama.Message(role="system", content=system_prompt))

        output_format: dict[str, Any] | None = None
        if structure:
            output_format = convert(
                structure,
                custom_serializer=(
                    chat_log.llm_api.custom_serializer
                    if chat_log.llm_api
                    else llm.selector_serializer
                ),
            )
        
        _LOGGER.debug("Pablo: Usando modelo: %s", model)
        _LOGGER.debug("Pablo: Herramientas disponibles (tools): %s", tools)
        _LOGGER.debug("Pablo: Historial de mensajes final: %s", message_history.messages)
        
        # Get response
        # To prevent infinite loops, we limit the number of iterations
        for _iteration in range(MAX_TOOL_ITERATIONS):
            try:
                # If using function model manual mode, we DO NOT pass `tools` to Ollama API
                # because we are handling it via prompt and parsing.
                api_tools = tools if not use_function_model else None

                response_generator = await client.chat(
                    model=model,
                    # Make a copy of the messages because we mutate the list later
                    messages=list(message_history.messages), # type: ignore
                    tools=api_tools, # type: ignore
                    stream=True,
                    # keep_alive requires specifying unit. In this case, seconds
                    keep_alive=f"{settings.get(CONF_KEEP_ALIVE, DEFAULT_KEEP_ALIVE)}s",
                    options={CONF_NUM_CTX: settings.get(CONF_NUM_CTX, DEFAULT_NUM_CTX)},
                    think=settings.get(CONF_THINK),
                    format=output_format,
                )
            except (ollama.RequestError, ollama.ResponseError) as err:
                _LOGGER.error("Unexpected error talking to Ollama server: %s", err)
                raise HomeAssistantError(
                    f"Sorry, I had a problem talking to the Ollama server: {err}"
                ) from err

            # Select stream transformer
            transformer = _transform_stream
            if use_function_model:
                transformer = _buffered_json_transform_stream

            message_history.messages.extend(
                [
                    _convert_content(content)
                    async for content in chat_log.async_add_delta_content_stream(
                        self.entity_id, transformer(response_generator)
                    )
                ]
            )

            if not chat_log.unresponded_tool_results:
                break

    def _trim_history(self, message_history: MessageHistory, max_messages: int) -> None:
        """Trims excess messages from a single history.

        This sets the max history to allow a configurable size history may take
        up in the context window.

        Note that some messages in the history may not be from ollama only, and
        may come from other anents, so the assumptions here may not strictly hold,
        but generally should be effective.
        """
        if max_messages < 1:
            # Keep all messages
            return

        # Ignore the in progress user message
        num_previous_rounds = message_history.num_user_messages - 1
        if num_previous_rounds >= max_messages:
            # Trim history but keep system prompt (first message).
            # Every other message should be an assistant message, so keep 2x
            # message objects. Also keep the last in progress user message
            num_keep = 2 * max_messages + 1
            drop_index = len(message_history.messages) - num_keep
            message_history.messages = [
                message_history.messages[0],
                *message_history.messages[drop_index:],
            ]
