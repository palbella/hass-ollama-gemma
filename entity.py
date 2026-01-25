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
    """Attempt to repair incorrectly formatted json function arguments."""
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
    """Rewrite ollama tool arguments."""
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
        return ollama.Message(
            role=MessageRole.USER.value,
            content=chat_content.content,
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
    """Transform standard stream."""
    new_msg = True
    async for response in result:
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
        yield chunk

async def _buffered_json_transform_stream(
    result: AsyncIterator[ollama.ChatResponse],
) -> AsyncGenerator[conversation.AssistantContentDeltaDict]:
    """Buffer stream and force JSON tool extraction."""
    full_content = ""
    async for response in result:
        if (content := response["message"].get("content")) is not None:
            full_content += content

    clean_content = full_content.strip()
    if "```" in clean_content:
        clean_content = clean_content.split("```")[1]
        if clean_content.startswith("json"):
            clean_content = clean_content[4:]
    clean_content = clean_content.strip()

    try:
        data = json.loads(clean_content)
        if isinstance(data, dict) and "name" in data:
            yield {
                "role": "assistant",
                "tool_calls": [
                    llm.ToolInput(
                        tool_name=data["name"],
                        tool_args=_fix_invalid_arguments(data.get("parameters", data.get("arguments", {}))),
                    )
                ],
            }
            return
    except json.JSONDecodeError:
        _LOGGER.warning("Ollama sent invalid JSON: %s", clean_content)

    yield {"role": "assistant", "content": full_content}

class OllamaBaseLLMEntity(Entity):
    _attr_has_entity_name = True
    _attr_name = None

    def __init__(self, entry: OllamaConfigEntry, subentry: ConfigSubentry) -> None:
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

    async def _async_handle_chat_log(self, chat_log: conversation.ChatLog, structure: vol.Schema | None = None) -> None:
        settings = {**self.entry.data, **self.subentry.data}
        client = self.entry.runtime_data
        
        tools = []
        if chat_log.llm_api:
            for tool in chat_log.llm_api.tools:
                tools.append(_format_tool(tool, chat_log.llm_api.custom_serializer))

        model = settings[CONF_MODEL]
        use_function_model = False
        if tools and settings.get(CONF_MODEL) #CONF_FUNCTION_MODEL:
             model = settings[CONF_MODEL]
             use_function_model = True

        # Hasta aqui el modelo a usar es functiongemma en model.

        message_history = MessageHistory([_convert_content(content) for content in chat_log.content])
        
        if use_function_model:
            system_prompt = (
                "STRICT JSON MODE. You must only respond with a JSON object to call a tool.\n"
                f"TOOLS AVAILABLE: {json.dumps(tools)}\n"
                "To turn on a light, use: {\"name\": \"HassTurnOn\", \"parameters\": {\"name\": \"ENTITY_NAME\", \"domain\": [\"light\"]}}\n"
                "Respond ONLY with the JSON. No talk."
            )
            if message_history.messages and message_history.messages[0]["role"] == "system":
                 message_history.messages[0]["content"] += f"\n\n{system_prompt}"
            else:
                message_history.messages.insert(0, ollama.Message(role="system", content=system_prompt))

        # DEBUG: Capture request for inspection
        nuevo_pablo_request = json.dumps([
             {
                 "role": m.get("role"),
                 "content": m.get("content"),
                 "tool_calls": [
                     {
                         "function": {
                             "name": tc.function.name,
                             "arguments": tc.function.arguments
                         }
                     } for tc in (m.get("tool_calls") or [])
                 ] if m.get("tool_calls") else None
             }
             for m in message_history.messages
        ], default=str)

        for _ in range(MAX_TOOL_ITERATIONS):
            response_generator = await client.chat(
                model=model,
                messages=list(message_history.messages),
                tools=tools if not use_function_model else None,
                stream=True,
                keep_alive=f"{settings.get(CONF_KEEP_ALIVE, DEFAULT_KEEP_ALIVE)}s",
                options={CONF_NUM_CTX: settings.get(CONF_NUM_CTX, DEFAULT_NUM_CTX)},
            )

            transformer = _buffered_json_transform_stream if use_function_model else _transform_stream

            message_history.messages.extend([
                _convert_content(content) async for content in chat_log.async_add_delta_content_stream(
                    self.entity_id, transformer(response_generator)
                )
            ])
            if not chat_log.unresponded_tool_results:
                break
        
        # DEBUG: Capture final response for inspection
        if message_history.messages:
            last_msg = message_history.messages[-1]
            pablo_final_response = json.dumps({
                "role": last_msg.get("role"),
                "content": last_msg.get("content"),
                "tool_calls": [
                     {
                         "function": {
                             "name": tc.function.name,
                             "arguments": tc.function.arguments
                         }
                     } for tc in (last_msg.get("tool_calls") or [])
                 ] if last_msg.get("tool_calls") else None
            }, default=str)
     model = settings[CONF_MODEL]