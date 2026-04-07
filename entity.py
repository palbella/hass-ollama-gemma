"""Base entity for the Ollama integration."""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator, Callable
import json
import logging
import re
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
    CONF_NUM_CTX,
    CONF_THINK,
    CONF_TOOL_CALL_TYPE,
    DEFAULT_KEEP_ALIVE,
    DEFAULT_MAX_HISTORY,
    DEFAULT_NUM_CTX,
    DOMAIN,
    REACT_SYSTEM_PROMPT,
    TOOL_CALL_TYPE_REACT,
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


def _extract_json(text: str) -> dict[str, Any] | None:
    """Extract and validate JSON tool call from text."""
    # Look for something that looks like target JSON: {"action": "...", "parameters": {...}}
    # We use a non-greedy match for the action part and handle nested braces for parameters
    match = re.search(r'(\{.*"action"\s*:\s*".*"\s*,\s*"parameters"\s*:\s*\{.*\}\s*\})', text, re.DOTALL)
    if not match:
        # Fallback to any JSON-like structure if the template is not strictly followed
        match = re.search(r'(\{.*\})', text, re.DOTALL)
    
    if match:
        try:
            data = json.loads(match.group(1))
            if isinstance(data, dict) and "action" in data:
                return data
        except (json.JSONDecodeError, TypeError):
            pass
    return None


def _convert_content(
    chat_content: (
        conversation.Content
        | conversation.ToolResultContent
        | conversation.AssistantContent
    ),
    tool_call_type: str | None = None,
) -> ollama.Message:
    """Create tool response content."""
    if isinstance(chat_content, conversation.ToolResultContent):
        content = json.dumps(chat_content.tool_result)
        if tool_call_type == TOOL_CALL_TYPE_REACT:
            # In ReAct mode, observations are fed back as user messages or special formats
            return ollama.Message(
                role=MessageRole.USER.value,
                content=f"Observation: {content}",
            )
        return ollama.Message(
            role=MessageRole.TOOL.value,
            content=content,
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
    tool_call_type: str | None = None,
) -> AsyncGenerator[conversation.AssistantContentDeltaDict]:
    """Transform the response stream into HA format."""

    new_msg = True
    accumulated_content = ""
    action_sent = False
    
    async for response in result:
        _LOGGER.debug("Received response: %s", response)
        response_message = response["message"]
        chunk: conversation.AssistantContentDeltaDict = {}
        
        if new_msg:
            new_msg = False
            chunk["role"] = "assistant"
        
        # Native tool calling
        if (tool_calls := response_message.get("tool_calls")) is not None:
            chunk["tool_calls"] = [
                llm.ToolInput(
                    tool_name=tool_call["function"]["name"],
                    tool_args=_parse_tool_args(tool_call["function"]["arguments"]),
                )
                for tool_call in tool_calls
            ]
        
        # Text content handling (including ReAct parsing)
        if (content := response_message.get("content")) is not None:
            chunk["content"] = content
            if tool_call_type == TOOL_CALL_TYPE_REACT and not action_sent:
                accumulated_content += content
                if action := _extract_json(accumulated_content):
                    action_sent = True
                    chunk["tool_calls"] = [
                        llm.ToolInput(
                            tool_name=action["action"],
                            tool_args=_parse_tool_args(action.get("parameters", {})),
                        )
                    ]
        
        if (thinking := response_message.get("thinking")) is not None:
            chunk["thinking_content"] = thinking
        
        if response_message.get("done"):
            new_msg = True
            accumulated_content = ""
            action_sent = False
            
        yield chunk


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
        model = settings[CONF_MODEL]

        tools: list[dict[str, Any]] | None = None
        if chat_log.llm_api:
            tools = [
                _format_tool(tool, chat_log.llm_api.custom_serializer)
                for tool in chat_log.llm_api.tools
            ]

        tool_call_type = settings.get(CONF_TOOL_CALL_TYPE)

        message_history: MessageHistory = MessageHistory(
            [_convert_content(content, tool_call_type) for content in chat_log.content]
        )
        
        # ReAct Prompt Manipulation
        if tool_call_type == TOOL_CALL_TYPE_REACT and tools:
            # Build tools list for prompt
            tools_desc = "\n".join([
                f"- {t['function']['name']}: {t['function'].get('description', '')}. parameters: {json.dumps(t['function']['parameters'])}"
                for t in tools
            ])
            react_prompt = REACT_SYSTEM_PROMPT.format(tools_list=tools_desc)
            
            # Prefix the existing history with our React instructions if it doesn't have it yet
            if not any(react_prompt in m.get("content", "") for m in message_history.messages):
                message_history.messages.insert(0, ollama.Message(role=MessageRole.SYSTEM.value, content=react_prompt))
            
            # Disable native tool calling
            tools = None

        max_messages = int(settings.get(CONF_MAX_HISTORY, DEFAULT_MAX_HISTORY))
        self._trim_history(message_history, max_messages)

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

        # Get response
        # To prevent infinite loops, we limit the number of iterations
        for _iteration in range(MAX_TOOL_ITERATIONS):
            try:
                response_generator = await client.chat(
                    model=model,
                    # Make a copy of the messages because we mutate the list later
                    messages=list(message_history.messages),
                    tools=tools,
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

            message_history.messages.extend(
                [
                    _convert_content(content, tool_call_type)
                    async for content in chat_log.async_add_delta_content_stream(
                        self.entity_id, _transform_stream(response_generator, tool_call_type)
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
