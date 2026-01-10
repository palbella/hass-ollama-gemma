"""The conversation platform for the Gemma Multi-Model integration."""

from __future__ import annotations

from typing import Literal
import json

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import OllamaConfigEntry
from .const import CONF_PROMPT, DOMAIN
from .entity import OllamaBaseLLMEntity

import logging

_LOGGER = logging.getLogger(__name__)

async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: OllamaConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up conversation entities."""
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != "conversation":
            continue

        async_add_entities(
            [GemmaMultiModelAgent(config_entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )

class GemmaMultiModelAgent(
    conversation.ConversationEntity,
    conversation.AbstractConversationAgent,
    OllamaBaseLLMEntity,
):
    """Gemma Multi-Model conversation agent."""

    # Desactivamos streaming para asegurar que la orquestación entre modelos termine antes de mostrar texto
    _attr_supports_streaming = False

    def __init__(self, entry: OllamaConfigEntry, subentry: ConfigSubentry) -> None:
        """Initialize the agent."""
        super().__init__(entry, subentry)
        if self.subentry.data.get(CONF_LLM_HASS_API):
            self._attr_supported_features = (
                conversation.ConversationEntityFeature.CONTROL
            )

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        return MATCH_ALL

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Proceso de orquestación local Gemma."""
        settings = {**self.entry.data, **self.subentry.data}

        _LOGGER.debug("Variable user_input tiene el valor: %s", user_input)
        _LOGGER.debug("Variable chat_log tiene el valor: %s", chat_log)
        _LOGGER.error("Fin Variables")
        
        # 1. Preparar datos para FunctionGemma (Modelo de lógica)
        try:
            await chat_log.async_provide_llm_data(
                user_input.as_llm_context(DOMAIN),
                settings.get(CONF_LLM_HASS_API),
                settings.get(CONF_PROMPT),
                user_input.extra_system_prompt,
            )
        except conversation.ConverseError as err:
            return err.as_conversation_result()

        # 2. Primera llamada: FunctionGemma decide qué hacer
        # Usará el modelo definido en la configuración (deberías poner functiongemma allí)
        await self._async_handle_chat_log(chat_log)

        # 3. Segunda llamada: Humanización con Gemma 3:1B
        # Solo si queremos que la respuesta final sea procesada por el modelo pequeño
        last_content = chat_log.unassimilated_messages[-1].content if chat_log.unassimilated_messages else ""
        
        # Creamos una petición interna rápida a Ollama para gemma3:1b
        # Esto sobreescribe la respuesta técnica con una natural
        try:
            hugging_prompt = f"Eres un asistente de casa inteligente. El sistema ejecutó: {last_content}. Responde de forma muy breve y natural al usuario."
            
            # Llamamos directamente al cliente de Ollama que ya está en la clase base
            # Forzamos el modelo gemma3:1b
            response = await self.client.chat(
                model="gemma3:1b",
                messages=[{"role": "user", "content": hugging_prompt}],
                options={"num_predict": 50, "temperature": 0.4}
            )
            
            # Sustituimos el contenido del log para que HA devuelva esta respuesta
            if response and "message" in response:
                chat_log.unassimilated_messages[-1].content = response["message"]["content"]

        except Exception:
            # Si falla la humanización, devolvemos la respuesta original de FunctionGemma
            pass

        return conversation.async_get_result_from_chat_log(user_input, chat_log)

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self.entry, self)

    async def async_will_remove_from_hass(self) -> None:
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()
