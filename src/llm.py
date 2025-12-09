import json
from dataclasses import dataclass
from enum import Enum

import openai
from loguru import logger
from openai import OpenAI, OpenAIError, AuthenticationError, RateLimitError, APIError
from typing import Union, Optional, List, Any

from openai.types.chat import ChatCompletionMessage, ChatCompletion

from .config import ConfigLoader
from .event import emit_event, EventType


class RoleType(Enum):
    """Message role types"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

class Function:
    """Function to call"""
    name: str
    arguments: str

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "arguments": self.arguments
        }

@dataclass
class ToolCall:
    """A tool/function call"""

    id: str
    type: str = "function"
    function: Function = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "function": self.function.to_dict()
        }

@dataclass
class LlmMessage:
    """LLM Message"""

    role: RoleType
    content: Optional[str] = None
    tool_calls: Optional[List[dict]] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None

    def to_dict(self) -> dict:
        message = {"role": self.role.value}
        if self.content is not None:
            message["content"] = self.content
        if self.tool_calls is not None:
            message["tool_calls"] = self.tool_calls \
                if self.tool_calls is not None and len(self.tool_calls) > 0 \
                else None
        if self.name is not None:
            message["name"] = self.name
        if self.tool_call_id is not None:
            message["tool_call_id"] = self.tool_call_id
        return message

    @classmethod
    def user_message(cls, content: str) -> "LlmMessage":
        """Create a user message"""
        return cls(role=RoleType.USER, content=content)

    @classmethod
    def system_message(cls, content: str) -> "LlmMessage":
        """Create a system message"""
        return cls(role=RoleType.SYSTEM, content=content)

    @classmethod
    def assistant_message(cls, content: Optional[str] = None) -> "LlmMessage":
        """Create an assistant message"""
        return cls(role=RoleType.ASSISTANT, content=content)

    @classmethod
    def tool_message(
        cls, content: str, name, tool_call_id: str
    ) -> "LlmMessage":
        """Create a tool message"""
        return cls(
            role=RoleType.TOOL,
            content=content,
            name=name,
            tool_call_id=tool_call_id,
        )

    @classmethod
    def from_tool_calls(
        cls,
        tool_calls: List[Any],
        content: Union[str, List[str]] = "",
        **kwargs,
    ):
        """Create ToolCallsMessage from raw tool calls.

        Args:
            tool_calls: Raw tool calls from LLM
            content: Optional message content
        """
        formatted_calls = [
            {"id": call.id, "function": call.function.model_dump(), "type": "function"}
            for call in tool_calls
        ]
        return cls(
            role=RoleType.ASSISTANT,
            content=content,
            tool_calls=formatted_calls,
            **kwargs,
        )


class LlmCaller:
    api_key = ConfigLoader.get_config().get("llm", {}).get("api_key")
    base_url = ConfigLoader.get_config().get("llm", {}).get("base_url")

    def __init__(self, system_prompt: str, model: str = None):
        self.client = OpenAI(
            api_key=LlmCaller.api_key,
            base_url=LlmCaller.base_url
        )
        self.model = model or ConfigLoader.get_config().get("llm", {}).get("default_model")
        self.system_prompt = system_prompt

    def ask(self, question: Union[str, list], response_format: dict = { "type": "text" }) -> str:
        try:
            if isinstance(question, str):
                question = question.strip()

            emit_event(EventType.LLM, f"[→LLM] {question}")
            res = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': self.system_prompt},
                    {'role': 'user', 'content': question}
                ],
                response_format=response_format,
                temperature=0.0,
                seed=1234,
                extra_body={ "top_k": None }
            )
            # emit_event(EventType.DEBUG, json.dumps(res, ensure_ascii=False))
            answer = res.choices[0].message.content.strip()
            emit_event(EventType.LLM, f"[←LLM] {answer}")
            emit_event(EventType.LLM, f"[CostToken] {res.usage.total_tokens}")
            return answer
        except openai.BadRequestError as e:
            logger.error(f"question: {question}, response_format: {response_format}, error: {e}")
            if response_format.get('type', 'text') == 'json_object':
                return "{}"
            return ""
        except Exception as e:
            logger.exception(f"llmCaller.ask error ! question:{question}", e)
            if response_format.get('type', 'text') == 'json_object':
                return "{}"
            return ""

    def ask_images(self, question: str, image_urls: list[str],
                   response_format: dict = { "type": "text" }) -> str:
        try:

            user_content = [
                {"type": "image_url", "image_url": {"url": image_url}}
                for image_url in image_urls
            ]
            user_content.append({"type": "text", "text": question})

            question = question.strip()
            emit_event(EventType.LLM, f"[→LLM] image_urls:{image_urls}\nquestion:\n{question}")
            msgs = [
                {'role': 'system', 'content': [{ 'type': 'text', 'text': self.system_prompt }]},
                {'role': 'user', 'content': user_content}
            ]
            emit_event(EventType.LLM, f"[→LLM] ask_images_multi:{json.dumps(msgs, ensure_ascii=False)}")
            res = self.client.chat.completions.create(
                model="qwen-vl-max-latest",
                messages=msgs,
                temperature=0.0,
                seed=1234,
                extra_body={ "top_k": None }
            )
            # emit_event(EventType.DEBUG, json.dumps(res, ensure_ascii=False))
            answer = res.choices[0].message.content.strip()
            emit_event(EventType.LLM, f"[←LLM] {answer}")
            emit_event(EventType.LLM, f"[CostToken] {res.usage.total_tokens}")
            return answer
        except Exception as e:
            logger.exception(f"llmCaller.ask error ! question:{question}", e)
            return ""

    def ask_images_multi(self, messages: list[dict]) -> str:
        try:
            msgs = [
                {'role': 'system', 'content': [{ 'type': 'text', 'text': self.system_prompt }]}
            ] + messages

            emit_event(EventType.LLM, f"[→LLM] ask_images_multi:{json.dumps(msgs, ensure_ascii=False)}")
            res = self.client.chat.completions.create(
                model="qwen-vl-max-latest",
                messages=msgs,
                temperature=0.0,
                seed=1234,
                extra_body={ "top_k": None }
            )

            answer = res.choices[0].message.content.strip()
            emit_event(EventType.LLM, f"[←LLM] {answer}")
            emit_event(EventType.LLM, f"[CostToken] {res.usage.total_tokens}")
            return answer
        except Exception as e:
            logger.exception(f"llmCaller.ask_images_multi error ! messages:{json.dumps(messages, ensure_ascii=False)}", e)
            return ""


    def stream_ask(self, question: str):
        try:
            if isinstance(question, str):
                question = question.strip()

            emit_event(EventType.LLM, f"[→LLM] question:\n{question}")
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': self.system_prompt},
                    {'role': 'user', 'content': question}
                ],
                stream=True,
                stream_options={"include_usage": True},
                temperature=0.0,
                seed=1234,
                extra_body={ "top_k": None }
            )
            for chunk in completion:
                if len(chunk.choices) == 0:
                    emit_event(EventType.LLM, f"[CostToken] {chunk.usage.total_tokens}")
                    break

                content = chunk.choices[0].delta.content.strip()
                yield content
        except Exception as e:
            logger.exception(f"llmCaller.stream_ask error ! question:{question}", e)
            yield ""

    def ask_tool(self,
                 messages: list[LlmMessage],
                 timeout: int = 300,
                 tools: Optional[List[dict]] = None,
                 **kwargs
        ) -> ChatCompletionMessage:
        """Ask LLM using functions/tools and return the response.

        :param messages: List of messages
        :param timeout: Timeout in seconds
        :param tools: List of tools to use
        :param kwargs: Additional arguments
        :return: LLM response
        """
        try:
            # Set up messages
            msgs = [
                {'role': 'system', 'content': self.system_prompt},
            ]
            for msg in messages:
                msgs.append(msg.to_dict())

            # Set up params
            params = {
                "model": self.model,
                "messages": msgs,
                "tools": tools,
                "tool_choice": "auto",
                "timeout": timeout,
                "temperature": 0.0,
                "seed": 1234,
                "extra_body": { "top_k": None },
                **kwargs,
            }

            # Call LLM to get response
            emit_event(EventType.LLM, f"[→LLM] {json.dumps(params, ensure_ascii=False)}")
            response: ChatCompletion = self.client.chat.completions.create(
                **params
            )

            # Check if response is invalid
            if not response.choices or not response.choices[0].message:
                print(response)
                return None

            answer = response.choices[0].message
            emit_event(EventType.LLM, f"[←LLM] {answer}")
            emit_event(EventType.LLM, f"[CostToken] {response.usage.total_tokens}")
            return answer

        except OpenAIError as oe:
            logger.error(f"OpenAI API error: {oe}")
            if isinstance(oe, AuthenticationError):
                logger.error("Authentication failed. Check API key.")
            elif isinstance(oe, RateLimitError):
                logger.error("Rate limit exceeded. Consider increasing retry attempts.")
            elif isinstance(oe, APIError):
                logger.error(f"API error: {oe}")
            raise
        except Exception as e:
            logger.exception(f"llmCaller.ask error ! messages:{messages}", e)
            raise


if __name__ == '__main__':
    llm = LlmCaller(system_prompt="你是一个用户助手用户回答用户问题")
    out_message = llm.ask_tool(
        messages=[
            LlmMessage(role=RoleType.USER, content="明天杭州的天气怎么样?")
        ],
        timeout=300,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "description": "当你想知道现在的时间时非常有用。",
                    "parameters": {}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "当你想查询指定城市的天气时非常有用。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "城市或县区，比如北京市、杭州市、余杭区等。"
                            }
                        },
                        "required": ["location"]
                    }
                }
            }
        ]
    )
    print(out_message)
