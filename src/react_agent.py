import json
import time
import uuid
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Any, Dict, List, Union

from loguru import logger
from openai.types.chat import ChatCompletionMessageToolCall
from pydantic import BaseModel, Field

from .event import emit_event, EventType
from .llm import LlmCaller, RoleType, LlmMessage


class MessageType(Enum):
    """Message Type"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    DONE = "done"


class Message(BaseModel):
    """Message that agent uses to interact with environment"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    time: float = Field(default_factory=lambda: time.time())
    type: MessageType = Field(default=None)
    content: Any = Field(default=None)

    @classmethod
    def from_llm_message(cls, message: LlmMessage) -> "Message":
        if message.role == RoleType.ASSISTANT:
            return cls(
                type=MessageType.ASSISTANT,
                content={
                    "content": message.content,
                    "tool_calls": message.tool_calls,
                }
            )
        elif message.role == RoleType.USER:
            return cls(
                type=MessageType.USER,
                content=message.content
            )
        elif message.role == RoleType.SYSTEM:
            return cls(
                type=MessageType.SYSTEM,
                content=message.content
            )
        elif message.role == RoleType.TOOL:
            return cls(
                type=MessageType.TOOL,
                content={
                    "content": message.content,
                    "name": message.name,
                    "tool_call_id": message.tool_call_id,
                }
            )
        raise ValueError(f"Invalid role type: {message.role}")

    def to_llm_message(self) -> LlmMessage:
        if self.type == MessageType.ASSISTANT:
            return LlmMessage(
                role=RoleType.ASSISTANT,
                content=self.content.get("content"),
                tool_calls=self.content.get("tool_calls"),
            )
        elif self.type == MessageType.USER:
            return LlmMessage(
                role=RoleType.USER,
                content=self.content
            )
        elif self.type == MessageType.SYSTEM:
            return LlmMessage(
                role=RoleType.SYSTEM,
                content=self.content
            )
        elif self.type == MessageType.TOOL:
            return LlmMessage(
                role=RoleType.TOOL,
                content=self.content.get("content"),
                name=self.content.get("name"),
                tool_call_id=self.content.get("tool_call_id"),
            )
        raise ValueError(f"Invalid message type: {self.type}")


    def is_same(self, other: "Message") -> bool:
        if self.type != other.type:
            return False

        if self.type == MessageType.SYSTEM:
            return self.content and self.content == other.content
        elif self.type == MessageType.USER:
            return self.content and self.content == other.content
        elif self.type == MessageType.ASSISTANT:
            return self.content \
                and self.content.get('content') \
                and self.content.get('content') == other.content.get('content')
        elif self.type == MessageType.TOOL:
            return False
        elif self.type == MessageType.DONE:
            return self.content and self.content == other.content

        raise ValueError(f"Invalid message type: {self.type}")

    def is_empty(self) -> bool:
        if self.type == MessageType.SYSTEM:
            return self.content is None or self.content.strip() == ""
        elif self.type == MessageType.USER:
            return self.content is None or self.content.strip() == ""
        elif self.type == MessageType.ASSISTANT:
            return self.content is None or self.content.get('content', '').strip() == ""
        elif self.type == MessageType.TOOL:
            return False
        elif self.type == MessageType.DONE:
            return self.content is None


class Tool(ABC, BaseModel):
    """Tool for agent to execute"""

    name: str
    description: str
    parameters: Optional[dict] = None

    def __call__(self, **kwargs) -> Any:
        """Execute the tool with given parameters."""
        return self.execute(**kwargs)

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """Execute the tool with given parameters."""

    def to_param(self) -> Dict:
        """Convert tool to function call format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

TERMINATE_PROMPT = """Terminate the interaction when the request is met OR if the assistant cannot proceed further with the task.
When you have finished all the tasks, call this tool to end the work."""

class TerminateTool(Tool):
    name: str = "terminate"
    description: str = TERMINATE_PROMPT
    parameters: dict = {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "description": "The finish status of the interaction.",
                "enum": ["success", "failure"],
            }
        },
        "required": ["status"],
    }

    async def execute(self, status: str) -> str:
        """Finish the current execution"""
        return f"The interaction has been completed with status: {status}"


class ActionResult(BaseModel):
    """Action result"""
    success: bool
    value: Any
    error: Optional[str] = None


class Action(ABC):
    """Action to take by the agent"""

    @abstractmethod
    def execute(self, env: "Environment") -> ActionResult:
        raise NotImplementedError

    @abstractmethod
    def to_dict(self) -> dict:
        return {}

class ReasonAction(Action):
    def __init__(
            self,
            thought: str,
            tools: list[Tool],
            tool_calls: list[Union[
                dict,
                ChatCompletionMessageToolCall
            ]]
    ):
        """Reason action

        :param thought: Thought (reason content)
        :param tools: Tool set
        :param tool_calls: Tool calls of the action
        """
        self.type = "reason"
        self.thought = thought
        self.tools = tools
        self.tool_calls = tool_calls

        self.tools_map = {
            tool.name: tool
            for tool in tools
        }

    def execute(self, env: "Environment") -> ActionResult:
        try:
            messages = []

            thought_message = Message(
                type=MessageType.ASSISTANT,
                content={
                    "content": self.thought,
                    "tool_calls": self.tool_calls,
                }
            )
            messages.append(thought_message)
            emit_event(EventType.AGENT,
                       f"[Thought] {thought_message.content.get('content')}\n"
                       f"tool_calls:{thought_message.content.get('tool_calls')}")

            for tool_call in self.tool_calls:
                tool_name = tool_call.get('function', {}).get('name')
                tool: Tool = self.tools_map.get(tool_name)

                if tool is None:
                    logger.warning(f"Tool {tool_name} not found.")
                    continue

                if tool_name == "terminate":
                    terminate_message = Message(
                        type=MessageType.DONE,
                        content=f"{self.thought}"
                    )
                    emit_event(EventType.AGENT, f"[Terminate] {terminate_message.content}")
                    messages.append(terminate_message)
                    return ActionResult(
                        success=True,
                        value=messages
                    )

                tool_args = tool_call.get('function', {}).get('arguments')
                tool_args = self._parse_tool_args(tool_args)

                tool_res = tool.execute(**{
                    'env': env,
                    **tool_args
                })
                tool_message = Message(
                    type=MessageType.TOOL,
                    content={
                        "content": f"Observed output of Tool `{tool_name}({tool_args})` executed:\n{str(tool_res)}"
                                    if tool_res
                                    else f"Tool `{tool_name}({tool_args})` completed with no output",
                        "name": tool_name,
                        "tool_call_id": tool_call.get("id")
                    },
                )
                emit_event(EventType.AGENT, f"[Action] {tool_message.content}")
                messages.append(tool_message)

            return ActionResult(
                success=True,
                value=messages
            )
        except Exception as e:
            logger.exception(f"Tool execution failed: {e}")
            return ActionResult(
                success=False,
                value=None,
                error=str(e)
            )

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "thought": self.thought,
            "tool_calls": self.tool_calls,
        }

    @staticmethod
    def _parse_tool_args(tool_args) -> dict:
        if isinstance(tool_args, dict):
            return tool_args

        if isinstance(tool_args, str):
            try:
                return json.loads(tool_args)
            except json.JSONDecodeError as e:
                logger.exception(f"Tool arguments is not valid json: {tool_args}, error: {e}")

                try:
                    return eval(tool_args)
                except Exception as e:
                    logger.exception(f"Tool arguments is not valid: {tool_args}, error: {e}")
        return {}


class Memory(BaseModel):
    messages: List[Message] = Field(default_factory=list)
    max_messages: int = Field(default=100)

    def add_message(self, message: Message) -> None:
        """Add a message to memory"""
        self.messages.append(message)

        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

    def add_messages(self, messages: List[Message]) -> None:
        """Add multiple messages to memory"""
        self.messages.extend(messages)
        # Optional: Implement message limit
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

    def clear(self) -> None:
        """Clear all messages"""
        self.messages.clear()

    def get_recent_messages(self, n: int) -> List[Message]:
        """Get n most recent messages"""
        return self.messages[-n:]

    def to_dict_list(self) -> List[dict]:
        """Convert messages to list of dicts"""
        return [msg.to_dict() for msg in self.messages]


class Environment(ABC):
    """Environment that agent interact with"""

    def __init__(self, initial_message: str = None):
        self.messages: list[Message] = []
        self.consumed: Dict[str, bool] = {}
        self.context = {}

        if initial_message is not None:
            self.add_message(Message(
                type=MessageType.USER,
                content=initial_message
            ))

    def add_message(self, message: Message):
        self.messages.append(message)
        self.consumed[message.id] = False

    def add_messages(self, messages: list[Message]):
        for message in messages:
            self.add_message(message)

    def pull_messages(self) -> list[Message]:
        unconsumed_messages: list[Message] = []
        for msg in self.messages:
            if self.consumed[msg.id] is not True:
                unconsumed_messages.append(msg)

        # Once pulled from the environment,
        # the message is considered as consumed.
        for unconsumed_message in unconsumed_messages:
            self.consumed[unconsumed_message.id] = True
        return unconsumed_messages

    def pull_one_message(self) -> Optional[Message]:
        for msg in self.messages:
            if self.consumed[msg.id] is not True:
                self.consumed[msg.id] = True
                return msg
        return None

    def peek_message(self) -> Optional[Message]:
        return self.messages[-1] if len(self.messages) > 0 else None

    def peek_latest_not_empty_message(self, msg_type: MessageType) -> Optional[Message]:
        for msg in reversed(self.messages):
            if msg.type == msg_type and not msg.is_empty():
                return msg
        return None

    def set_context_value(self, key, value):
        self.context[key] = value


class AgentStatus(Enum):
    """Status of the agent"""

    RUNNING = "running"
    IDLE = "idle"


COT_PROMPT = """
Based on user needs, proactively select the most appropriate tool or combination of tools. 
For complex tasks, you can break down the problem and use different tools step by step to solve it.

1. Break down the problem: Divide complex problems into smaller, more manageable parts
2. Think step by step: Think through each part in detail, showing your reasoning process

* If You want to use tools, you have to clearly explain your plan and why you want to use tools. 
    After using each tool, clearly explain the execution results and suggest the next steps.
    You can use multiple tools in one step.
* If you do not need to use tool, you have to clearly explain your plan and suggests the next steps.
* If you think the task is already done, dont hesitate to use the `terminate` tool/function call to stop.
"""

FOCUS_PROMPT = """Do not repeat historical thinking/reasoning, and execution. 
Focus only on the progress of current thinking and execution.
"""

class ReActAgent(ABC):

    def __init__(
        self,
        description: str,
        model: str = "qwen-max-latest",
        tools: Optional[list[Tool]] = None
    ):
        """Defines the Reason Action Agent

        :param description: Description of the agent
        :param model: LLM model to be used
        :param tools: List of tools can be used
        """

        self.llm_caller = LlmCaller(f"{description}\n\n{COT_PROMPT}", model)
        self.memory = Memory()
        self.tools = tools + [TerminateTool()] \
            if tools is not None \
            else [TerminateTool()]

    def run(self,
            env: Environment,
            max_steps: Optional[int] = 20
        ) -> Message:
        """Main loop for the agent to run (Reason + Act)

        1. Observe: Observe messages from the environment.
        2. Thought: Think (reason) to make an action.
        3. Action: Execute the action and put result messages to environment.

        Run the loop until the final action is been executed.

        :param env: Environment that the agent interact with.
        :param max_steps: Max steps for the agent to run.
            No steps limitation if None is given.
        :return: the final message
        """
        emit_event(
            EventType.AGENT,
            f"[Observe] [{env.peek_message().type.value}] "
            f"{env.peek_message().content}"
        )

        current_step = 0
        while True:
            self.observe(env)
            action = self.reason()
            self.act(action, env)

            latest_message: Message = env.peek_message()
            if latest_message and latest_message.type == MessageType.DONE:
                return env.peek_latest_not_empty_message(MessageType.ASSISTANT)

            current_step += 1
            if max_steps is not None and current_step >= max_steps:
                return latest_message

    def observe(self, env):
        """Observe unread messages from environment.

        :param env: Environment to observe
        """
        messages = env.pull_messages()
        self.memory.add_messages(messages)

        # for message in messages:
        #     emit_event(
        #         EventType.AGENT,
        #         f"[Observe] [{message.type.value}] {message.content}"
        #     )

    def reason(self) -> Action:
        """Reason to make an action

        :return: Action to take
        """
        # Recall memories
        llm_messages = []
        for message in self.memory.messages:
            llm_messages.append(
                message.to_llm_message()
            )

        # Focus on the latest message
        focus_prompt = FOCUS_PROMPT
        # Check if stuck
        if self._is_stuck():
            focus_prompt = self._handle_stuck(focus_prompt)

        llm_messages.append(
            Message(
                type=MessageType.USER,
                content=focus_prompt
            ).to_llm_message()
        )

        # Call LLM to reason
        llm_response = self.llm_caller.ask_tool(
            messages=llm_messages,
            timeout=300,
            tools=[tool.to_param() for tool in self.tools]
        )

        return ReasonAction(
            thought=llm_response.content,
            tools=self.tools,
            tool_calls=[tool_call.model_dump()
                        for tool_call in llm_response.tool_calls]
                        if llm_response.tool_calls is not None
                        else []
        )

    def act(self, action: Action, env: Environment):
        """Execute the action and put results to the environment.

        :param action: Action to be executed.
        :param env: Environment the agent interact with.
        """
        # Execute the action
        action_result = action.execute(env)

        # Put action result into environment
        if not action_result.success or not action_result.value:
            env.add_message(Message(
                type=MessageType.USER,
                content=f"Action executed error because of {action_result.error}. The action is {action.to_dict()}"
            ))
            return

        env.add_messages(action_result.value)

    def _is_stuck(self):
        """Check if there are repeated thinking in memory (Stuck).

        :return: true if stuck
        """
        if len(self.memory.messages) < 2:
            return False

        last_message = self.memory.messages[-1]
        duplicate_count = 0
        for message in reversed(self.memory.messages[:-1]):
            if message.type == MessageType.ASSISTANT\
                    and last_message.is_same(message):
                duplicate_count += 1

        return duplicate_count > 0

    def _handle_stuck(self, next_prompt: str) -> str:
        """Handle stuck situation"""

        # stuck_prompt = ("Observed duplicate responses. Consider new strategies "
        #                 "and avoid repeating ineffective paths already attempted.")

        stuck_prompt = "已经发现你正在重复思考，请避免重复已经思考过的内容并尝试新的思考，如果思考结束请使用terminate工具"
        return f"{stuck_prompt}\n{next_prompt}"