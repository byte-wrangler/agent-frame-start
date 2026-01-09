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
    """Agent 与环境交互的消息类"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    time: float = Field(default_factory=lambda: time.time())
    type: MessageType = Field(default=None)
    content: Any = Field(default=None)

    @classmethod
    def from_llm_message(cls, message: LlmMessage) -> "Message":
        """从 LLM 消息转换为 Agent 消息

        Args:
            message: LLM 消息对象

        Returns:
            Agent 消息对象
        """
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
        """转换为 LLM 消息格式

        Returns:
            LLM 消息对象
        """
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
        """判断两条消息是否相同

        Args:
            other: 另一条消息

        Returns:
            是否相同
        """
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
        """判断消息是否为空

        Returns:
            是否为空
        """
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
    """Agent 可执行的工具抽象类"""

    name: str
    description: str
    parameters: Optional[dict] = None

    def __call__(self, **kwargs) -> Any:
        """使用给定参数执行工具"""
        return self.execute(**kwargs)

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """使用给定参数执行工具

        Args:
            **kwargs: 工具参数

        Returns:
            工具执行结果
        """

    def to_param(self) -> Dict:
        """转换为函数调用格式

        Returns:
            函数调用参数字典
        """
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
    """终止工具 - 用于结束任务执行"""

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
        """完成当前执行

        Args:
            status: 完成状态（success 或 failure）

        Returns:
            完成消息
        """
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
    """推理动作 - 包含思考和工具调用"""

    def __init__(
            self,
            thought: str,
            tools: list[Tool],
            tool_calls: list[Union[dict, ChatCompletionMessageToolCall]]
    ):
        """初始化推理动作

        Args:
            thought: 思考内容（推理过程）
            tools: 可用工具集合
            tool_calls: 要调用的工具列表
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
        """执行推理动作

        Args:
            env: 交互环境

        Returns:
            包含执行结果的 ActionResult
        """
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
        """解析工具参数

        Args:
            tool_args: 工具参数（可能是字典或 JSON 字符串）

        Returns:
            解析后的参数字典
        """
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
    """Agent 的记忆类"""

    messages: List[Message] = Field(default_factory=list)
    max_messages: int = Field(default=100)

    def add_message(self, message: Message) -> None:
        """添加消息到记忆

        Args:
            message: 要添加的消息
        """
        self.messages.append(message)

        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

    def add_messages(self, messages: List[Message]) -> None:
        """批量添加消息到记忆

        Args:
            messages: 消息列表
        """
        self.messages.extend(messages)
        # 可选：实现消息数量限制
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

    def clear(self) -> None:
        """清空所有消息"""
        self.messages.clear()

    def get_recent_messages(self, n: int) -> List[Message]:
        """获取最近的 n 条消息

        Args:
            n: 消息数量

        Returns:
            最近的消息列表
        """
        return self.messages[-n:]

    def to_dict_list(self) -> List[dict]:
        """转换消息为字典列表

        Returns:
            字典列表
        """
        return [msg.to_dict() for msg in self.messages]


class Environment(ABC):
    """Agent 交互的环境类"""

    def __init__(self, initial_message: str = None):
        """初始化环境

        Args:
            initial_message: 初始消息内容（可选）
        """
        self.messages: list[Message] = []
        self.consumed: Dict[str, bool] = {}
        self.context = {}

        if initial_message is not None:
            self.add_message(Message(
                type=MessageType.USER,
                content=initial_message
            ))

    def add_message(self, message: Message):
        """添加消息到环境

        Args:
            message: 要添加的消息
        """
        self.messages.append(message)
        self.consumed[message.id] = False

    def add_messages(self, messages: list[Message]):
        """批量添加消息到环境

        Args:
            messages: 消息列表
        """
        for message in messages:
            self.add_message(message)

    def pull_messages(self) -> list[Message]:
        """拉取所有未消费的消息

        Returns:
            未消费的消息列表
        """
        unconsumed_messages: list[Message] = []
        for msg in self.messages:
            if self.consumed[msg.id] is not True:
                unconsumed_messages.append(msg)

        # 拉取后标记为已消费
        for unconsumed_message in unconsumed_messages:
            self.consumed[unconsumed_message.id] = True
        return unconsumed_messages

    def pull_one_message(self) -> Optional[Message]:
        """拉取一条未消费的消息

        Returns:
            未消费的消息，如果没有则返回 None
        """
        for msg in self.messages:
            if self.consumed[msg.id] is not True:
                self.consumed[msg.id] = True
                return msg
        return None

    def peek_message(self) -> Optional[Message]:
        """查看最新消息（不标记为已消费）

        Returns:
            最新的消息，如果没有则返回 None
        """
        return self.messages[-1] if len(self.messages) > 0 else None

    def peek_latest_not_empty_message(self, msg_type: MessageType) -> Optional[Message]:
        """查看指定类型的最新非空消息

        Args:
            msg_type: 消息类型

        Returns:
            最新的非空消息，如果没有则返回 None
        """
        for msg in reversed(self.messages):
            if msg.type == msg_type and not msg.is_empty():
                return msg
        return None

    def set_context_value(self, key, value):
        """设置上下文值

        Args:
            key: 键
            value: 值
        """
        self.context[key] = value


class AgentStatus(Enum):
    """Agent 状态枚举"""

    RUNNING = "running"  # 运行中
    IDLE = "idle"        # 空闲


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
    """ReAct Agent 智能体类"""

    def __init__(
        self,
        description: str,
        model: str = "qwen-max-latest",
        tools: Optional[list[Tool]] = None
    ):
        """初始化 ReAct Agent

        Args:
            description: Agent 描述
            model: 使用的 LLM 模型名称
            tools: 可用工具列表
        """

        self.llm_caller = LlmCaller(f"{description}\n\n{COT_PROMPT}", model)
        self.memory = Memory()
        self.tools = tools + [TerminateTool()] \
            if tools is not None \
            else [TerminateTool()]

    def run(self, env: Environment, max_steps: Optional[int] = 20) -> Message:
        """Agent 主循环（观察-推理-行动）

        执行流程：
        1. Observe: 从环境中观察消息
        2. Reason: 推理并决定下一步动作
        3. Act: 执行动作并将结果放回环境

        循环执行直到最终动作被执行

        Args:
            env: Agent 交互的环境
            max_steps: 最大执行步数，None 表示无限制

        Returns:
            最终消息
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
        """从环境中观察未读消息

        Args:
            env: 要观察的环境
        """
        messages = env.pull_messages()
        self.memory.add_messages(messages)

        # for message in messages:
        #     emit_event(
        #         EventType.AGENT,
        #         f"[Observe] [{message.type.value}] {message.content}"
        #     )

    def reason(self) -> Action:
        """推理并决定下一步动作

        Returns:
            要执行的动作
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
        """执行动作并将结果放入环境

        Args:
            action: 要执行的动作
            env: Agent 交互的环境
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
        """检查记忆中是否存在重复思考（卡住）

        Returns:
            如果卡住返回 True
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
        """处理卡住的情况

        Args:
            next_prompt: 下一个提示词

        Returns:
            包含卡住提示的新提示词
        """
        stuck_prompt = "已经发现你正在重复思考，请避免重复已经思考过的内容并尝试新的思考，如果思考结束请使用terminate工具"
        return f"{stuck_prompt}\n{next_prompt}"