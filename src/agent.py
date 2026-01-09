import json
import time
import uuid
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Dict

from .event import emit_event, EventType
from .llm import LlmCaller


class MessageType(Enum):
    NORMAL = 1
    ACTION = 2
    DONE = 3

class Message:
    """Agent 与环境通信的消息类"""

    def __init__(self, mtype: MessageType, content: str):
        """初始化消息

        Args:
            mtype: 消息类型
            content: 消息内容
        """
        self.id = str(uuid.uuid4())
        self.time = time.time()
        self.mtype = mtype
        self.content = content

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id

class Tool(ABC):
    """Agent 可使用的工具抽象类"""

    @abstractmethod
    def get_schema(self):
        """获取工具的描述信息（名称、参数等）"""
        pass

    @abstractmethod
    def execute(self, kwargs) -> str:
        """执行工具的具体逻辑

        Args:
            kwargs: 工具参数字典

        Returns:
            工具执行结果
        """
        pass


class Action(ABC):
    """Agent 执行的动作抽象类"""

    @abstractmethod
    def execute(self) -> Message:
        """执行动作并返回消息"""
        pass

class ToolAction(Action):
    """工具调用动作"""

    def __init__(self, tools: Dict[str, Tool], tool_name: str, tool_args: Dict[str, object]):
        """初始化工具动作

        Args:
            tools: 可用工具字典
            tool_name: 要使用的工具名称
            tool_args: 工具参数
        """
        self.type = "tool_action"
        self.tools = tools
        self.tool_name = tool_name
        self.tool_args = tool_args

    def execute(self) -> Message:
        """执行工具调用

        Returns:
            包含工具执行结果的消息

        Raises:
            ValueError: 当工具不存在时
        """
        if self.tools.get(self.tool_name) is None:
            raise ValueError(f"Tool {self.tool_name} does not exist")

        tool: Tool = self.tools.get(self.tool_name)
        tool_res = tool.execute(self.tool_args)
        return Message(MessageType.ACTION, f"工具: {self.tool_name}({self.tool_args}) 的执行结果是: {tool_res}")

class ThinkAction(Action):
    """思考动作"""

    def __init__(self, content: str):
        """初始化思考动作

        Args:
            content: 思考内容
        """
        self.type = "think"
        self.content = content

    def execute(self) -> Message:
        """执行思考动作

        Returns:
            包含思考内容的普通消息
        """
        return Message(MessageType.NORMAL, self.content)


class FinalAction(Action):
    """最终答案动作"""

    def __init__(self, content: str):
        """初始化最终动作

        Args:
            content: 最终答案内容
        """
        self.type = "final"
        self.content = content

    def execute(self) -> Message:
        """执行最终动作

        Returns:
            包含最终答案的完成消息
        """
        return Message(MessageType.DONE, self.content)


class Environment(ABC):
    """Agent 交互的环境类"""

    def __init__(self, initial_message: str = None):
        """初始化环境

        Args:
            initial_message: 初始消息内容（可选）
        """
        self.messages: list[Message] = []
        self.consumed: Dict[str, bool] = {}

        if initial_message is not None:
            self.add_message(Message(MessageType.NORMAL, initial_message))

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

class Agent(ABC):
    """Agent 智能体抽象类"""

    TOOL_USER_AGENT_DESC = """
    ======
    工具使用、思考过程与输出格式：
    1. 你所拥有的工具包括：%s
    2. 无论给定任何问题，请一步一步思考并执行
    3. 你需要根据用户的问题做出判断: 是否需要继续思考、或者使用工具、或者给出最终回复
    4. 如果你要使用工具时，请直接输出工具名和工具的入参，不需要输出其他内容，格式为：call_tool|工具名|工具入参|使用该工具的理由，你每次只可以使用一个工具；
       你可以使用同一个工具多次，但是尽量减少使用同一个工具的次数，同一个工具最多不要连续使用超过5次
    5. 如果不需要使用工具，但该问题尚未结束，你可以继续思考，输出格式为：continue_think|具体的思考内容
    6. 如果不需要使用工具，并且判断该问题已经结束，输出格式为：final_result|具体返回结果
    
    使用工具返回示例:
    call_tool|self_defined_tool|{"content": "data"}|我想通过该工具实现xxx
    
    不使用工具继续思考示例:
    continue_think|I want to do something
    
    不使用工具做出最终回复的返回示例:
    final_result|This is the final result
    """

    def __init__(self, desc: str, model: str = "qwen-max", tools: dict = None, verbose: bool = True):
        """初始化 Agent

        Args:
            desc: Agent 描述
            model: 使用的 LLM 模型名称
            tools: 可用工具字典
            verbose: 是否输出详细日志
        """
        self.memory = Memory()
        self.tools = tools or {}
        self.llm_caller = LlmCaller(self._make_sys_prompt(desc), model)
        self.env: Optional[Environment] = None
        self._verbose = verbose

    def run(self, env: Environment) -> Message:
        """Agent 主循环（观察-推理-行动）

        执行流程：
        1. Observe: 从环境中观察消息
        2. Reason: 思考并决定下一步动作
        3. Act: 执行动作并将结果放回环境

        Args:
            env: 交互环境

        Returns:
            最终完成消息
        """
        self.env = env
        while True:
            self.observe(env)
            actions = self.reason()
            out_messages = self.act(actions)
            env.add_messages(out_messages)

            last_message = env.peek_message()
            if last_message is not None and last_message.mtype == MessageType.DONE:
                self.env = None
                return last_message

    def observe(self, env: Environment) -> list[Message]:
        """从环境中观察消息

        Args:
            env: 交互环境

        Returns:
            观察到的消息列表
        """
        messages = env.pull_messages()

        # 保存到记忆中
        for message in messages:
            self.memory.save_in_memory(message)

        if self._verbose:
            for message in messages:
                emit_event(EventType.AGENT, f"[Observe] type={message.mtype} content={message.content}")
        return messages

    def reason(self) -> list[Action]:
        """推理并决定下一步动作

        使用 LLM 分析当前情况，决定是使用工具、继续思考还是给出最终答案

        Returns:
            要执行的动作列表
        """
        # 使用 LLM 进行推理
        first_message = "" if self.memory.is_empty() else self.memory.get_first_message().content
        messages_content = f"""# 原始问题\n{first_message}\n"""

        # 添加历史信息记录
        history_message = self.memory.recall()
        if history_message is not None and len(history_message) > 0:
            messages_content += f"\n# 历史的信息记录\n{history_message}\n"

        # 添加当前思考或工具执行结果
        latest_message_content = self.memory.get_latest_message_content()
        if latest_message_content is not None and len(latest_message_content) > 0:
            messages_content += f"\n# 当下的思考或工具执行结果\n{self.memory.get_latest_message_content()}\n\n不要重复历史推理，只关注当前思考与执行的推进。\n"

        llm_output = self.llm_caller.ask(messages_content)

        # 解析 LLM 输出并生成动作
        actions = []
        if llm_output is not None and llm_output.startswith('call_tool'):
            tool_calls = llm_output.split("\n")
            for tool_call in tool_calls:
                tool_name = tool_call.split("|")[1]
                tool_args = json.loads(tool_call.split("|")[2])
                actions.append(ToolAction(self.tools, tool_name, tool_args))
                emit_event(EventType.AGENT, f"[Thought] I need {llm_output}")

        elif llm_output is not None and llm_output.startswith('continue_think'):
            content = '|'.join(llm_output.split("|")[1:])
            actions.append(ThinkAction(content))
            emit_event(EventType.AGENT, f"[Thought] {llm_output}")

        elif llm_output is not None and llm_output.startswith('final_result'):
            content = '|'.join(llm_output.split("|")[1:])
            actions.append(FinalAction(content))
            emit_event(EventType.AGENT, f"[Thought] Export final message: {llm_output}")

        elif llm_output is not None:
            actions.append(FinalAction(llm_output))
            emit_event(EventType.AGENT, f"[Thought] Export final message: {llm_output}")

        return actions

    def act(self, actions: list[Action]) -> list[Message]:
        """执行动作并返回输出消息

        Args:
            actions: 要执行的动作列表

        Returns:
            执行结果消息列表
        """
        out_messages = []
        for action in actions:
            if isinstance(action, FinalAction):
                final_message = action.execute()
                out_messages.append(final_message)

                if self._verbose:
                    emit_event(EventType.AGENT, f"[FinalMessage] {final_message.content}")
                return out_messages

            elif isinstance(action, ToolAction):
                out_message = action.execute()
                out_messages.append(out_message)
                if self._verbose:
                    emit_event(EventType.AGENT, f"[Action] Action executed: tool_name={action.tool_name} "
                                f"tool_args={action.tool_args} "
                                f"out_message={out_message.mtype}:{out_message.content}")

            elif isinstance(action, ThinkAction):
                out_message = action.execute()
                out_messages.append(out_message)
        return out_messages

    def _make_sys_prompt(self, desc: str) -> str:
        """构建系统提示词

        Args:
            desc: Agent 描述

        Returns:
            完整的系统提示词
        """
        return (desc + "\n" +
                Agent.TOOL_USER_AGENT_DESC % str([tool.get_schema() for tool in self.tools.values()]))


class Memory:
    """Agent 的短期记忆类"""

    def __init__(self):
        """初始化记忆"""
        self.history: list[Message] = []

    def save_in_memory(self, message: Message):
        """保存消息到记忆中

        Args:
            message: 要保存的消息
        """
        self.history.append(message)

    def recall(self, n: int = 2):
        """回忆历史消息（排除首尾消息）

        Args:
            n: 回忆的消息数量（当前实现返回所有中间消息）

        Returns:
            格式化的历史消息字符串
        """
        if len(self.history[1:-1]) == 0:
            return ''
        return '- ' + '\n- '.join([msg.content for msg in self.history[1:-1]])

    def get_latest_message_content(self) -> str:
        """获取最新消息内容

        Returns:
            最新消息的内容，如果记忆为空则返回空字符串
        """
        return self.history[-1].content if len(self.history) > 1 else ''

    def get_first_message(self):
        """获取第一条消息（通常是原始问题）

        Returns:
            第一条消息，如果记忆为空则返回 None
        """
        return self.history[0] if len(self.history) > 0 else None

    def is_empty(self):
        """检查记忆是否为空

        Returns:
            如果记忆为空返回 True，否则返回 False
        """
        return len(self.history) == 0

