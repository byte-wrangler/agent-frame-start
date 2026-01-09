import json
import time
import uuid
from typing import Optional, Dict
from abc import ABC, abstractmethod
from enum import Enum

from .event import emit_event, EventType
from .llm import LlmCaller


class MessageType(Enum):
    NORMAL = 1
    REASON = 2
    TOOL_EXECUTED = 3
    DONE = 4


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
    def execute(self) -> list[Message]:
        """执行动作并返回消息列表"""
        pass


class ReasonAction(Action):
    """推理动作（包含思考和工具调用）"""

    type = "reason"

    def __init__(self, thought: str, tools: dict[str, Tool], tool_calls: list):
        """初始化推理动作

        Args:
            thought: 推理思考内容
            tools: 可用工具字典
            tool_calls: 要调用的工具列表
        """
        self.type = ReasonAction.type
        self.thought = thought
        self.tools = tools
        self.tool_calls = tool_calls

    def execute(self) -> list[Message]:
        """执行推理动作

        Returns:
            包含推理内容和工具执行结果的消息列表
        """
        msgs = [Message(MessageType.REASON, self.thought)]
        for tool_call in self.tool_calls:
            tool_name: str = tool_call.get("tool_name", None)
            tool_args = tool_call.get("tool_args", {})
            if isinstance(tool_args, str):
                tool_args = json.loads(tool_args)

            if self.tools.get(tool_name) is None:
                continue

            tool: Tool = self.tools.get(tool_name)
            tool_res = tool.execute(tool_args)
            tool_executed_message = Message(MessageType.TOOL_EXECUTED, f"工具: {tool_name}({tool_args}) 的执行结果是: {tool_res}")
            msgs.append(tool_executed_message)
        return msgs

class FinalAction(Action):
    """最终答案动作"""

    type = "final"

    def __init__(self, content: str):
        """初始化最终动作

        Args:
            content: 最终答案内容
        """
        self.type = FinalAction.type
        self.content = content

    def execute(self) -> list[Message]:
        """执行最终动作

        Returns:
            包含最终答案的完成消息列表
        """
        return [Message(MessageType.DONE, self.content)]


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
    """ReAct Agent 智能体抽象类"""

    TOOL_USER_AGENT_DESC = """
    ======
    工具使用、思考过程与输出格式：
    1. 你所拥有的工具包括：%s
    2. 无论给定任何问题，请一步一步思考并执行
    3. 不要重复历史思考推理与执行，只关注当前思考与执行的推进。
    4. 你需要根据用户的问题给出回复，你只能选择这两种回复: ①思考推理的回复、②给出最终回复
      ① 思考推理回复：如果该问题尚未结束，请尽可能给出详细思考推理的回复，在推理中如果需要使用工具可以包含工具的调用。
      如果多次调用工具之间没有先后依赖，可以一次使用多个工具，反之应该获取上一个工具结果后再调用下一个工具，输出格式为JSON格式，不要输出任何其他内容: 
        {
            "type": "reason",
            "content": "详细的推理的内容",
            "tool_calls": [
                {"tool_name": "工具1名称", "tool_args": {具体的工具入参}},
                {"tool_name": "工具2名称", "tool_args": {具体的工具入参}},
            ]
        }
      ② 最终回复：如果判断该问题已经结束，给出最终的回复，输出格式为: final|最终回复的内容，不要输出任务其他内容
      
    5. 返回的示例：  
    当前问题尚未结束，思考推理回复（不使用工具）示例:
    {
        "type": "reason",
        "content": "根据当前给定的信息，我判断xxx",
    }
    
    当前问题尚未结束，思考推理回复（使用工具）示例:
    {
        "type": "reason",
        "content": "根据当前给定的信息，我判断xxx，需要调用工具",
        "tool_calls": [
            {"tool_name": "工具1名称", "tool_args": {具体的工具入参}},
            {"tool_name": "工具2名称", "tool_args": {具体的工具入参}},
        ]
    }
    
    当前问题结束，给出最终回复示例：
    final|待回答的内容
    """

    def __init__(self, desc: str, model: str = "qwen-max", tools: dict = None, verbose: bool = True):
        """初始化 ReAct Agent

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
        2. Reason: 推理并决定下一步动作
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
            emit_event(EventType.AGENT, f"[Observe] type={message.mtype} content={message.content}")
        return messages

    def reason(self) -> list[Action]:
        """推理并决定下一步动作

        使用 LLM 分析当前情况，决定是继续推理、使用工具还是给出最终答案

        Returns:
            要执行的动作列表
        """
        # 使用 LLM 进行推理
        first_message = "" if self.memory.is_empty() else self.memory.get_first_message().content
        messages_content = f"""# 原始问题\n{first_message}\n"""

        # 添加历史推理过程
        history_message = self.memory.recall()
        if history_message is not None and len(history_message) > 0:
            messages_content += f"\n# 历史思考推理与执行过程\n**不要重复历史思考推理与执行，只关注当前思考与执行的推进。**\n{history_message}\n"

        # 添加当前思考或工具执行结果
        latest_message_content = self.memory.get_latest_message_content()
        if latest_message_content is not None and len(latest_message_content) > 0:
            messages_content += f"\n# 当下的思考或工具执行结果\n{self.memory.get_latest_message_content()}\n\n"

        llm_output = self.llm_caller.ask(messages_content)

        # 清理 LLM 输出
        llm_output = llm_output.replace("`", "").replace("json", "").strip()

        # 解析最终答案动作
        if llm_output.startswith("final"):
            content = llm_output.split("|")[1]
            action = FinalAction(content)
            emit_event(EventType.AGENT, f"[Thought] Export final message: {llm_output}")
            return [action]

        # 解析推理动作
        try:
            action_output = json.loads(llm_output)
        except json.JSONDecodeError:
            action_output = eval(llm_output)

        if action_output.get('type') == ReasonAction.type:
            action = ReasonAction(
                thought=action_output.get('content'),
                tools=self.tools,
                tool_calls=action_output.get('tool_calls', {}),
            )
            emit_event(EventType.AGENT, f"[Thought] {action_output.get('content')}\n"
                                        f"调用工具: {action_output.get('tool_calls', {})}")
            return [action]

        return []


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
                final_messages = action.execute()
                emit_event(EventType.AGENT, f"[FinalMessage] {final_messages[0].content}")
                out_messages.extend(final_messages)

            elif isinstance(action, ReasonAction):
                action_messages = action.execute()

                for action_message in action_messages:
                    if action_message.mtype == MessageType.TOOL_EXECUTED:
                        emit_event(EventType.AGENT, f"[Action] Action executed: {action_message.content}")
                out_messages.extend(action_messages)
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

