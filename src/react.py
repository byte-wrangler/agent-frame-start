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
    """Message that agent used to talk to env"""

    def __init__(self,
                 mtype: MessageType,
                 content: str):
        self.id = str(uuid.uuid4())
        self.time = time.time()
        self.mtype = mtype
        self.content = content

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id


class Tool(ABC):
    """Tool that an agent can use"""

    @abstractmethod
    def get_schema(self):
        pass

    @abstractmethod
    def execute(self, kwargs) -> str:
        pass


class Action(ABC):
    """Action to executed by an agent"""

    @abstractmethod
    def execute(self) -> list[Message]:
        """Execute the action"""


class ReasonAction(Action):

    type = "reason"

    def __init__(self, thought: str, tools: dict[str, Tool], tool_calls: list):
        self.type = ReasonAction.type
        self.thought = thought
        self.tools = tools
        self.tool_calls = tool_calls

    def execute(self) -> list[Message]:
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

    type = "final"

    def __init__(self, content: str):
        self.type = FinalAction.type
        self.content = content

    def execute(self) -> list[Message]:
        return [Message(MessageType.DONE, self.content)]


class Environment(ABC):
    """Environment that agents talk to"""

    def __init__(self, initial_message: str = None):
        self.messages: list[Message] = []
        self.consumed: Dict[str, bool] = {}

        if initial_message is not None:
            self.add_message(Message(MessageType.NORMAL, initial_message))

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


class Agent(ABC):
    """Agent abstraction"""
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

    def __init__(self,
                 desc: str,
                 model: str = "qwen-max",
                 tools: dict = None,
                 verbose: bool = True):
        self.memory = Memory()
        self.tools = tools or {}
        self.llm_caller = LlmCaller(self._make_sys_prompt(desc), model)
        self.env: Optional[Environment] = None

        self._verbose = verbose

    def run(self, env: Environment) -> Message:
        """Main loop for the agent to run (Reason + Act)

        1. Observe: Observe the message from the environment
        2. Thought: Think (Reason) to make an action
        3. Action: Execute the action and put the result to Environment
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
        """Observe a message from the environment

        :return: a message
        """
        messages = env.pull_messages()

        # Save in memory
        for message in messages:
            self.memory.save_in_memory(message)
            emit_event(EventType.AGENT, f"[Observe] type={message.mtype} content={message.content}")
        return messages

    def reason(self) -> list[Action]:
        """Observe the message in the environment,
        and determine which action to be executed.

        :return: a set of actions executed
        """

        # Use LLM to reason
        first_message = "" if self.memory.is_empty() else self.memory.get_first_message().content
        messages_content = f"""# 原始问题\n{first_message}\n"""

        history_message = self.memory.recall()
        if history_message is not None and len(history_message) > 0:
            messages_content += f"""
# 历史思考推理与执行过程
**不要重复历史思考推理与执行，只关注当前思考与执行的推进。**
{history_message}
"""
        latest_message_content = self.memory.get_latest_message_content()
        if latest_message_content is not None and len(latest_message_content) > 0:
            messages_content += f"""
# 当下的思考或工具执行结果
{self.memory.get_latest_message_content()}

"""
        llm_output = self.llm_caller.ask(messages_content)

        llm_output = llm_output.replace("`", "")\
                               .replace("json", "").strip()

        # Final action
        if llm_output.startswith("final"):
            content = llm_output.split("|")[1]
            action = FinalAction(content)
            emit_event(EventType.AGENT, f"[Thought] Export final message: {llm_output}")
            return [action]

        # Reason action
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
        """Execute actions and return the output messages

        :return: a list of messages
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
        return (desc + "\n" +
                Agent.TOOL_USER_AGENT_DESC % str([tool.get_schema() for tool in self.tools.values()]))


class Memory:
    """Short-term memory of agent"""

    def __init__(self):
        self.history: list[Message] = []

    def save_in_memory(self, message: Message):
        """Save a message to Memory"""
        self.history.append(message)

    def recall(self, n: int = 2):
        """Recall latest messages in memory

        :param n: number of messages to recall
        :return: latest n messages
        """
        if len(self.history[1:-1]) == 0:
            return ''
        return '- ' + '\n- '.join([msg.content for msg in self.history[1:-1]])

    def get_latest_message_content(self) -> str:
        return self.history[-1].content if len(self.history) > 1 else ''

    def get_first_message(self):
        return self.history[0] if len(self.history) > 0 else None

    def is_empty(self):
        return len(self.history) == 0

