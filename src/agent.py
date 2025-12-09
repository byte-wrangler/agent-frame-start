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
    def execute(self) -> Message:
        """Execute the action"""

class ToolAction(Action):
    def __init__(self,
                 tools: Dict[str, Tool],
                 tool_name: str,
                 tool_args: Dict[str, object]):
        self.type = "tool_action"
        self.tools = tools
        self.tool_name = tool_name
        self.tool_args = tool_args

    def execute(self) -> Message:
        """Execute the action
        """
        if self.tools.get(self.tool_name) is None:
            raise ValueError(f"Tool {self.tool_name} does not exist")

        tool: Tool = self.tools.get(self.tool_name)
        tool_res = tool.execute(self.tool_args)
        return Message(MessageType.ACTION, f"工具: {self.tool_name}({self.tool_args}) 的执行结果是: {tool_res}")

class ThinkAction(Action):
    def __init__(self, content: str):
        self.type = "think"
        self.content = content

    def execute(self) -> Message:
        return Message(MessageType.NORMAL, self.content)


class FinalAction(Action):

    def __init__(self, content: str):
        self.type = "final"
        self.content = content

    def execute(self) -> Message:
        return Message(MessageType.DONE, self.content)


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

        if self._verbose:
            for message in messages:
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
# 历史的信息记录
{history_message}
"""
        latest_message_content = self.memory.get_latest_message_content()
        if latest_message_content is not None and len(latest_message_content) > 0:
            messages_content += f"""
# 当下的思考或工具执行结果
{self.memory.get_latest_message_content()}

不要重复历史推理，只关注当前思考与执行的推进。
"""
        llm_output = self.llm_caller.ask(messages_content)

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
        """Execute actions and return the output messages

        :return: a list of messages
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

