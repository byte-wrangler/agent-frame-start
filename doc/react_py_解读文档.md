# react.py 文件解读文档

## 📚 文档概述

本文档旨在帮助不同读者理解 `react.py` 文件的核心概念和实现逻辑。实现了一个基于 **ReAct（Reasoning + Acting）** 范式的 Agent 框架，是 `agent.py` 的改进版本，采用了更简洁的推理-行动模式。

---

## 🎯 什么是 ReAct？

**ReAct = Reasoning（推理） + Acting（行动）**

ReAct 是一种让 AI 更像人类思考的方法：
1. **先思考**：分析问题，制定计划
2. **再行动**：根据思考结果执行工具调用
3. **循环迭代**：根据行动结果继续思考，直到解决问题

与 `agent.py` 的区别：
- `agent.py`：思考和行动是分离的（ThinkAction、ToolAction）
- `react.py`：思考和行动合并在一起（ReasonAction 同时包含思考内容和工具调用）

---

## 📦 主要组件详解

### 一、消息系统（Message）

#### 1.1 MessageType（消息类型枚举）

```python
class MessageType(Enum):
    NORMAL = 1          # 普通消息（如用户问题）
    REASON = 2          # 推理消息（Agent 的思考过程）
    TOOL_EXECUTED = 3   # 工具执行结果消息
    DONE = 4            # 任务完成消息
```

**与 agent.py 的区别**：
- 新增了 `REASON` 类型，专门用于记录推理过程
- 新增了 `TOOL_EXECUTED` 类型，区分工具执行结果
- 去掉了 `ACTION` 类型，因为推理和行动合并了

#### 1.2 Message（消息类）

```python
class Message:
    def __init__(self, mtype: MessageType, content: str):
        self.id = str(uuid.uuid4())      # 唯一标识符
        self.time = time.time()          # 创建时间戳
        self.mtype = mtype               # 消息类型
        self.content = content           # 消息内容
```

**作用**：封装 Agent 与环境之间传递的信息，与 `agent.py` 中的 Message 类完全相同。

---

### 二、工具系统（Tool）

```python
class Tool(ABC):
    @abstractmethod
    def get_schema(self):
        """返回工具的描述信息（名称、参数等）"""
        pass

    @abstractmethod
    def execute(self, kwargs) -> str:
        """执行工具的具体逻辑"""
        pass
```

**作用**：与 `agent.py` 完全相同，定义工具的标准接口。

---

### 三、动作系统（Action）- 核心改进

ReAct 模式下只有两种动作类型：

#### 3.1 ReasonAction（推理动作）- 核心创新

```python
class ReasonAction(Action):
    def __init__(self, thought: str, tools: dict[str, Tool], tool_calls: list):
        self.thought = thought        # 推理的思考内容
        self.tools = tools            # 可用工具字典
        self.tool_calls = tool_calls  # 要调用的工具列表
```

**核心特点**：
1. **合并推理和行动**：一个 Action 同时包含思考内容和工具调用
2. **支持批量工具调用**：`tool_calls` 是一个列表，可以一次调用多个工具
3. **灵活性**：`tool_calls` 可以为空，表示只思考不调用工具

**执行流程**：

```python
def execute(self) -> list[Message]:
    # 1. 先创建推理消息
    msgs = [Message(MessageType.REASON, self.thought)]
    
    # 2. 依次执行所有工具调用
    for tool_call in self.tool_calls:
        tool_name = tool_call.get("tool_name", None)
        tool_args = tool_call.get("tool_args", {})
        
        # 处理参数（可能是字符串形式的 JSON）
        if isinstance(tool_args, str):
            tool_args = json.loads(tool_args)
        
        # 检查工具是否存在
        if self.tools.get(tool_name) is None:
            continue
        
        # 执行工具
        tool = self.tools.get(tool_name)
        tool_res = tool.execute(tool_args)
        
        # 创建工具执行结果消息
        tool_executed_message = Message(
            MessageType.TOOL_EXECUTED, 
            f"工具: {tool_name}({tool_args}) 的执行结果是: {tool_res}"
        )
        msgs.append(tool_executed_message)
    
    return msgs
```

**返回值**：返回一个消息列表，包含：
- 1 条推理消息（REASON）
- N 条工具执行结果消息（TOOL_EXECUTED）

#### 3.2 FinalAction（最终动作）

```python
class FinalAction(Action):
    def __init__(self, content: str):
        self.content = content  # 最终答案
    
    def execute(self) -> list[Message]:
        return [Message(MessageType.DONE, self.content)]
```

**作用**：标记任务完成，返回最终结果。与 `agent.py` 类似，但返回的是消息列表。

---

### 四、环境系统（Environment）

Environment 类与 `agent.py` 完全相同，包括：
- `add_message(message)`：添加消息
- `add_messages(messages)`：批量添加消息
- `pull_messages()`：拉取所有未消费的消息
- `pull_one_message()`：拉取一条未消费的消息
- `peek_message()`：查看最新消息（不标记为已消费）

---

### 五、Agent（智能体核心）- ReAct 实现

#### 5.1 Agent 的工作流程

```
┌─────────────────────────────────────┐
│         Agent 主循环 (run)          │
└─────────────────────────────────────┘
              │
              ▼
    ┌──────────────────┐
    │  1. Observe      │  从环境中观察消息
    │  （观察）         │  - 拉取新消息
    └──────────────────┘  - 保存到记忆
              │
              ▼
    ┌──────────────────┐
    │  2. Reason       │  推理下一步动作
    │  （推理）         │  - 使用 LLM 分析
    └──────────────────┘  - 生成 ReasonAction 或 FinalAction
              │
              ▼
    ┌──────────────────┐
    │  3. Act          │  执行动作
    │  （行动）         │  - 执行 ReasonAction（思考+工具调用）
    └──────────────────┘  - 或执行 FinalAction（返回结果）
              │
              ▼
    ┌──────────────────┐
    │  4. 更新环境      │  将执行结果放回环境
    └──────────────────┘  - 推理消息
              │            - 工具执行结果消息
              ▼
         是否完成？
         /        \
       是          否
       │           │
       ▼           └──► 返回步骤 1
    返回结果
```

#### 5.2 核心方法详解

##### observe（观察）

```python
def observe(self, env: Environment) -> list[Message]:
    messages = env.pull_messages()  # 从环境拉取消息
    
    # 保存到记忆并输出日志
    for message in messages:
        self.memory.save_in_memory(message)
        emit_event(EventType.AGENT, f"[Observe] type={message.mtype} content={message.content}")
    
    return messages
```

**作用**：获取环境中的新信息并保存到记忆中。

##### reason（推理）- ReAct 核心

```python
def reason(self) -> list[Action]:
    # 1. 构建提示词
    first_message = self.memory.get_first_message().content
    messages_content = f"# 原始问题\n{first_message}\n"
    
    # 2. 添加历史信息
    history_message = self.memory.recall()
    if history_message:
        messages_content += f"""
# 历史思考推理与执行过程
**不要重复历史思考推理与执行，只关注当前思考与执行的推进。**
{history_message}
"""
    
    # 3. 添加最新消息
    latest_message_content = self.memory.get_latest_message_content()
    if latest_message_content:
        messages_content += f"""
# 当下的思考或工具执行结果
{latest_message_content}
"""
    
    # 4. 调用 LLM
    llm_output = self.llm_caller.ask(messages_content)
    
    # 5. 清理输出（去除 markdown 代码块标记）
    llm_output = llm_output.replace("`", "").replace("json", "").strip()
    
    # 6. 解析输出，生成动作
    if llm_output.startswith("final"):
        # 最终回复：final|答案内容
        content = llm_output.split("|")[1]
        return [FinalAction(content)]
    
    # 推理回复：JSON 格式
    try:
        action_output = json.loads(llm_output)
    except json.JSONDecodeError:
        action_output = eval(llm_output)
    
    if action_output.get('type') == 'reason':
        return [ReasonAction(
            thought=action_output.get('content'),
            tools=self.tools,
            tool_calls=action_output.get('tool_calls', []),
        )]
    
    return []
```

**LLM 输出格式**：

1. **推理回复（JSON 格式）**：
```json
{
    "type": "reason",
    "content": "详细的推理内容",
    "tool_calls": [
        {"tool_name": "工具1名称", "tool_args": {"参数1": "值1"}},
        {"tool_name": "工具2名称", "tool_args": {"参数2": "值2"}}
    ]
}
```

2. **最终回复（文本格式）**：
```
final|这是最终答案
```

**关键特性**：
- 支持一次调用多个工具（并行工具调用）
- `tool_calls` 可以为空（只思考不调用工具）
- 自动清理 LLM 输出中的 markdown 标记

##### act（行动）

```python
def act(self, actions: list[Action]) -> list[Message]:
    out_messages = []
    
    for action in actions:
        if isinstance(action, FinalAction):
            # 执行最终动作
            final_messages = action.execute()
            emit_event(EventType.AGENT, f"[FinalMessage] {final_messages[0].content}")
            out_messages.extend(final_messages)
        
        elif isinstance(action, ReasonAction):
            # 执行推理动作（包含工具调用）
            action_messages = action.execute()
            
            # 输出工具执行日志
            for action_message in action_messages:
                if action_message.mtype == MessageType.TOOL_EXECUTED:
                    emit_event(EventType.AGENT, f"[Action] Action executed: {action_message.content}")
            
            out_messages.extend(action_messages)
    
    return out_messages
```

**作用**：执行推理阶段决定的动作，并返回执行结果消息列表。

---

### 六、记忆系统（Memory）

Memory 类与 `agent.py` 完全相同：

```python
class Memory:
    def __init__(self):
        self.history = []  # 消息历史记录
    
    def save_in_memory(self, message: Message):
        """保存消息到记忆"""
        self.history.append(message)
    
    def recall(self, n: int = 2):
        """回忆历史消息（排除首尾）"""
        if len(self.history[1:-1]) == 0:
            return ''
        return '- ' + '\n- '.join([msg.content for msg in self.history[1:-1]])
    
    def get_latest_message_content(self) -> str:
        """获取最新消息内容"""
        return self.history[-1].content if len(self.history) > 1 else ''
    
    def get_first_message(self):
        """获取第一条消息（原始问题）"""
        return self.history[0] if len(self.history) > 0 else None
    
    def is_empty(self):
        """检查记忆是否为空"""
        return len(self.history) == 0
```

---

## 🔄 完整执行流程示例

假设用户问："帮我查询北京和上海的天气，然后对比一下"

### 步骤 1：初始化

```python
env = Environment(initial_message="帮我查询北京和上海的天气，然后对比一下")
agent = Agent(desc="天气助手", tools={"weather_tool": WeatherTool()})
```

### 步骤 2：第一轮循环

1. **Observe**：Agent 读取问题
2. **Reason**：LLM 输出
```json
{
    "type": "reason",
    "content": "需要分别查询北京和上海的天气，可以并行调用工具",
    "tool_calls": [
        {"tool_name": "weather_tool", "tool_args": {"city": "北京"}},
        {"tool_name": "weather_tool", "tool_args": {"city": "上海"}}
    ]
}
```
3. **Act**：执行 ReasonAction
   - 创建 1 条 REASON 消息："需要分别查询..."
   - 调用工具查询北京天气 → 创建 TOOL_EXECUTED 消息："北京晴天，25°C"
   - 调用工具查询上海天气 → 创建 TOOL_EXECUTED 消息："上海多云，22°C"
4. **更新环境**：3 条消息放回环境

### 步骤 3：第二轮循环

1. **Observe**：Agent 读取工具执行结果
2. **Reason**：LLM 输出
```
final|北京今天晴天，温度 25°C；上海今天多云，温度 22°C。北京比上海温度高 3°C，天气更晴朗。
```
3. **Act**：执行 FinalAction，返回最终答案
4. **结束**：检测到 DONE 消息，退出循环

---

## 🆚 与 agent.py 的核心区别

### 1. 动作类型

| agent.py | react.py |
|----------|----------|
| ThinkAction（只思考） | ReasonAction（思考+工具调用） |
| ToolAction（只调用工具） | - |
| FinalAction | FinalAction |

### 2. 工具调用方式

**agent.py**：
- 一次只能调用一个工具
- 思考和工具调用是分离的
- LLM 输出：`call_tool|工具名|参数|理由`

**react.py**：
- 一次可以调用多个工具（并行）
- 思考和工具调用合并在一起
- LLM 输出：JSON 格式，包含思考内容和工具列表

### 3. 消息类型

**agent.py**：
- NORMAL、ACTION、DONE

**react.py**：
- NORMAL、REASON、TOOL_EXECUTED、DONE
- 更细粒度的消息分类

### 4. 执行效率

**agent.py**：
- 串行执行：思考 → 调用工具 → 再思考 → 再调用工具

**react.py**：
- 并行执行：思考 → 同时调用多个工具 → 再思考

---

## 💡 ReAct 模式的优势

### 1. 更高效

- 支持并行工具调用，减少交互轮次
- 例如：查询多个城市天气，一次性调用完成

### 2. 更清晰

- 推理和行动合并，逻辑更连贯
- 消息类型更细分，便于追踪

### 3. 更灵活

- `tool_calls` 可以为空，支持纯思考
- 可以根据需要决定调用几个工具

### 4. 更符合人类思维

- 先思考整体计划
- 然后批量执行
- 再根据结果继续思考

---

## 📝 总结

`react.py` 实现了 **ReAct（Reasoning + Acting）** 范式的 Agent 框架，核心特点是：

1. **合并推理和行动**：ReasonAction 同时包含思考内容和工具调用
2. **支持并行工具调用**：一次可以调用多个工具，提高效率
3. **更细粒度的消息类型**：REASON、TOOL_EXECUTED 等，便于追踪
4. **更符合人类思维**：先整体思考，再批量执行

相比 `agent.py`，ReAct 模式更高效、更清晰、更灵活，是构建复杂 AI Agent 的推荐方案。

---

## 🔗 相关文件

- `agent.py`：基础 Agent 实现
- `llm.py`：大语言模型调用
- `event.py`：事件系统
- 具体的 Tool 实现文件

---

## 📚 扩展阅读

- **ReAct 论文**：[ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- **Agent 设计模式**：观察-推理-行动循环
- **并行工具调用**：如何设计支持并行的工具系统

---
