# Agent 架构文档

> 📚 **快速开始**: 如果你是第一次接触本项目，建议先阅读 [DEMO_START.md](DEMO_START.md) 了解如何配置环境和测试框架。

## 概述

本项目包含四个不同的 Agent 实现，它们代表了 Agent 架构的演进历程，从简单的 Agent 模式逐步发展到更复杂的 ReAct（Reason + Action）模式，最终形成了具有一致性检查能力的 RAC（Reason + Action + Check）模式。

### 四个文件的定位

1. **agent.py** - 基础 Agent 实现（第一代）
2. **react.py** - ReAct 模式实现（第二代）
3. **react_agent.py** - 现代化 ReAct Agent（第三代）
4. **rac_agent.py** - RAC Agent 带一致性检查（第四代）

---

## 一、agent.py - 基础 Agent 实现

### 功能概述
这是最基础的 Agent 实现，使用简单的文本格式与 LLM 交互，支持工具调用和思考过程。

### 核心特点

#### 1. 消息类型
```python
class MessageType(Enum):
    NORMAL = 1      # 普通消息
    ACTION = 2      # 工具执行消息
    DONE = 3        # 完成消息
```

#### 2. 行动类型
- **ToolAction**: 执行工具调用
- **ThinkAction**: 纯思考，不执行工具
- **FinalAction**: 最终回复

#### 3. 交互格式
使用简单的管道符分隔格式：
- 调用工具: `call_tool|tool_name|tool_args|reason`
- 继续思考: `continue_think|thought_content`
- 最终回复: `final_result|content`

#### 4. 工作流程
```
Observe → Reason → Act
   ↑                  ↓
   └──────────────────┘
```

### 优缺点

**优点：**
- 实现简单，易于理解
- 轻量级，适合简单场景

**缺点：**
- 每次只能调用一个工具
- 输出格式不够标准化
- 缺乏复杂的推理能力

---

## 二、react.py - ReAct 模式实现

### 功能概述
引入了 ReAct（Reasoning and Acting）模式，允许 Agent 进行更复杂的推理并支持多工具并行调用。

### 核心特点

#### 1. 消息类型
```python
class MessageType(Enum):
    NORMAL = 1           # 普通消息
    REASON = 2           # 推理消息
    TOOL_EXECUTED = 3    # 工具执行完成
    DONE = 4             # 完成消息
```

#### 2. 行动类型
- **ReasonAction**: 推理行动，可包含多个工具调用
- **FinalAction**: 最终回复

#### 3. 交互格式
使用 JSON 格式进行交互：
```json
{
    "type": "reason",
    "content": "推理内容",
    "tool_calls": [
        {"tool_name": "tool1", "tool_args": {...}},
        {"tool_name": "tool2", "tool_args": {...}}
    ]
}
```

#### 4. 工作流程
```
Observe → Reason (可包含多工具调用) → Act → Observe
   ↑                                            ↓
   └────────────────────────────────────────────┘
```

### 优缺点

**优点：**
- 支持多工具并行调用
- JSON 格式更结构化
- 引入了 ReAct 思想

**缺点：**
- 使用自定义 JSON 格式，非标准
- 与现代 LLM 工具调用格式不兼容

---

## 三、react_agent.py - 现代化 ReAct Agent

### 功能概述
采用标准 OpenAI 工具调用格式的现代化 ReAct Agent 实现，是 RAC Agent 的基础框架。

### 核心特点

#### 1. 消息类型
```python
class MessageType(Enum):
    SYSTEM = "system"       # 系统消息
    USER = "user"          # 用户消息
    ASSISTANT = "assistant" # 助手消息
    TOOL = "tool"          # 工具消息
    DONE = "done"          # 完成消息
```

#### 2. Message 类设计
使用 Pydantic BaseModel，支持与 LLM 消息格式的双向转换：
- `to_llm_message()`: 转换为 LLM 格式
- `from_llm_message()`: 从 LLM 格式转换

#### 3. 工具调用格式
采用 OpenAI 标准格式：
```python
{
    "type": "function",
    "function": {
        "name": "tool_name",
        "description": "...",
        "parameters": {...}
    }
}
```

#### 4. 核心组件

**ReasonAction**:
- 执行推理和工具调用
- 支持标准的 ChatCompletionMessageToolCall
- 自动处理工具执行结果

**Memory**:
- 基于 Pydantic 的消息存储
- 支持消息历史管理
- 可设置最大消息数量

**Environment**:
- 消息队列管理
- 支持上下文存储
- 消息消费机制

**TerminateTool**:
- 内置终止工具
- 用于结束交互

#### 5. 防卡住机制
- 检测重复思考: `_is_stuck()`
- 自动提示避免重复

#### 6. 工作流程
```
Observe → Reason → Act
   ↑                 ↓
   └─────────────────┘
```

### 优缺点

**优点：**
- 使用标准 OpenAI 工具调用格式
- 代码结构清晰，使用现代 Python 特性（Pydantic）
- 支持多工具并行调用
- 内置防卡住机制
- 消息格式标准化

**缺点：**
- 缺乏执行结果验证机制
- 无法检测推理与执行的一致性

---

## 四、rac_agent.py - RAC Agent 带一致性检查

### 功能概述
在 ReAct Agent 基础上增加了 Check（检查）步骤，用于验证推理和执行的一致性，确保 Agent 的行为符合预期。

### 核心特点

#### 1. RAC 三步循环
```
Reason (推理) → Action (执行) → Check (检查)
     ↑                                ↓
     └────────────────────────────────┘
```

#### 2. CheckAction 类
专门用于一致性验证的行动类：

**检查项目：**
1. **工具调用完整性**: 验证所有计划的工具是否都被执行
2. **执行数量匹配**: 检查推理中的工具数量与实际执行数量是否一致
3. **执行错误检测**: 识别工具执行过程中的错误
4. **推理逻辑连贯性**: 验证思考内容是否提及实际使用的工具
5. **终止逻辑验证**: 确保终止调用与完成消息的一致性

#### 3. CheckResult 模型
```python
class CheckResult(BaseModel):
    success: bool                    # 检查是否成功
    consistency_score: float         # 一致性得分 (0-1)
    issues: List[str]               # 发现的问题列表
    recommendations: List[str]       # 改进建议
```

#### 4. 一致性评分机制
- 初始分数: 1.0
- 每发现一个问题扣分
- 阈值: 0.7（可配置）
- 低于阈值视为检查失败

#### 5. 失败处理
当检查失败时：
1. 记录失败原因和建议
2. 将失败消息添加到环境
3. 继续下一轮循环
4. Agent 可以根据反馈调整策略

#### 6. 可配置参数
```python
RACAgent(
    description="...",
    model="qwen-max-latest",
    tools=[...],
    check_threshold=0.7,  # 一致性阈值
)

agent.run(
    env=env,
    max_steps=50,
    enable_check=True,  # 是否启用检查
)
```

#### 7. 工作流程
```
Observe → Reason → Action → Check
   ↑                           ↓
   │    (检查通过)              │
   └───────────────────────────┘
   ↑                           ↓
   │    (检查失败，重新推理)    │
   └───────────────────────────┘
```

### 优缺点

**优点：**
- 增加了执行验证机制，提高可靠性
- 可量化的一致性评分
- 详细的问题诊断和建议
- 可配置的检查阈值
- 可选启用/禁用检查功能
- 继承了 ReAct Agent 的所有优点

**缺点：**
- 增加了额外的计算开销
- 检查逻辑需要持续优化
- 对于简单任务可能过度工程化

---

## 架构演进对比

| 特性 | agent.py | react.py | react_agent.py | rac_agent.py |
|------|----------|----------|----------------|--------------|
| **推理模式** | 简单推理 | ReAct | ReAct | RAC |
| **工具调用格式** | 文本格式 | JSON | OpenAI 标准 | OpenAI 标准 |
| **多工具并行** | ❌ | ✅ | ✅ | ✅ |
| **一致性检查** | ❌ | ❌ | ❌ | ✅ |
| **防卡住机制** | ❌ | ❌ | ✅ | ✅ |
| **消息标准化** | 基础 | 中等 | 完整 | 完整 |
| **代码现代化** | 基础 | 基础 | Pydantic | Pydantic |
| **适用场景** | 简单任务 | 中等复杂 | 复杂任务 | 关键任务 |

---

## 使用建议

### 选择 agent.py 的场景
- 极简单的工具调用场景
- 学习 Agent 基础概念
- 对性能要求极高的场景

### 选择 react.py 的场景
- 需要多工具并行调用
- 旧项目维护
- 不依赖 OpenAI 格式

### 选择 react_agent.py 的场景
- 标准的 Agent 应用
- 需要与现代 LLM 集成
- 需要清晰的代码结构
- 大多数生产环境

### 选择 rac_agent.py 的场景
- 对可靠性要求高的场景
- 需要验证 Agent 执行正确性
- 复杂的多步骤任务
- 需要问题诊断能力
- 关键业务逻辑

---

## 技术实现细节

### 共同依赖
所有实现都依赖：
- `LlmCaller`: LLM 调用封装
- `event`: 事件系统
- `loguru`: 日志记录

### 核心设计模式

1. **观察者模式**: Environment 管理消息队列
2. **策略模式**: 不同的 Action 类型
3. **模板模式**: Agent 的 run 方法定义主循环
4. **工厂模式**: Tool 的创建和管理

### Memory 管理

- **agent.py & react.py**: 简单列表存储
- **react_agent.py & rac_agent.py**: Pydantic 模型，支持消息限制

### 消息流转

```
User Input → Environment → Agent.observe() 
    → Agent.reason() → Action.execute() 
    → Environment → Agent.observe() → ...
```

---

## 最佳实践

### 1. 选择合适的实现
根据任务复杂度和可靠性要求选择：
- 简单任务 → agent.py
- 标准任务 → react_agent.py
- 关键任务 → rac_agent.py

### 2. 工具设计
- 继承 Tool 基类
- 实现 execute 方法
- 提供清晰的描述和参数定义

### 3. Prompt 设计
- 使用 COT_PROMPT 引导思考
- 使用 FOCUS_PROMPT 避免重复
- 针对特定领域定制系统提示

### 4. 错误处理
- 实现 ActionResult 错误传递
- 在 Environment 中记录错误消息
- 利用 RAC 的检查机制捕获问题

### 5. 性能优化
- 设置合理的 max_steps
- 使用 Memory 的消息限制
- 在不需要时禁用 check

---

## 总结

这四个文件展示了 Agent 架构从简单到复杂的演进过程：

1. **agent.py**: 证明了基础 Agent 概念的可行性
2. **react.py**: 引入了 ReAct 思想，提升了推理能力
3. **react_agent.py**: 采用现代化技术栈，标准化了实现
4. **rac_agent.py**: 在可靠性上更进一步，增加了验证机制

对于新项目，建议优先考虑 **react_agent.py** 作为起点，如果对可靠性有更高要求，则使用 **rac_agent.py**。旧的实现（agent.py 和 react.py）主要用于维护和学习目的。

---

## 附录：代码示例

### 使用 react_agent.py
```python
from .react_agent import ReActAgent, Environment, Tool

# 定义工具
tools = [...]

# 创建 Agent
agent = ReActAgent(
    description="Your agent description",
    model="qwen-max-latest",
    tools=tools
)

# 创建环境并运行
env = Environment(initial_message="用户问题")
result = agent.run(env, max_steps=20)
```

### 使用 rac_agent.py
```python
from .rac_agent import RACAgent
from .react_agent import Environment, Tool

# 定义工具
tools = [...]

# 创建 RAC Agent
agent = RACAgent(
    description="Your agent description",
    model="qwen-max-latest",
    tools=tools,
    check_threshold=0.7
)

# 创建环境并运行（启用检查）
env = Environment(initial_message="用户问题")
result = agent.run(env, max_steps=50, enable_check=True)
```

---

## 参考与启发

本项目的设计和实现受到以下优秀资源的启发：

1. **[Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)** - Anthropic
   - Anthropic 关于构建有效 Agent 的工程实践和设计思想

2. **[OpenManus](https://github.com/FoundationAgents/OpenManus)** - FoundationAgents
   - 开源的 Agent 框架实现，提供了丰富的参考案例