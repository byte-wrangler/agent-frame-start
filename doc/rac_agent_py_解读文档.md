# rac_agent.py 文件解读文档

## 📚 文档概述

本文档旨在帮助不同读者理解 `rac_agent.py` 文件的核心概念和实现逻辑。实现了一个 **RAC（Reason-Action-Check）** Agent 框架，是 `react_agent.py` 的增强版本，在 ReAct 模式基础上增加了**一致性检查（Consistency Check）**机制，确保 Agent 的推理和执行保持一致。

---

## 🎯 什么是 RAC？

**RAC = Reasoning（推理） + Acting（行动） + Checking（检查）**

RAC 是在 ReAct 基础上增加了自我检查能力的 Agent 模式：

1. **Reasoning（推理）**：分析问题，制定计划
2. **Acting（行动）**：执行工具调用
3. **Checking（检查）**：验证推理和执行的一致性

### 为什么需要 Check？

在实际应用中，Agent 可能会出现以下问题：
- 计划使用某个工具，但实际没有执行
- 推理说要做 A，实际却做了 B
- 工具执行出错，但 Agent 没有察觉
- 推理逻辑和实际行动不一致

RAC 通过增加 Check 步骤，让 Agent 具备**自我验证**能力，提高可靠性。

---

## 🆚 与 react_agent.py 的核心区别

| 特性 | react_agent.py | rac_agent.py |
|------|----------------|--------------|
| **执行流程** | Observe → Reason → Act | Observe → Reason → Act → **Check** |
| **一致性验证** | ❌ 无 | ✅ 自动检查 |
| **错误检测** | 基础 | 增强（5 项检查） |
| **自我修正** | ❌ 无 | ✅ 检查失败后重试 |
| **可靠性** | 中等 | 高 |
| **适用场景** | 一般任务 | 关键任务、需要高可靠性 |

---

## 📦 主要组件详解

### 一、CheckResult（检查结果模型）

```python
class CheckResult(BaseModel):
    success: bool                    # 检查是否成功
    consistency_score: float         # 一致性得分 (0-1)
    issues: List[str]               # 发现的问题列表
    recommendations: List[str]       # 改进建议
```

**作用**：
- 封装一致性检查的结果
- 提供量化的一致性评分
- 列出具体问题和改进建议

**使用 Pydantic**：
- 自动数据验证
- 类型安全
- 支持序列化

---

### 二、CheckAction（检查动作）- 核心创新

```python
class CheckAction(Action):
    def __init__(
        self, 
        reason_action: ReasonAction, 
        execution_results: List[Message]
    ):
        self.type = "check"
        self.reason_action = reason_action      # 要检查的推理动作
        self.execution_results = execution_results  # 执行结果
        self.thought = reason_action.thought
        self.tool_calls = reason_action.tool_calls
```

**作用**：
- 对比推理计划和实际执行结果
- 检测不一致性
- 生成改进建议

#### 2.1 执行流程

```python
def execute(self, env: "Environment") -> ActionResult:
    try:
        # 1. 执行一致性检查
        check_result = self._perform_consistency_check()
        
        # 2. 检查通过
        if check_result.success:
            emit_event(EventType.AGENT, 
                f"[Check] ✓ Consistency check passed (score: {check_result.consistency_score:.2f})")
            return ActionResult(success=True, value=check_result)
        
        # 3. 检查失败
        else:
            issues_str = "; ".join(check_result.issues)
            emit_event(EventType.AGENT, 
                f"[Check] ✗ Consistency check failed: {issues_str}")
            
            # 创建失败消息，反馈给 Agent
            failure_message = Message(
                type=MessageType.USER,
                content=f"检查失败: {issues_str}. 建议: {'; '.join(check_result.recommendations)}"
            )
            
            return ActionResult(
                success=False,
                value=failure_message,
                error=f"Consistency check failed: {issues_str}"
            )
    
    except Exception as e:
        logger.exception(f"Check execution failed: {e}")
        return ActionResult(success=False, value=None, error=str(e))
```

**关键特性**：
- 检查通过：返回成功结果
- 检查失败：生成反馈消息，让 Agent 重新思考
- 异常处理：捕获所有异常

#### 2.2 五项一致性检查

```python
def _perform_consistency_check(self) -> CheckResult:
    issues = []
    recommendations = []
    consistency_score = 1.0  # 初始满分
```

##### 检查 1：验证所有计划的工具都被执行

```python
# 提取计划使用的工具
planned_tools = set()
for tool_call in self.tool_calls:
    tool_name = tool_call.get('function', {}).get('name')
    if tool_name:
        planned_tools.add(tool_name)

# 提取实际执行的工具
executed_tools = set()
for result in self.execution_results:
    if result.type == MessageType.TOOL:
        tool_name = result.content.get('name')
        if tool_name:
            executed_tools.add(tool_name)

# 检查缺失的工具
missing_tools = planned_tools - executed_tools
if missing_tools:
    issues.append(f"计划使用的工具 {missing_tools} 未被执行")
    recommendations.append("重新执行缺失的工具调用")
    consistency_score -= 0.3
```

**检测问题**：计划调用某工具，但实际没有执行  
**扣分**：-0.3

##### 检查 2：验证推理与执行数量匹配

```python
if len(self.tool_calls) != len([r for r in self.execution_results if r.type == MessageType.TOOL]):
    issues.append("推理中的工具调用数量与实际执行数量不匹配")
    recommendations.append("检查工具调用的执行逻辑")
    consistency_score -= 0.2
```

**检测问题**：计划调用 3 个工具，实际只执行了 2 个  
**扣分**：-0.2

##### 检查 3：检查执行错误

```python
execution_errors = []
for result in self.execution_results:
    if result.type == MessageType.TOOL:
        content = result.content.get('content', '')
        if 'error' in content.lower() or 'failed' in content.lower():
            execution_errors.append(result.content.get('name', 'unknown'))

if execution_errors:
    issues.append(f"工具执行出现错误: {execution_errors}")
    recommendations.append("检查工具参数和执行环境")
    consistency_score -= 0.3
```

**检测问题**：工具执行返回错误信息  
**扣分**：-0.3

##### 检查 4：验证推理逻辑连贯性

```python
if self.thought:
    thought_lower = self.thought.lower()
    for tool_name in executed_tools:
        if tool_name.lower() not in thought_lower:
            issues.append(f"推理内容中未提及实际使用的工具: {tool_name}")
            recommendations.append("确保推理过程与实际行动一致")
            consistency_score -= 0.1
```

**检测问题**：推理中没有提到实际使用的工具  
**扣分**：-0.1（每个工具）

##### 检查 5：验证终止逻辑

```python
has_terminate_call = any(
    tool_call.get('function', {}).get('name') == 'terminate' 
    for tool_call in self.tool_calls
)
has_done_message = any(
    result.type == MessageType.DONE 
    for result in self.execution_results
)

if has_terminate_call and not has_done_message:
    issues.append("计划终止但未生成完成消息")
    recommendations.append("检查终止逻辑的执行")
    consistency_score -= 0.2
```

**检测问题**：计划调用 terminate，但没有生成 DONE 消息  
**扣分**：-0.2

##### 最终判定

```python
# 确保分数在 0-1 范围内
consistency_score = max(0.0, min(1.0, consistency_score))

# 判定成功：分数 >= 0.7 且没有问题
success = consistency_score >= 0.7 and len(issues) == 0

return CheckResult(
    success=success,
    consistency_score=consistency_score,
    issues=issues,
    recommendations=recommendations
)
```

**成功条件**：
- 一致性得分 ≥ 0.7（默认阈值）
- 没有发现任何问题

---

### 三、RACAgent（RAC 智能体核心）

#### 3.1 初始化

```python
class RACAgent(ABC):
    def __init__(
        self,
        description: str,
        model: str = "qwen-max-latest",
        tools: Optional[list[Tool]] = None,
        check_threshold: float = 0.7
    ):
        # 推理用的 LLM
        self.llm_caller = LlmCaller(
            f"{description}\n\n{COT_PROMPT}", 
            model
        )
        
        # 检查用的 LLM（可以使用不同的提示词）
        self.check_llm_caller = LlmCaller(
            f"{description}\n\n{CHECK_PROMPT}", 
            model
        )
        
        self.memory = Memory()
        self.tools = tools + [TerminateTool()] \
            if tools is not None \
            else [TerminateTool()]
        
        # 一致性检查阈值
        self.check_threshold = check_threshold
```

**关键特性**：

1. **双 LLM Caller**：
   - `llm_caller`：用于推理
   - `check_llm_caller`：用于检查（预留，当前未使用）

2. **可配置阈值**：
   - `check_threshold`：一致性检查的通过阈值
   - 默认 0.7，可根据任务严格程度调整

**CHECK_PROMPT（检查提示词）**：

```python
CHECK_PROMPT = """
Based on the reasoning and execution results, perform a consistency check to ensure:
1. All planned actions were executed correctly
2. The reasoning logic matches the actual execution
3. No critical steps were skipped
4. Tool calls were executed in the expected order
5. Any errors or inconsistencies are identified

Focus on validating the coherence between intention and execution.
"""
```

#### 3.2 主循环（run）- RAC 流程

```python
def run(
    self,
    env: Environment,
    max_steps: Optional[int] = 50,
    enable_check: bool = True
) -> Message:
    """主循环：观察 → 推理 → 行动 → 检查
    
    :param env: 环境
    :param max_steps: 最大步数（默认 50）
    :param enable_check: 是否启用检查（默认 True）
    :return: 最终消息
    """
    emit_event(
        EventType.AGENT,
        f"[Observe] [{env.peek_message().type.value}] "
        f"{env.peek_message().content}"
    )
    
    current_step = 0
    while True:
        # 1. 观察
        self.observe(env)
        
        # 2. 推理
        reason_action = self.reason()
        
        # 3. 行动
        action_results = self.act(reason_action, env)
        
        # 4. 检查（如果启用且不是终止动作）
        if enable_check and not self._is_terminating_action(reason_action):
            check_result = self.check(reason_action, action_results, env)
            
            # 检查失败，添加失败消息并继续
            if not check_result.success:
                env.add_message(check_result.value)
                current_step += 1
                if max_steps is not None and current_step >= max_steps:
                    break
                continue  # 重新开始循环，让 Agent 根据反馈重新思考
        
        # 5. 检查是否完成
        latest_message = env.peek_message()
        if latest_message and latest_message.type == MessageType.DONE:
            return env.peek_latest_not_empty_message(MessageType.ASSISTANT)
        
        # 6. 检查步数限制
        current_step += 1
        if max_steps is not None and current_step >= max_steps:
            return latest_message
```

**执行流程图**：

```
┌─────────────────────────────────────┐
│         RAC Agent 主循环            │
└─────────────────────────────────────┘
              │
              ▼
    ┌──────────────────┐
    │  1. Observe      │  从环境中观察消息
    └──────────────────┘
              │
              ▼
    ┌──────────────────┐
    │  2. Reason       │  推理下一步动作
    └──────────────────┘
              │
              ▼
    ┌──────────────────┐
    │  3. Act          │  执行动作
    └──────────────────┘
              │
              ▼
    ┌──────────────────┐
    │  4. Check        │  检查一致性
    └──────────────────┘
              │
         /        \
    检查通过    检查失败
       │           │
       │           ▼
       │    添加失败消息
       │           │
       │           └──► 返回步骤 1（重新思考）
       │
       ▼
    是否完成？
    /        \
  是          否
  │           │
  ▼           └──► 返回步骤 1
返回结果
```

**关键特性**：

1. **检查失败后重试**：
   - 检查失败时，将失败消息添加到环境
   - `continue` 跳过本轮，重新开始循环
   - Agent 会看到失败消息，重新思考

2. **跳过终止动作的检查**：
   - 如果是 terminate 调用，不进行检查
   - 避免不必要的检查开销

3. **可选的检查功能**：
   - `enable_check=False` 可以禁用检查
   - 降级为普通的 ReAct Agent

#### 3.3 核心方法

##### observe（观察）

```python
def observe(self, env):
    """从环境中观察未读消息"""
    messages = env.pull_messages()
    self.memory.add_messages(messages)
```

与 `react_agent.py` 完全相同。

##### reason（推理）

```python
def reason(self) -> ReasonAction:
    """推理并生成动作"""
    # 准备消息历史
    llm_messages = []
    for message in self.memory.messages:
        llm_messages.append(message.to_llm_message())
    
    # 添加聚焦提示词
    focus_prompt = FOCUS_PROMPT
    
    # 检查是否陷入重复思考
    if self._is_stuck():
        focus_prompt = self._handle_stuck(focus_prompt)
    
    llm_messages.append(
        Message(
            type=MessageType.USER,
            content=focus_prompt
        ).to_llm_message()
    )
    
    # 调用 LLM
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
```

与 `react_agent.py` 完全相同。

##### act（行动）

```python
def act(self, action: ReasonAction, env: Environment) -> List[Message]:
    """执行动作并返回结果消息列表"""
    # 执行动作
    action_result = action.execute(env)
    
    # 处理执行结果
    if not action_result.success or not action_result.value:
        error_message = Message(
            type=MessageType.USER,
            content=f"Action executed error because of {action_result.error}. "
                    f"The action is {action.to_dict()}"
        )
        env.add_message(error_message)
        return [error_message]
    
    # 执行成功，添加结果消息
    env.add_messages(action_result.value)
    return action_result.value
```

**与 react_agent.py 的区别**：
- 返回消息列表而非 None
- 用于后续的 Check 步骤

##### check（检查）- 新增方法

```python
def check(
    self, 
    reason_action: ReasonAction, 
    execution_results: List[Message], 
    env: Environment
) -> ActionResult:
    """检查推理和执行的一致性"""
    check_action = CheckAction(reason_action, execution_results)
    return check_action.execute(env)
```

**作用**：
- 创建 CheckAction
- 执行一致性检查
- 返回检查结果

##### 辅助方法

```python
def _is_terminating_action(self, action: ReasonAction) -> bool:
    """检查动作是否包含 terminate 工具调用"""
    return any(
        tool_call.get('function', {}).get('name') == 'terminate'
        for tool_call in action.tool_calls
    )

def _is_stuck(self):
    """检查是否陷入重复思考"""
    if len(self.memory.messages) < 2:
        return False
    
    last_message = self.memory.messages[-1]
    duplicate_count = 0
    for message in reversed(self.memory.messages[:-1]):
        if message.type == MessageType.ASSISTANT \
                and last_message.is_same(message):
            duplicate_count += 1
    
    return duplicate_count > 0

def _handle_stuck(self, next_prompt: str) -> str:
    """处理卡住情况"""
    stuck_prompt = "已经发现你正在重复思考，请避免重复已经思考过的内容并尝试新的思考，如果思考结束请使用terminate工具"
    return f"{stuck_prompt}\n{next_prompt}"
```

与 `react_agent.py` 完全相同。

---

## 🔄 完整执行流程示例

假设用户问："帮我查询北京天气，如果温度超过 30 度就发送提醒"

### 步骤 1：初始化

```python
from rac_agent import RACAgent
from react_agent import Environment, Tool

# 定义工具（与之前相同）
tools = [WeatherTool(), AlertTool()]

# 创建 RAC Agent
env = Environment(initial_message="帮我查询北京天气，如果温度超过 30 度就发送提醒")
agent = RACAgent(
    description="天气助手",
    tools=tools,
    check_threshold=0.7  # 一致性阈值
)
```

### 步骤 2：第一轮循环

#### 2.1 Observe
Agent 读取用户问题

#### 2.2 Reason
LLM 输出：
```python
{
    "content": "我需要先查询北京的天气",
    "tool_calls": [
        {
            "function": {
                "name": "get_weather",
                "arguments": '{"city": "北京"}'
            }
        }
    ]
}
```

#### 2.3 Act
执行 ReasonAction：
- 创建 ASSISTANT 消息
- 执行 `get_weather` 工具
- 创建 TOOL 消息："北京今天晴天，温度 32°C"
- 返回 2 条消息

#### 2.4 Check
执行 CheckAction：

**检查 1**：计划工具 = {get_weather}，执行工具 = {get_weather} ✓  
**检查 2**：计划 1 个工具，执行 1 个工具 ✓  
**检查 3**：无执行错误 ✓  
**检查 4**：推理中提到了 "查询北京的天气" ✓  
**检查 5**：无 terminate 调用 ✓  

**结果**：
- `consistency_score = 1.0`
- `success = True`
- 检查通过 ✓

### 步骤 3：第二轮循环

#### 3.1 Observe
Agent 读取天气查询结果

#### 3.2 Reason
LLM 输出：
```python
{
    "content": "北京温度是 32°C，超过了 30 度，需要发送提醒",
    "tool_calls": [
        {
            "function": {
                "name": "send_alert",
                "arguments": '{"message": "北京温度 32°C，超过 30 度"}'
            }
        }
    ]
}
```

#### 3.3 Act
执行 ReasonAction：
- 创建 ASSISTANT 消息
- 执行 `send_alert` 工具
- 创建 TOOL 消息："已发送提醒：..."
- 返回 2 条消息

#### 3.4 Check
执行 CheckAction：

**检查 1**：计划工具 = {send_alert}，执行工具 = {send_alert} ✓  
**检查 2**：计划 1 个工具，执行 1 个工具 ✓  
**检查 3**：无执行错误 ✓  
**检查 4**：推理中提到了 "发送提醒" ✓  
**检查 5**：无 terminate 调用 ✓  

**结果**：
- `consistency_score = 1.0`
- `success = True`
- 检查通过 ✓

### 步骤 4：第三轮循环

#### 4.1 Observe
Agent 读取提醒发送结果

#### 4.2 Reason
LLM 输出：
```python
{
    "content": "已完成天气查询和提醒发送，任务完成",
    "tool_calls": [
        {
            "function": {
                "name": "terminate",
                "arguments": '{"status": "success"}'
            }
        }
    ]
}
```

#### 4.3 Act
执行 ReasonAction：
- 检测到 `terminate` 工具
- 创建 DONE 消息
- 返回消息

#### 4.4 Check
跳过检查（因为是 terminate 动作）

#### 4.5 结束
检测到 DONE 消息，退出循环

---

## 🚨 检查失败的场景示例

假设在某一轮中，Agent 出现了不一致的情况：

### 场景：计划使用工具但未执行

#### Reason
```python
{
    "content": "我需要查询北京和上海的天气",
    "tool_calls": [
        {"function": {"name": "get_weather", "arguments": '{"city": "北京"}'}},
        {"function": {"name": "get_weather", "arguments": '{"city": "上海"}'}}
    ]
}
```

#### Act
由于某种原因（如工具执行错误），只执行了北京的查询：
- TOOL 消息："北京今天晴天，温度 32°C"

#### Check
执行 CheckAction：

**检查 1**：
- 计划工具：2 次 get_weather 调用
- 执行工具：1 次 get_weather 调用
- ✗ 不匹配！

**检查结果**：
```python
CheckResult(
    success=False,
    consistency_score=0.7,  # 1.0 - 0.3 = 0.7
    issues=["计划使用的工具部分未被执行"],
    recommendations=["重新执行缺失的工具调用"]
)
```

#### 处理
1. 检查失败，`check_result.success = False`
2. 创建失败消息：
```python
Message(
    type=MessageType.USER,
    content="检查失败: 计划使用的工具部分未被执行. 建议: 重新执行缺失的工具调用"
)
```
3. 添加到环境，`continue` 重新开始循环
4. Agent 在下一轮会看到这条失败消息，重新思考并修正

---

## 🆚 三个版本的完整对比

### 1. 执行流程对比

| 版本 | 执行流程 |
|------|----------|
| **react.py** | Observe → Reason → Act |
| **react_agent.py** | Observe → Reason → Act |
| **rac_agent.py** | Observe → Reason → Act → **Check** |

### 2. 功能对比

| 功能 | react.py | react_agent.py | rac_agent.py |
|------|----------|----------------|--------------|
| **基础 ReAct** | ✅ | ✅ | ✅ |
| **Pydantic 模型** | ❌ | ✅ | ✅ |
| **消息转换** | ❌ | ✅ | ✅ |
| **防重复机制** | ❌ | ✅ | ✅ |
| **TerminateTool** | ❌ | ✅ | ✅ |
| **一致性检查** | ❌ | ❌ | ✅ |
| **自我修正** | ❌ | ❌ | ✅ |
| **检查阈值配置** | ❌ | ❌ | ✅ |

### 3. 可靠性对比

| 指标 | react.py | react_agent.py | rac_agent.py |
|------|----------|----------------|--------------|
| **错误检测** | 基础 | 中等 | 高 |
| **自我验证** | ❌ | ❌ | ✅ |
| **错误恢复** | ❌ | 部分 | ✅ |
| **适用场景** | 简单任务 | 一般任务 | 关键任务 |

### 4. 性能对比

| 指标 | react.py | react_agent.py | rac_agent.py |
|------|----------|----------------|--------------|
| **执行速度** | 快 | 快 | 中等（多了 Check） |
| **资源消耗** | 低 | 低 | 中等 |
| **可靠性** | 中 | 高 | 很高 |
| **适合场景** | 原型开发 | 生产环境 | 高可靠性需求 |

---

## 💡 核心设计模式

### 1. 责任链模式（Chain of Responsibility）

```
Observe → Reason → Act → Check
   ↓        ↓       ↓       ↓
 观察    → 推理  → 行动  → 检查
```

每个步骤负责特定的职责，形成处理链。

### 2. 策略模式（Strategy Pattern）

不同的 Action 类型：
- `ReasonAction`：推理 + 工具调用
- `CheckAction`：一致性检查

### 3. 模板方法模式（Template Method）

`run()` 方法定义了固定的执行模板，子类可以扩展具体步骤。

### 4. 观察者模式（Observer Pattern）

通过 `emit_event` 发送事件，外部可以监听 Agent 的执行过程。

---

## 🔧 实用技巧

### 1. 调整检查阈值

```python
# 严格模式：要求完美一致
agent = RACAgent(
    description="...",
    tools=[...],
    check_threshold=0.9  # 90% 一致性
)

# 宽松模式：允许一些小问题
agent = RACAgent(
    description="...",
    tools=[...],
    check_threshold=0.5  # 50% 一致性
)

# 默认模式
agent = RACAgent(
    description="...",
    tools=[...],
    check_threshold=0.7  # 70% 一致性（推荐）
)
```

### 2. 禁用检查（降级为 ReAct）

```python
# 在某些场景下可以禁用检查
result = agent.run(
    env=env,
    max_steps=20,
    enable_check=False  # 禁用检查
)
```

**使用场景**：
- 简单任务，不需要检查
- 性能敏感的场景
- 调试时快速迭代

### 3. 自定义检查逻辑

```python
class MyCheckAction(CheckAction):
    def _perform_consistency_check(self) -> CheckResult:
        # 调用父类的检查
        result = super()._perform_consistency_check()
        
        # 添加自定义检查
        if self._check_custom_rule():
            result.issues.append("自定义规则检查失败")
            result.consistency_score -= 0.2
        
        return result
    
    def _check_custom_rule(self) -> bool:
        # 实现自定义检查逻辑
        pass
```

### 4. 监控检查结果

```python
from event import EventType

def on_check_event(event_type, message):
    if "[Check]" in message:
        if "✓" in message:
            print(f"检查通过: {message}")
        elif "✗" in message:
            print(f"检查失败: {message}")

# 注册事件监听器
# （具体实现取决于 event.py 的接口）
```

---

## 📝 总结

`rac_agent.py` 实现了 **RAC（Reason-Action-Check）** Agent 框架，核心特点是：

### 核心优势

1. **自我验证**：通过 Check 步骤验证推理和执行的一致性
2. **自我修正**：检查失败后自动重试，提高可靠性
3. **量化评估**：提供 0-1 的一致性评分
4. **详细反馈**：列出具体问题和改进建议
5. **可配置**：支持调整检查阈值和启用/禁用检查
6. **五项检查**：全面覆盖常见的不一致问题

### 五项一致性检查

1. ✅ 验证所有计划的工具都被执行
2. ✅ 验证推理与执行数量匹配
3. ✅ 检查执行错误
4. ✅ 验证推理逻辑连贯性
5. ✅ 验证终止逻辑

### 适用场景

- ✅ 关键业务任务（如金融、医疗）
- ✅ 需要高可靠性的场景
- ✅ 复杂的多步骤任务
- ✅ 需要审计和验证的场景
- ✅ 对错误容忍度低的应用

### 与其他版本的关系

```
agent.py (基础版)
    ↓ 改进
react.py (ReAct 模式)
    ↓ 企业级增强
react_agent.py (生产级 ReAct)
    ↓ 增加一致性检查
rac_agent.py (RAC 模式)
```

---

## 🔗 相关文件

- `agent.py`：基础 Agent 实现
- `react.py`：ReAct 模式实现
- `react_agent.py`：生产级 ReAct 实现
- `llm.py`：LLM 调用封装
- `event.py`：事件系统

---

## 📚 扩展阅读

- **ReAct 论文**：https://arxiv.org/abs/2210.03629
- **Self-Consistency**：自我一致性检查的相关研究
- **Agent 可靠性**：如何提高 AI Agent 的可靠性
- **Pydantic 文档**：https://docs.pydantic.dev/
- **思维链（CoT）**：Chain-of-Thought Prompting
