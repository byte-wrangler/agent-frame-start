# Agent Framework 测试指南

本文档将指导你从零开始配置环境并测试本 Agent 框架的各个组件。

## 📋 目录

- [环境要求](#环境要求)
- [环境配置](#环境配置)
- [数据库初始化](#数据库初始化)
- [配置文件说明](#配置文件说明)
- [Agent 测试](#agent-测试)
  - [1. Agent.py 测试](#1-agentpy-测试)
  - [2. React.py 测试](#2-reactpy-测试)
  - [3. ReActAgent 测试](#3-reactagent-测试)
  - [4. RACAgent 测试](#4-racagent-测试)
- [常见问题](#常见问题)

---

## 环境要求

- **Python**: 3.8 或更高版本
- **操作系统**: macOS / Linux / Windows
- **网络**: 需要访问阿里云 DashScope API

---

## 环境配置

### 1. 克隆或下载项目

```bash
cd /path/to/your/workspace
# 如果是 git 项目
git clone <your-repo-url>
cd agent-frame-start
```

### 2. 创建虚拟环境（推荐）

```bash
# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
# macOS/Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

项目依赖精简，只包含核心必需包：
- `loguru==0.7.3` - 日志记录
- `openai==1.66.3` - OpenAI SDK（用于调用 LLM）
- `pydantic==2.10.4` - 数据验证和模型定义
- `SQLAlchemy==2.0.37` - 数据库 ORM

所有依赖都已经过精简优化，确保项目轻量高效 ✨

---

## 数据库初始化

### 1. 自动初始化（推荐）

项目提供了自动初始化脚本：

```bash
python test_db_init.py
```

**预期输出：**
```
============================================================
开始初始化SQLite数据库...
============================================================

数据库连接信息: sqlite:///data/test.db

✓ 数据库表结构创建完成！

已创建的表：
  - job_runs
  - events

============================================================
测试数据库操作...
============================================================

✓ 成功插入测试数据: JobRun(id=1)
✓ 成功查询数据: <JobRun(id=1)>
  转换为字典: {'id': 1, 'status': 'created', ...}

============================================================
✅ 数据库初始化和测试全部完成！
============================================================
```

### 2. 数据库文件位置

初始化成功后，数据库文件会保存在：
```
data/test.db
```

### 3. 验证数据库

```bash
# 使用 sqlite3 命令行工具查看
sqlite3 data/test.db

# 查看所有表
.tables

# 查看表结构
.schema job_runs

# 退出
.quit
```

---

## 配置文件说明

### config/config.json

```json
{
  "llm": {
    "default_model": "qwen-max",
    "api_key": "sk-your-api-key-here",
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
  },
  "sql_db": {
    "type": "sqlite",
    "database": "data/test.db"
  }
}
```

### 配置项说明

#### LLM 配置
- **`default_model`**: 默认使用的模型，可选值：
  - `qwen-max` - 通义千问最强模型
  - `qwen-max-latest` - 通义千问最新版本
  - `qwen-plus` - 通义千问进阶版
  - `qwen-turbo` - 通义千问轻量版

- **`api_key`**: 阿里云 DashScope API Key
  - 获取方式：访问 [DashScope 控制台](https://dashscope.console.aliyun.com/)
  - 注册并创建 API Key
  - **注意**: 请妥善保管你的 API Key，不要提交到代码仓库

- **`base_url`**: API 端点地址（通常不需要修改）

#### 数据库配置
- **`type`**: 数据库类型，当前支持 `sqlite`
- **`database`**: 数据库文件路径

### 修改配置

1. 复制配置文件（可选，用于保留原始配置）：
```bash
cp config/config.json config/config.json.backup
```

2. 编辑配置文件：
```bash
# 使用你喜欢的编辑器
vim config/config.json
# 或
nano config/config.json
```

3. **替换 API Key**（必须）：
```json
"api_key": "sk-your-actual-api-key-here"
```

---

## Agent 测试

本框架包含 4 种 Agent 实现，每种适用于不同场景。下面将逐一介绍如何测试。

### 1. Agent.py 测试

**特点**: 基础 Agent 实现，使用简单文本格式交互

#### 创建测试文件

创建 `test_agent.py`:

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试基础 Agent"""

from src.agent import Agent, Environment, Tool

class CalculatorTool(Tool):
    """简单的计算器工具"""
    
    def get_schema(self):
        return {
            "name": "calculator",
            "description": "执行简单的数学计算",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，例如: 2+3*4"
                    }
                },
                "required": ["expression"]
            }
        }
    
    def execute(self, kwargs):
        try:
            expression = kwargs.get("expression", "")
            result = eval(expression)
            return f"计算结果: {result}"
        except Exception as e:
            return f"计算错误: {str(e)}"

class SearchTool(Tool):
    """模拟搜索工具"""
    
    def get_schema(self):
        return {
            "name": "search",
            "description": "搜索信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词"
                    }
                },
                "required": ["query"]
            }
        }
    
    def execute(self, kwargs):
        query = kwargs.get("query", "")
        return f"搜索 '{query}' 的结果: [模拟结果] 这是关于 {query} 的信息..."

def test_basic_agent():
    """测试基础 Agent"""
    print("=" * 60)
    print("测试基础 Agent (agent.py)")
    print("=" * 60)
    
    # 创建工具
    tools = {
        "calculator": CalculatorTool(),
        "search": SearchTool()
    }
    
    # 创建 Agent
    agent = Agent(
        desc="你是一个智能助手，可以进行计算和搜索",
        model="qwen-max",
        tools=tools,
        verbose=True
    )
    
    # 创建环境
    env = Environment(initial_message="请帮我计算 (25 + 75) * 2 的结果")
    
    # 运行 Agent
    result = agent.run(env)
    
    print("\n" + "=" * 60)
    print(f"最终结果: {result.content}")
    print("=" * 60)

if __name__ == "__main__":
    test_basic_agent()
```

#### 运行测试

```bash
python test_agent.py
```

#### 预期行为

Agent 会：
1. 观察用户问题
2. 思考需要使用 calculator 工具
3. 调用工具执行计算
4. 返回最终结果

---

### 2. React.py 测试

**特点**: ReAct 模式实现，支持多工具并行调用

#### 创建测试文件

创建 `test_react.py`:

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试 ReAct Agent"""

from src.react import ReActAgent, Environment, Tool

class WeatherTool(Tool):
    """天气查询工具"""
    
    def get_schema(self):
        return {
            "name": "get_weather",
            "description": "查询指定城市的天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，例如: 北京、上海"
                    }
                },
                "required": ["city"]
            }
        }
    
    def execute(self, kwargs):
        city = kwargs.get("city", "")
        # 模拟返回
        return f"{city}的天气: 晴天，温度 25°C，湿度 60%"

class TimeTool(Tool):
    """时间查询工具"""
    
    def get_schema(self):
        return {
            "name": "get_time",
            "description": "获取当前时间",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    
    def execute(self, kwargs):
        from datetime import datetime
        return f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

def test_react_agent():
    """测试 ReAct Agent"""
    print("=" * 60)
    print("测试 ReAct Agent (react.py)")
    print("=" * 60)
    
    # 创建工具
    tools = {
        "get_weather": WeatherTool(),
        "get_time": TimeTool()
    }
    
    # 创建 Agent
    agent = ReActAgent(
        description="你是一个智能助手，可以查询天气和时间",
        model="qwen-max",
        tools=tools,
        verbose=True
    )
    
    # 创建环境
    env = Environment(initial_message="请告诉我现在的时间和北京的天气")
    
    # 运行 Agent
    result = agent.run(env)
    
    print("\n" + "=" * 60)
    print(f"最终结果: {result.content}")
    print("=" * 60)

if __name__ == "__main__":
    test_react_agent()
```

#### 运行测试

```bash
python test_react.py
```

---

### 3. ReActAgent 测试

**特点**: 现代化 ReAct Agent，使用标准 OpenAI 格式

#### 创建测试文件

创建 `test_react_agent.py`:

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试现代化 ReAct Agent"""

from src.react_agent import ReActAgent, Environment, Tool

class DataQueryTool(Tool):
    """数据查询工具"""
    
    name: str = "query_database"
    description: str = "从数据库查询信息"
    parameters: dict = {
        "type": "object",
        "properties": {
            "table": {
                "type": "string",
                "description": "表名"
            },
            "condition": {
                "type": "string",
                "description": "查询条件"
            }
        },
        "required": ["table"]
    }
    
    def execute(self, **kwargs):
        table = kwargs.get("table", "")
        condition = kwargs.get("condition", "all")
        
        # 模拟查询
        return f"从表 '{table}' 查询到 5 条记录 (条件: {condition})"

class FileOperationTool(Tool):
    """文件操作工具"""
    
    name: str = "file_operation"
    description: str = "执行文件操作（读取、写入、删除）"
    parameters: dict = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "description": "操作类型",
                "enum": ["read", "write", "delete"]
            },
            "filepath": {
                "type": "string",
                "description": "文件路径"
            }
        },
        "required": ["operation", "filepath"]
    }
    
    def execute(self, **kwargs):
        operation = kwargs.get("operation", "")
        filepath = kwargs.get("filepath", "")
        
        # 模拟操作
        return f"已执行 {operation} 操作: {filepath}"

def test_modern_react_agent():
    """测试现代化 ReAct Agent"""
    print("=" * 60)
    print("测试现代化 ReAct Agent (react_agent.py)")
    print("=" * 60)
    
    # 创建工具
    tools = [
        DataQueryTool(),
        FileOperationTool()
    ]
    
    # 创建 Agent
    agent = ReActAgent(
        description="你是一个数据处理助手，可以查询数据库和操作文件",
        model="qwen-max-latest",
        tools=tools
    )
    
    # 创建环境
    env = Environment(
        initial_message="请从 users 表查询所有活跃用户，并将结果写入 active_users.txt 文件"
    )
    
    # 运行 Agent (最多 20 步)
    result = agent.run(env, max_steps=20)
    
    print("\n" + "=" * 60)
    print(f"最终结果: {result.content}")
    print("=" * 60)

if __name__ == "__main__":
    test_modern_react_agent()
```

#### 运行测试

```bash
python test_react_agent.py
```

#### 关键特性

- ✅ 使用标准 OpenAI 工具调用格式
- ✅ 支持多工具并行调用
- ✅ 内置防卡住机制
- ✅ 自动终止工具（terminate）
- ✅ 使用 Pydantic 进行数据验证

---

### 4. RACAgent 测试

**特点**: RAC Agent（Reason-Action-Check），带一致性检查

#### 创建测试文件

创建 `test_rac_agent.py`:

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试 RAC Agent（带一致性检查）"""

from src.rac_agent import RACAgent
from src.react_agent import Environment, Tool

class APICallTool(Tool):
    """API 调用工具"""
    
    name: str = "call_api"
    description: str = "调用外部 API"
    parameters: dict = {
        "type": "object",
        "properties": {
            "endpoint": {
                "type": "string",
                "description": "API 端点"
            },
            "method": {
                "type": "string",
                "description": "HTTP 方法",
                "enum": ["GET", "POST", "PUT", "DELETE"]
            },
            "data": {
                "type": "object",
                "description": "请求数据"
            }
        },
        "required": ["endpoint", "method"]
    }
    
    def execute(self, **kwargs):
        endpoint = kwargs.get("endpoint", "")
        method = kwargs.get("method", "GET")
        data = kwargs.get("data", {})
        
        # 模拟 API 调用
        return f"API 调用成功: {method} {endpoint}, 返回状态码: 200"

class DataValidationTool(Tool):
    """数据验证工具"""
    
    name: str = "validate_data"
    description: str = "验证数据格式和有效性"
    parameters: dict = {
        "type": "object",
        "properties": {
            "data": {
                "type": "string",
                "description": "要验证的数据"
            },
            "schema": {
                "type": "string",
                "description": "验证规则"
            }
        },
        "required": ["data", "schema"]
    }
    
    def execute(self, **kwargs):
        data = kwargs.get("data", "")
        schema = kwargs.get("schema", "")
        
        # 模拟验证
        return f"数据验证通过: {data[:50]}..."

def test_rac_agent():
    """测试 RAC Agent"""
    print("=" * 60)
    print("测试 RAC Agent (rac_agent.py)")
    print("=" * 60)
    
    # 创建工具
    tools = [
        APICallTool(),
        DataValidationTool()
    ]
    
    # 创建 RAC Agent
    agent = RACAgent(
        description="你是一个 API 集成助手，可以调用 API 并验证数据",
        model="qwen-max-latest",
        tools=tools,
        check_threshold=0.7  # 一致性检查阈值
    )
    
    # 创建环境
    env = Environment(
        initial_message="请调用 /api/users 接口获取用户列表，并验证返回的数据格式"
    )
    
    # 运行 Agent（启用一致性检查）
    result = agent.run(
        env, 
        max_steps=50,
        enable_check=True  # 启用一致性检查
    )
    
    print("\n" + "=" * 60)
    print(f"最终结果: {result.content}")
    print("=" * 60)

def test_rac_agent_without_check():
    """测试 RAC Agent（禁用检查）"""
    print("\n" + "=" * 60)
    print("测试 RAC Agent (禁用一致性检查)")
    print("=" * 60)
    
    tools = [
        APICallTool(),
        DataValidationTool()
    ]
    
    agent = RACAgent(
        description="你是一个 API 集成助手",
        model="qwen-max-latest",
        tools=tools
    )
    
    env = Environment(
        initial_message="请调用 /api/health 接口检查服务状态"
    )
    
    # 禁用一致性检查
    result = agent.run(
        env, 
        max_steps=20,
        enable_check=False  # 禁用检查，行为类似 ReActAgent
    )
    
    print("\n" + "=" * 60)
    print(f"最终结果: {result.content}")
    print("=" * 60)

if __name__ == "__main__":
    # 测试 1: 启用一致性检查
    test_rac_agent()
    
    # 测试 2: 禁用一致性检查
    test_rac_agent_without_check()
```

#### 运行测试

```bash
python test_rac_agent.py
```

#### 一致性检查机制

RAC Agent 会在每次 Action 后进行一致性检查：

1. **工具调用完整性**: 验证所有计划的工具是否都被执行
2. **执行数量匹配**: 检查推理中的工具数量与实际执行数量是否一致
3. **执行错误检测**: 识别工具执行过程中的错误
4. **推理逻辑连贯性**: 验证思考内容是否提及实际使用的工具
5. **终止逻辑验证**: 确保终止调用与完成消息的一致性

如果检查失败（一致性分数 < 0.7），Agent 会：
- 记录失败原因和改进建议
- 将失败消息添加到环境
- 继续下一轮推理，尝试修正

---

## 常见问题

### Q1: API Key 无效或过期

**错误信息**: `Authentication failed` 或 `Invalid API Key`

**解决方法**:
1. 检查 `config/config.json` 中的 `api_key` 是否正确
2. 访问 [DashScope 控制台](https://dashscope.console.aliyun.com/) 验证 API Key
3. 确保账户有足够的额度

### Q2: 数据库初始化失败

**错误信息**: `OperationalError: unable to open database file`

**解决方法**:
```bash
# 确保 data 目录存在
mkdir -p data

# 检查目录权限
chmod 755 data

# 重新运行初始化脚本
python test_db_init.py
```

### Q3: 依赖包安装失败

**错误信息**: `ERROR: Could not find a version that satisfies the requirement...`

**解决方法**:
```bash
# 升级 pip
pip install --upgrade pip

# 使用国内镜像源
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# 或者使用清华镜像
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q4: Agent 卡住不响应

**可能原因**:
1. LLM 响应超时
2. 网络连接问题
3. Agent 进入重复思考循环

**解决方法**:
- 检查网络连接
- 减少 `max_steps` 参数
- 使用 `verbose=True` 查看详细日志
- 对于 ReActAgent 和 RACAgent，内置了防卡住机制

### Q5: 如何选择合适的 Agent？

| Agent 类型 | 适用场景 | 优点 | 缺点 |
|-----------|---------|------|------|
| **agent.py** | 简单任务 | 轻量、易理解 | 功能有限 |
| **react.py** | 中等复杂任务 | 支持多工具 | 非标准格式 |
| **react_agent.py** | 大多数场景 | 标准化、功能完整 | 无验证机制 |
| **rac_agent.py** | 关键任务 | 带一致性检查 | 性能开销较大 |

**推荐**:
- 新项目首选 **react_agent.py**
- 对可靠性要求高的场景使用 **rac_agent.py**

### Q6: 如何自定义工具？

所有工具都需要继承 `Tool` 基类：

```python
from src.react_agent import Tool

class MyCustomTool(Tool):
    name: str = "my_tool"
    description: str = "我的自定义工具"
    parameters: dict = {
        "type": "object",
        "properties": {
            "param1": {
                "type": "string",
                "description": "参数1说明"
            }
        },
        "required": ["param1"]
    }
    
    def execute(self, **kwargs):
        # 实现你的逻辑
        param1 = kwargs.get("param1")
        result = f"处理结果: {param1}"
        return result
```

### Q7: 如何查看详细日志？

```python
# 方法 1: 使用 verbose 参数
agent = ReActAgent(
    description="...",
    model="qwen-max-latest",
    tools=tools
)

# 方法 2: 配置 loguru 日志级别
from loguru import logger
logger.add("agent.log", level="DEBUG")
```

---

## 📚 进阶阅读

- [README.md](README.md) - 架构详细说明
- [src/agent.py](src/agent.py) - 基础 Agent 源码
- [src/react_agent.py](src/react_agent.py) - ReAct Agent 源码  
- [src/rac_agent.py](src/rac_agent.py) - RAC Agent 源码

---

## 🤝 贡献

如果你在测试过程中发现问题或有改进建议，欢迎提交 Issue 或 Pull Request。

---

## 📄 许可证

请参考项目根目录的 LICENSE 文件。

---

**祝测试顺利！** 🎉
