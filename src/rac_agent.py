from abc import ABC
from typing import Optional, List

from loguru import logger
from pydantic import BaseModel, Field

from .event import emit_event, EventType
from .llm import LlmCaller
from .react_agent import (
    MessageType, Message, Tool, TerminateTool, ActionResult,
    Action, ReasonAction, Memory, Environment
)


class CheckResult(BaseModel):
    """Check result for validation"""
    success: bool
    consistency_score: float  # 0-1 score indicating consistency
    issues: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)

class CheckAction(Action):
    """检查动作 - 验证推理和执行的一致性"""

    def __init__(self, reason_action: ReasonAction, execution_results: List[Message]):
        """初始化检查动作

        Args:
            reason_action: 要检查的推理动作
            execution_results: 执行动作的结果
        """
        self.type = "check"
        self.reason_action = reason_action
        self.execution_results = execution_results
        self.thought = reason_action.thought
        self.tool_calls = reason_action.tool_calls
        
    def execute(self, env: "Environment") -> ActionResult:
        """执行检查动作

        Args:
            env: 交互环境

        Returns:
            包含检查结果的 ActionResult
        """
        try:
            check_result = self._perform_consistency_check()
            
            if check_result.success:
                emit_event(EventType.AGENT, f"[Check] ✓ Consistency check passed (score: {check_result.consistency_score:.2f})")
                return ActionResult(
                    success=True,
                    value=check_result
                )
            else:
                issues_str = "; ".join(check_result.issues)
                emit_event(EventType.AGENT, f"[Check] ✗ Consistency check failed: {issues_str}")
                
                # Add failure message to environment
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
            return ActionResult(
                success=False,
                value=None,
                error=str(e)
            )
    
    def _perform_consistency_check(self) -> CheckResult:
        """执行推理和执行之间的一致性检查

        Returns:
            检查结果
        """
        issues = []
        recommendations = []
        consistency_score = 1.0
        
        # 检查 1: 验证所有计划的工具调用是否都已执行
        planned_tools = set()
        for tool_call in self.tool_calls:
            tool_name = tool_call.get('function', {}).get('name')
            if tool_name:
                planned_tools.add(tool_name)
        
        executed_tools = set()
        for result in self.execution_results:
            if result.type == MessageType.TOOL:
                tool_name = result.content.get('name')
                if tool_name:
                    executed_tools.add(tool_name)
        
        missing_tools = planned_tools - executed_tools
        if missing_tools:
            issues.append(f"计划使用的工具 {missing_tools} 未被执行")
            recommendations.append("重新执行缺失的工具调用")
            consistency_score -= 0.3
        
        # 检查 2: 验证推理是否与执行顺序匹配
        if len(self.tool_calls) != len([r for r in self.execution_results if r.type == MessageType.TOOL]):
            issues.append("推理中的工具调用数量与实际执行数量不匹配")
            recommendations.append("检查工具调用的执行逻辑")
            consistency_score -= 0.2
        
        # 检查 3: 检查执行错误
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
        
        # 检查 4: 验证推理逻辑的连贯性
        if self.thought:
            # 简单启发式：检查思考内容是否提及实际使用的工具
            thought_lower = self.thought.lower()
            for tool_name in executed_tools:
                if tool_name.lower() not in thought_lower:
                    issues.append(f"推理内容中未提及实际使用的工具: {tool_name}")
                    recommendations.append("确保推理过程与实际行动一致")
                    consistency_score -= 0.1
        
        # 检查 5: 验证终止逻辑
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
        
        # 确保一致性分数在范围内
        consistency_score = max(0.0, min(1.0, consistency_score))

        # 确定总体成功（阈值：0.7）
        success = consistency_score >= 0.7 and len(issues) == 0
        
        return CheckResult(
            success=success,
            consistency_score=consistency_score,
            issues=issues,
            recommendations=recommendations
        )
    
    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "thought": self.thought,
            "tool_calls": self.tool_calls,
            "execution_results_count": len(self.execution_results)
        }

CHECK_PROMPT = """
Based on the reasoning and execution results, perform a consistency check to ensure:
1. All planned actions were executed correctly
2. The reasoning logic matches the actual execution
3. No critical steps were skipped
4. Tool calls were executed in the expected order
5. Any errors or inconsistencies are identified

Focus on validating the coherence between intention and execution.
"""

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

class RACAgent(ABC):
    """RAC Agent - 带一致性验证的推理-行动-检查智能体"""

    def __init__(
        self,
        description: str,
        model: str = "qwen-max-latest",
        tools: Optional[list[Tool]] = None,
        check_threshold: float = 0.7
    ):
        """初始化 RAC Agent

        Args:
            description: Agent 描述
            model: 使用的 LLM 模型名称
            tools: 可用工具列表
            check_threshold: 一致性检查阈值（0-1）
        """
        self.llm_caller = LlmCaller(f"{description}\n\n{COT_PROMPT}", model)
        self.check_llm_caller = LlmCaller(f"{description}\n\n{CHECK_PROMPT}", model)
        self.memory = Memory()
        self.tools = tools + [TerminateTool()] \
            if tools is not None \
            else [TerminateTool()]
        self.check_threshold = check_threshold

    def run(
        self,
        env: Environment,
        max_steps: Optional[int] = 50,
        enable_check: bool = True
    ) -> Message:
        """Agent 主循环（推理-行动-检查）

        执行流程：
        1. Observe: 从环境中观察消息
        2. Reason: 推理并决定下一步动作
        3. Action: 执行动作并将结果放回环境
        4. Check: 验证推理和执行之间的一致性

        Args:
            env: Agent 交互的环境
            max_steps: 最大执行步数
            enable_check: 是否启用一致性检查

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
            # 观察
            self.observe(env)

            # 推理
            reason_action = self.reason()

            # 行动
            action_results = self.act(reason_action, env)

            # 检查（如果启用且不是终止动作）
            if enable_check and not self._is_terminating_action(reason_action):
                check_result = self.check(reason_action, action_results, env)
                if not check_result.success:
                    # 如果检查失败，添加失败消息并继续
                    env.add_message(check_result.value)
                    current_step += 1
                    if max_steps is not None and current_step >= max_steps:
                        break
                    continue

            # 检查是否完成
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

    def reason(self) -> ReasonAction:
        """推理并决定下一步动作

        Returns:
            推理动作
        """
        # 回忆记忆
        llm_messages = []
        for message in self.memory.messages:
            llm_messages.append(message.to_llm_message())

        # 聚焦于最新消息
        focus_prompt = FOCUS_PROMPT
        # 检查是否卡住
        if self._is_stuck():
            focus_prompt = self._handle_stuck(focus_prompt)

        llm_messages.append(
            Message(
                type=MessageType.USER,
                content=focus_prompt
            ).to_llm_message()
        )

        # 调用 LLM 进行推理
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

    def act(self, action: ReasonAction, env: Environment) -> List[Message]:
        """执行动作并将结果放入环境

        Args:
            action: 要执行的推理动作
            env: Agent 交互的环境

        Returns:
            执行结果消息列表
        """
        # 执行动作
        action_result = action.execute(env)

        # 将动作结果放入环境
        if not action_result.success or not action_result.value:
            error_message = Message(
                type=MessageType.USER,
                content=f"Action executed error because of {action_result.error}. The action is {action.to_dict()}"
            )
            env.add_message(error_message)
            return [error_message]

        env.add_messages(action_result.value)
        return action_result.value

    def check(self, reason_action: ReasonAction, execution_results: List[Message], env: Environment) -> ActionResult:
        """检查推理和执行之间的一致性

        Args:
            reason_action: 推理动作
            execution_results: 执行结果
            env: 交互环境

        Returns:
            检查结果
        """
        check_action = CheckAction(reason_action, execution_results)
        return check_action.execute(env)

    def _is_terminating_action(self, action: ReasonAction) -> bool:
        """检查动作是否包含终止工具调用

        Args:
            action: 推理动作

        Returns:
            是否为终止动作
        """
        return any(
            tool_call.get('function', {}).get('name') == 'terminate'
            for tool_call in action.tool_calls
        )

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