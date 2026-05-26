import json
import logging
import os
from typing import Annotated, Literal
from langchain_core.messages import AIMessage, HumanMessage,  SystemMessage, ToolMessage
from langgraph.types import Command, interrupt
from langchain_openai import ChatOpenAI
import httpx
from state import State
from prompts import *
from tools import *


# AutoDL 代理配置（如果需要）
# 设置环境变量示例：export http_proxy=http://127.0.0.1:7890
_proxy = os.getenv("HTTP_PROXY") or os.getenv("http_proxy") or os.getenv("HTTPS_PROXY") or os.getenv("https_proxy") or None

# 验证代理地址是否有效（端口必须是数字）
_valid_proxy = None
if _proxy:
    try:
        test_url = httpx.URL(_proxy)
        if test_url.port and str(test_url.port).isdigit():
            _valid_proxy = _proxy
        else:
            logging.warning(f"代理端口无效 ({_proxy})，将不使用代理")
    except Exception as e:
        logging.warning(f"代理地址无效 ({_proxy})，将不使用代理: {e}")

# 临时清除无效环境变量，防止 httpx 自动读取导致报错
_removed_env_keys = []
for key in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]:
    val = os.environ.pop(key, None)
    if val is not None and val != (_valid_proxy or ""):
        _removed_env_keys.append((key, val))

if _valid_proxy:
    for k, v in _removed_env_keys:
        if v == _valid_proxy:
            os.environ[k] = v
    http_client = httpx.Client(proxy=_valid_proxy, timeout=120.0)
else:
    http_client = httpx.Client(timeout=120.0)

llm = ChatOpenAI(
    model="moonshot-v1-128k",
    temperature=0.7,
    base_url='https://api.moonshot.cn/v1',
    api_key='sk-FSgpFi6NTyJyVKzYOeYzJL6T83AY1lBJZxihTLrIsJg7Vuas',
    http_client=http_client,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
hander = logging.StreamHandler()
hander.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
hander.setFormatter(formatter)
logger.addHandler(hander)

# 自修复配置
MAX_REPAIR_ATTEMPTS = 3  # 每个文件最多修复尝试次数


def _generate_fallback_instruction(error_text: str, file_path: str, step_description: str) -> str:
    """生成自修复失败后的降级指令，引导 LLM 做出合理的降级决策。
    
    降级策略按优先级：
    1. 简化/注释掉出错部分，让剩余代码能跑通
    2. 用更基础的 API 替换高级用法
    3. 如果完全无法修复，输出有意义的中间结果后跳过
    """
    return f"""
<self_repair_failed>
⚠️ 代码自修复已达到最大重试次数（{MAX_REPAIR_ATTEMPTS}次），仍未成功。

## 错误文件
`{file_path}`

## 持续存在的错误
```
{error_text}
```

## 请立即选择以下一种降级策略继续执行：

### 策略 A：精简代码（推荐）
- 注释掉或删除报错的那部分代码
- 保留能正常运行的其余逻辑
- 用更简单的方式实现近似功能（如用 print 替代复杂绘图）

### 策略 B：替换依赖
- 如果是库/版本问题，用已安装的替代库重写出错部分
- 例如：seaborn → matplotlib，plotly → matplotlib

### 策略 C：输出部分结果后跳过
- 如果当前步骤的核心逻辑无法运行
- 先保存已有的中间计算结果到文件
- 在总结中说明该步骤因技术限制未完成

## 要求
1. 必须使用 str_replace 或 create_file 工具修改代码
2. 修改后必须重新 shell_exec 执行确认不再报错
3. 不要再尝试原来的修复方向，换个思路
</self_repair_failed>

当前步骤描述: {step_description}
"""


def _detect_python_error(tool_result: dict) -> tuple[bool, str]:
    """检测 shell_exec 结果中是否包含 Python 执行错误。
    
    Returns:
        (has_error, error_text): 是否有错误，错误文本
    """
    if not isinstance(tool_result, dict):
        return False, ""
    
    # 检查 message.stderr 或 error 字段
    stderr = ""
    if "message" in tool_result and isinstance(tool_result["message"], dict):
        stderr = tool_result["message"].get("stderr", "")
    elif "error" in tool_result:
        err = tool_result["error"]
        stderr = err.get("stderr", "") if isinstance(err, dict) else str(err)
    
    if not stderr:
        return False, ""
    
    # 检测 Python 错误特征
    error_indicators = ["Traceback", "Error", "Exception", "SyntaxError", 
                        "NameError", "TypeError", "ValueError", "ImportError",
                        "ModuleNotFoundError", "FileNotFoundError", "KeyError",
                        "IndexError", "AttributeError", "IndentationError"]
    
    has_error = any(indicator in stderr for indicator in error_indicators)
    return has_error, stderr


def _extract_file_from_command(command: str) -> str | None:
    """从 shell_exec 命令中提取执行的 Python 文件路径。"""
    import re
    # 匹配 python/python3 xxx.py 形式
    match = re.search(r'(?:python|python3)\s+([\w\./\-]+\.py)', command)
    return match.group(1) if match else None


def _attempt_self_repair(messages: list, error_output: str, file_path: str, 
                         tools: dict, llm_with_tools, max_attempts: int = MAX_REPAIR_ATTEMPTS) -> bool:
    """尝试自修复代码错误。
    
    Args:
        messages: 当前对话消息列表
        error_output: 错误输出内容
        file_path: 出错的文件路径
        tools: 可用工具字典
        llm_with_tools: 绑定了工具的 LLM
        max_attempts: 最大修复尝试次数
    
    Returns:
        bool: 是否修复成功
    """
    logger.warning(f"[自修复] 开始修复文件: {file_path}")
    
    repair_messages = messages.copy()
    repair_messages.append(HumanMessage(content=SELF_REPAIR_PROMPT.format(
        error_output=error_output,
        file_path=file_path
    )))
    
    for attempt in range(1, max_attempts + 1):
        logger.info(f"[自修复] 第 {attempt}/{max_attempts} 次修复尝试...")
        
        try:
            raw_response = llm_with_tools.invoke(repair_messages)
            response = raw_response.model_dump_json(indent=4, exclude_none=True)
            response = json.loads(response)
            
            made_repair = False
            last_command = None
            
            if response.get('tool_calls'):
                _reasoning = response.get('reasoning_content', '') or ''
                _ai_msg = AIMessage(
                    content=raw_response.content,
                    tool_calls=response['tool_calls'],
                    additional_kwargs={'reasoning_content': _reasoning} if _reasoning else {},
                )
                repair_messages += [_ai_msg]
                
                for tool_call in response['tool_calls']:
                    t_name = tool_call['name']
                    t_args = tool_call['args']
                    t_result = tools[t_name].invoke(t_args)
                    logger.info(f"[自修复] 工具调用: {t_name}, 结果: {str(t_result)[:200]}")
                    repair_messages += [ToolMessage(
                        content=f"tool_name:{t_name},tool_result:{t_result}", 
                        tool_call_id=tool_call['id']
                    )]
                    
                    if t_name == 'str_replace':
                        made_repair = True
                    elif t_name == 'shell_exec':
                        last_command = t_args.get('command', '')
                        
            elif '**' in response.get('content', ''):
                tool_call_text = response['content'].split('**')[-1].split('**')[0].strip()
                try:
                    tool_call = json.loads(tool_call_text)
                    t_name = tool_call['name']
                    t_args = tool_call['args']
                    t_result = tools[t_name].invoke(t_args)
                    logger.info(f"[自修复] 工具调用(text): {t_name}")
                    repair_messages += [AIMessage(content=extract_answer(response['content']))]
                    repair_messages += [HumanMessage(content=f"tool_result:{t_result}")]
                    
                    if t_name == 'str_replace':
                        made_repair = True
                    elif t_name == 'shell_exec':
                        last_command = t_args.get('command', '')
                except (json.JSONDecodeError, KeyError):
                    pass
            
            # 如果做了代码修改，自动重新运行验证
            if made_repair and file_path:
                verify_cmd = f"python {file_path}"
                logger.info(f"[自修复] 验证运行: {verify_cmd}")
                verify_result = tools['shell_exec'].invoke({"command": verify_cmd})
                repair_messages += [HumanMessage(content=f"验证执行结果: {verify_result}")]
                
                has_err, err_text = _detect_python_error(verify_result)
                if not has_err:
                    logger.info(f"[自修复] 修复成功！")
                    # 将修复过程同步回主消息列表
                    messages.extend(repair_messages[len(messages):])
                    return True
                else:
                    logger.warning(f"[自修复] 验证仍失败: {err_text[:300]}")
                    repair_messages.append(HumanMessage(
                        content=f"修复后仍有错误，请继续修正:\n{err_text}"
                    ))
            
            # 如果直接重新执行了 shell_exec（LLM 自己决定重跑）
            if last_command:
                # 从历史消息获取最后一次 shell_exec 结果
                pass
            
        except Exception as e:
            logger.error(f"[自修复] 第 {attempt} 次尝试异常: {e}")
            repair_messages.append(HumanMessage(content=f"修复过程出错: {e}，请重试"))
    
    logger.error(f"[自修复] 达到最大重试次数 ({max_attempts})，修复失败")
    messages.extend(repair_messages[len(messages):])
    return False

def extract_json(text):
    if '```json' not in text:
        return text
    text = text.split('```json')[1].split('```')[0].strip()
    return text

def _looks_like_tool_call_json(text: str) -> bool:
    """快速预判文本是否像工具调用的 JSON 格式，避免无效的 json.loads 尝试。"""
    text = text.strip()
    if not text or len(text) < 5:
        return False
    # 合法的 tool call JSON 必须以 { 开头，且包含 "name" 和 "args" 字段
    if not text.startswith('{'):
        return False
    return '"name"' in text and '"args"' in text


def extract_answer(text):
    if '**' in text:
        answer = text.split("**")[-1]
        return answer.strip()
    
    return text

def create_planner_node(state: State):
    logger.info("***正在运行Create Planner node***")
    messages = [SystemMessage(content=PLAN_SYSTEM_PROMPT), HumanMessage(content=PLAN_CREATE_PROMPT.format(user_message = state['user_message']))]
    response = llm.invoke(messages)
    response = response.model_dump_json(indent=4, exclude_none=True)
    response = json.loads(response)
    plan = json.loads(extract_json(extract_answer(response['content'])))
    state['messages'] += [AIMessage(content=json.dumps(plan, ensure_ascii=False))]
    return Command(goto="execute", update={"plan": plan})

def update_planner_node(state: State):
    logger.info("***正在运行Update Planner node***")
    plan = state['plan']
    goal = plan['goal']
    state['messages'].extend([SystemMessage(content=PLAN_SYSTEM_PROMPT), HumanMessage(content=UPDATE_PLAN_PROMPT.format(plan = plan, goal=goal))])
    messages = state['messages']
    while True:
        try:
            response = llm.invoke(messages)
            response = response.model_dump_json(indent=4, exclude_none=True)
            response = json.loads(response)
            plan = json.loads(extract_json(extract_answer(response['content'])))
            state['messages']+=[AIMessage(content=json.dumps(plan, ensure_ascii=False))]
            return Command(goto="execute", update={"plan": plan})
        except Exception as e:
            messages += [HumanMessage(content=f"json格式错误:{e}")]
            
def execute_node(state: State):
    logger.info("***正在运行execute_node***")
  
    plan = state['plan']
    steps = plan['steps']
    current_step = None
    current_step_index = 0
    
    # 获取第一个未完成STEP
    for i, step in enumerate(steps):
        status = step['status']
        if status == 'pending':
            current_step = step
            current_step_index = i
            break
        
    logger.info(f"当前执行STEP:{current_step}")
    
    ## 此处只是简单跳转到report节点，实际应该根据当前STEP的描述进行判断
    if current_step is None or current_step_index == len(steps)-1:
        return Command(goto='report')
    
    messages = state['observations'] + [SystemMessage(content=EXECUTE_SYSTEM_PROMPT), HumanMessage(content=EXECUTION_PROMPT.format(user_message=state['user_message'], step=current_step['description']))]
    
    tool_result = None
    last_shell_command = None  # 追踪最近的 shell_exec 命令
    llm_with_tools = llm.bind_tools([create_file, str_replace, shell_exec])
    
    while True:
        raw_response = llm_with_tools.invoke(messages)
        response = raw_response.model_dump_json(indent=4, exclude_none=True)
        response = json.loads(response)
        tools = {"create_file": create_file, "str_replace": str_replace, "shell_exec": shell_exec}     
        
        if response.get('tool_calls'):
            # Kimi K2.6 thinking 模式：回传带 tool_calls 的消息时需保留 reasoning_content
            _reasoning = response.get('reasoning_content', '') or ''
            _ai_msg_for_history = AIMessage(
                content=raw_response.content,
                tool_calls=response['tool_calls'],
                additional_kwargs={'reasoning_content': _reasoning} if _reasoning else {},
            )
            messages += [_ai_msg_for_history]
            for tool_call in response['tool_calls']:
                tool_name = tool_call['name']
                tool_args = tool_call['args']
                tool_result = tools[tool_name].invoke(tool_args)
                logger.info(f"tool_name:{tool_name},tool_args:{tool_args}\ntool_result:{tool_result}")
                messages += [ToolMessage(content=f"tool_name:{tool_name},tool_args:{tool_args}\ntool_result:{tool_result}", tool_call_id=tool_call['id'])]
                
                # === 自修复：检测 shell_exec 执行错误 ===
                if tool_name == 'shell_exec':
                    last_shell_command = tool_args.get('command', '')
                    has_error, error_text = _detect_python_error(tool_result)
                    if has_error:
                        file_path = _extract_file_from_command(last_shell_command) or "未知文件"
                        logger.warning(f"[execute_node] 检测到 Python 执行错误，触发自修复: {file_path}")
                        repair_success = _attempt_self_repair(
                            messages=messages,
                            error_output=error_text,
                            file_path=file_path,
                            tools=tools,
                            llm_with_tools=llm_with_tools,
                        )
                        if repair_success:
                            messages.append(HumanMessage(content=f"[系统] 自修复成功，文件 {file_path} 已修复并可正常运行"))
                        else:
                            # === 降级模式：注入降级指令，让 LLM 在下一轮循环中执行降级策略 ===
                            fallback_msg = _generate_fallback_instruction(
                                error_text=error_text,
                                file_path=file_path,
                                step_description=current_step.get('description', '未知步骤'),
                            )
                            messages.append(HumanMessage(content=fallback_msg))
                            logger.warning(f"[execute_node] 已注入降级指令，LLM 将尝试降级策略")
        
        elif '**' in response.get('content', ''):
            tool_call_text = response['content'].split('**')[-1].split('**')[0].strip()
            if _looks_like_tool_call_json(tool_call_text):
                try:
                    tool_call = json.loads(tool_call_text)
                    
                    tool_name = tool_call['name']
                    tool_args = tool_call['args']
                    tool_result = tools[tool_name].invoke(tool_args)
                    logger.info(f"tool_name:{tool_name},tool_args:{tool_args}\ntool_result:{tool_result}")
                    messages += [AIMessage(content=extract_answer(response['content']))]
                    messages += [HumanMessage(content=f"tool_result:{tool_result}")]
                    
                    # === 自修复：检测 shell_exec 执行错误 (text-based 模式) ===
                    if tool_name == 'shell_exec':
                        last_shell_command = tool_args.get('command', '')
                        has_error, error_text = _detect_python_error(tool_result)
                        if has_error:
                            file_path = _extract_file_from_command(last_shell_command) or "未知文件"
                            logger.warning(f"[execute_node-text] 检测到 Python 执行错误，触发自修复: {file_path}")
                            repair_success = _attempt_self_repair(
                                messages=messages,
                                error_output=error_text,
                                file_path=file_path,
                                tools=tools,
                                llm_with_tools=llm_with_tools,
                            )
                            if repair_success:
                                messages.append(HumanMessage(content="[系统] 自修复成功"))
                            else:
                                # === 降级模式：注入降级指令 ===
                                fallback_msg = _generate_fallback_instruction(
                                    error_text=error_text,
                                    file_path=file_path,
                                    step_description=current_step.get('description', '未知步骤'),
                                )
                                messages.append(HumanMessage(content=fallback_msg))
                                
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"text-based tool call 解析失败，视为普通响应: {e}")
                    break
            else:
                # 提取到的内容不像 tool_call JSON，说明 LLM 只是用 ** 做了 markdown 加粗
                logger.debug(f"** 内容非工具调用格式（{len(tool_call_text)}字符），跳过解析")
                break
        else:    
            break
        
    logger.info(f"当前STEP执行总结:{extract_answer(response['content'])}")
    state['messages'] += [AIMessage(content=extract_answer(response['content']))]
    state['observations'] += [AIMessage(content=extract_answer(response['content']))]
    return Command(goto='update_planner', update={'plan': plan})



def report_node(state: State):
    """Report node that write a final report."""
    logger.info("***正在运行report_node***")
    
    observations = state.get("observations")
    messages = observations + [SystemMessage(content=REPORT_SYSTEM_PROMPT)]
    
    while True:
        raw_response = llm.bind_tools([create_file, shell_exec]).invoke(messages)
        response = raw_response.model_dump_json(indent=4, exclude_none=True)
        response = json.loads(response)
        tools = {"create_file": create_file, "shell_exec": shell_exec} 
        if response.get('tool_calls'):    
            # Kimi K2.6 thinking 模式：回传带 tool_calls 的消息时需保留 reasoning_content
            _reasoning = response.get('reasoning_content', '') or ''
            _ai_msg_for_history = AIMessage(
                content=raw_response.content,
                tool_calls=response['tool_calls'],
                additional_kwargs={'reasoning_content': _reasoning} if _reasoning else {},
            )
            messages += [_ai_msg_for_history]
            for tool_call in response['tool_calls']:
                tool_name = tool_call['name']
                tool_args = tool_call['args']
                tool_result = tools[tool_name].invoke(tool_args)
                logger.info(f"tool_name:{tool_name},tool_args:{tool_args}\ntool_result:{tool_result}")
                messages += [ToolMessage(content=f"tool_name:{tool_name},tool_args:{tool_args}\ntool_result:{tool_result}", tool_call_id=tool_call['id'])]
                
        elif '**' in response.get('content', ''):
            tool_call_text = response['content'].split('**')[-1].split('**')[0].strip()
            if _looks_like_tool_call_json(tool_call_text):
                try:
                    tool_call = json.loads(tool_call_text)
                    
                    tool_name = tool_call['name']
                    tool_args = tool_call['args']
                    tool_result = tools[tool_name].invoke(tool_args)
                    logger.info(f"tool_name:{tool_name},tool_args:{tool_args}\ntool_result:{tool_result}")
                    messages += [AIMessage(content=extract_answer(response['content']))]
                    messages += [HumanMessage(content=f"tool_result:{tool_result}")]
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"text-based tool call 解析失败，视为普通响应: {e}")
                    break
            else:
                logger.debug(f"** 内容非工具调用格式（{len(tool_call_text)}字符），跳过解析")
                break
        else:
            break
            
    return {"final_report": response['content']}
