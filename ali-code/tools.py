# ==========================================
#  Step 5: 工具定义模块
#  每个工具包含两部分：
#  1. 函数实现（真正干活的代码）
#  2. JSON Schema 描述（告诉 LLM 怎么调用）
# ==========================================
import json
import datetime
import math


# ==========================================
#  工具 1: 计算器
# ==========================================
def calculator(expression: str) -> str:
    """
    执行数学计算
    Args:
        expression: 数学表达式，如 "2 + 3 * 4"
    Returns:
        计算结果
    """
    try:
        # 安全计算：只允许数字和基本运算符
        allowed_chars = set("0123456789+-*/().% ")
        if not all(c in allowed_chars for c in expression):
            return "错误：表达式包含不允许的字符"
        result = eval(expression, {"__builtins__": {}}, {"math": math})
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算出错: {e}"


CALCULATOR_SCHEMA = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "执行数学计算，支持加减乘除、括号、百分号等运算。当用户需要计算数学题时使用。",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "数学表达式，例如: 123 * 456, (10 + 5) / 3"
                }
            },
            "required": ["expression"]
        }
    }
}


# ==========================================
#  工具 2: 获取当前时间
# ==========================================
def get_current_time(timezone: str = "Asia/Shanghai") -> str:
    """
    获取当前日期和时间
    Args:
        timezone: 时区，默认 Asia/Shanghai
    Returns:
        当前时间字符串
    """
    now = datetime.datetime.now()
    weekday_map = {
        0: "星期一", 1: "星期二", 2: "星期三",
        3: "星期四", 4: "星期五", 5: "星期六", 6: "星期日"
    }
    weekday = weekday_map[now.weekday()]
    return f"当前时间: {now.strftime('%Y年%m月%d日 %H:%M:%S')} ({weekday}) 时区: {timezone}"


GET_TIME_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "获取当前的日期和时间。当用户问现在几点、今天几号、星期几时使用。",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "时区，默认是 Asia/Shanghai",
                    "default": "Asia/Shanghai"
                }
            },
            "required": []
        }
    }
}


# ==========================================
#  工具 3: 执行系统命令（只读，安全命令白名单）
# ==========================================
def run_system_command(command: str) -> str:
    """
    执行安全的系统命令（白名单限制）
    Args:
        command: 要执行的命令
    Returns:
        命令输出
    """
    # 安全白名单：只允许只读的信息查询命令
    allowed_commands = {
        "dir": "列出当前目录文件",
        "date": "显示系统日期",
        "time": "显示系统时间",
        "ver": "显示 Windows 版本",
        "whoami": "显示当前用户名",
        "hostname": "显示主机名",
        "ipconfig": "显示网络配置（简要）",
        "tasklist": "列出运行中的进程",
    }

    cmd_base = command.strip().split()[0].lower() if command.strip() else ""

    if cmd_base not in allowed_commands:
        return f"安全限制：不允许执行 '{cmd_base}' 命令。\n允许的命令: {', '.join(allowed_commands.keys())}"

    try:
        import subprocess
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=5,
            encoding="gbk"  # Windows CMD 默认 GBK
        )
        output = result.stdout + result.stderr
        # 限制输出长度
        if len(output) > 2000:
            output = output[:2000] + "\n... (输出过长，已截断)"
        return output if output else "(命令执行完成，无输出)"
    except Exception as e:
        return f"命令执行出错: {e}"


RUN_CMD_SCHEMA = {
    "type": "function",
    "function": {
        "name": "run_system_command",
        "description": "执行系统命令来查看系统信息。只能执行白名单内的安全命令（dir, date, time, ver, whoami, hostname, ipconfig, tasklist）。",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "要执行的系统命令，例如: dir, whoami, ipconfig"
                }
            },
            "required": ["command"]
        }
    }
}


# ==========================================
#  工具 4: 知识库检索 (RAG)
# ==========================================
def search_knowledge_base(query: str, top_k: int = 3) -> str:
    """
    从已加载的文档知识库中检索相关内容
    Args:
        query: 查询问题
        top_k: 返回最相关的前几条结果
    Returns:
        检索到的相关文档内容
    """
    from rag import get_retriever
    retriever = get_retriever()

    if retriever.chunk_count == 0:
        return "知识库为空，请先使用 /add-docs 命令添加文档。"

    results = retriever.search(query, top_k=top_k)

    if not results:
        return "未找到相关内容。"

    output = f"从知识库中找到 {len(results)} 段相关内容：\n\n"
    for i, r in enumerate(results, 1):
        output += f"--- 第 {i} 段 (来源: {r['source']}, 相关度: {r['score']}) ---\n"
        output += r["chunk"] + "\n\n"

    return output


SEARCH_KB_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": "从已加载的文档知识库中检索相关信息。当用户询问文档内容、项目信息、或者需要根据文档回答问题时使用。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "要查询的问题或关键词"
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回最相关的前几段，默认 3 段",
                    "default": 3
                }
            },
            "required": ["query"]
        }
    }
}


# ==========================================
#  工具注册表
# ==========================================

# 所有工具的 Schema 列表（发给 LLM 看的）
ALL_TOOL_SCHEMAS = [
    CALCULATOR_SCHEMA,
    GET_TIME_SCHEMA,
    RUN_CMD_SCHEMA,
    SEARCH_KB_SCHEMA,
]

# 工具名 → 函数实现 的映射（真正执行用的）
TOOL_REGISTRY = {
    "calculator": calculator,
    "get_current_time": get_current_time,
    "run_system_command": run_system_command,
    "search_knowledge_base": search_knowledge_base,
}


def execute_tool(tool_name: str, tool_args: dict) -> str:
    """
    根据工具名和参数执行工具
    Args:
        tool_name: 工具名称
        tool_args: 工具参数字典
    Returns:
        工具执行结果字符串
    """
    if tool_name not in TOOL_REGISTRY:
        return f"错误：未知工具 '{tool_name}'"

    try:
        func = TOOL_REGISTRY[tool_name]
        result = func(**tool_args)
        return str(result)
    except Exception as e:
        return f"工具执行出错 ({tool_name}): {e}"
