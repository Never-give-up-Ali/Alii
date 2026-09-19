# ==========================================
#  终端 UI 模块 (Ali-code 风格，无表情)
# ==========================================
from colorama import init, Fore, Back, Style

init(autoreset=True)


class C:
    """颜色常量"""
    RESET = Style.RESET_ALL
    BOLD = Style.BRIGHT

    RED = Fore.RED
    GREEN = Fore.GREEN
    YELLOW = Fore.YELLOW
    BLUE = Fore.BLUE
    MAGENTA = Fore.MAGENTA
    CYAN = Fore.CYAN
    WHITE = Fore.WHITE
    GRAY = Fore.LIGHTBLACK_EX

    LRED = Fore.LIGHTRED_EX
    LGREEN = Fore.LIGHTGREEN_EX
    LYELLOW = Fore.LIGHTYELLOW_EX
    LBLUE = Fore.LIGHTBLUE_EX
    LMAGENTA = Fore.LIGHTMAGENTA_EX
    LCYAN = Fore.LIGHTCYAN_EX


def print_banner():
    """打印启动 Banner"""
    banner = f"""
{C.BOLD}{C.LCYAN}==================================================
              Ali-code  v1.0
         Command Line AI Assistant
=================================================={C.RESET}
{C.GRAY}输入 /help 查看帮助，输入 /exit 退出{C.RESET}
"""
    print(banner)


def print_help_menu():
    """打印帮助菜单"""
    help_text = f"""
{C.BOLD}{C.LCYAN}可用命令{C.RESET}
  {C.GREEN}/help{C.RESET}     - 显示此帮助
  {C.GREEN}/clear{C.RESET}    - 清空对话记忆
  {C.GREEN}/model{C.RESET}    - 查看当前模型
  {C.GREEN}/status{C.RESET}   - 查看记忆状态
  {C.GREEN}/tools{C.RESET}    - 查看可用工具列表
  {C.GREEN}/docs{C.RESET}     - 查看已加载的文档
  {C.GREEN}/add-docs{C.RESET}  - 添加文档到知识库
  {C.GREEN}/exit{C.RESET}     - 退出程序
"""
    print(help_text)


def print_tools_info(tool_schemas: list):
    """打印工具列表"""
    print(f"\n{C.BOLD}{C.LCYAN}可用工具列表{C.RESET}")
    for tool in tool_schemas:
        func = tool["function"]
        name = func["name"]
        desc = func["description"]
        print(f"  {C.CYAN}- {name}{C.RESET}")
        print(f"    {C.GRAY}{desc}{C.RESET}")
    print()


def print_memory_info(rounds: int, msg_count: int, est_tokens: int, max_rounds: int):
    """打印记忆状态"""
    print(f"\n{C.BOLD}{C.LCYAN}记忆状态{C.RESET}")
    print(f"  对话轮数: {C.YELLOW}{rounds}{C.RESET} / {max_rounds} 轮")
    print(f"  消息条数: {C.YELLOW}{msg_count}{C.RESET} 条 (含 system)")
    print(f"  估算 token: ~{C.YELLOW}{est_tokens}{C.RESET} tokens")
    print()


def print_user_prompt() -> str:
    """打印用户提示，返回输入"""
    return input(f"{C.BOLD}{C.LGREEN}你{C.RESET}: ").strip()


def print_agent_prefix():
    """打印 Agent 前缀"""
    print(f"{C.BOLD}{C.LBLUE}Ali-code{C.RESET}: ", end="", flush=True)


def print_error(text: str):
    """打印错误信息"""
    print(f"{C.RED}错误{C.RESET}: {text}")


def print_warning(text: str):
    """打印警告信息"""
    print(f"{C.YELLOW}警告{C.RESET}: {text}")


def print_info(text: str):
    """打印信息"""
    print(f"{C.CYAN}信息{C.RESET}: {text}")


def print_success(text: str):
    """打印成功信息"""
    print(f"{C.GREEN}{text}{C.RESET}")


def print_tool_call(tool_name: str, args_str: str):
    """打印工具调用"""
    print(f"  {C.LYELLOW}[工具调用]{C.RESET} {C.CYAN}{tool_name}{C.RESET}({args_str})", end=" ", flush=True)


def print_tool_done():
    """打印工具完成"""
    print(f"{C.GREEN}完成{C.RESET}")


def stream_char(char: str):
    """流式输出单个字符"""
    print(f"{C.LBLUE}{char}{C.RESET}", end="", flush=True)


def print_goodbye():
    """打印再见"""
    print(f"\n{C.LCYAN}再见！有问题随时来找我。{C.RESET}\n")
