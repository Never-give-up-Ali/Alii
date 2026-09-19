# ==========================================
#  Ali-code -  AI Agent
#  功能：彩色界面 + 流式输出 + 工具调用 + 记忆管理 + RAG 知识库
# ==========================================
import os
import sys
import json
from dotenv import load_dotenv
from openai import OpenAI
from memory import Memory
from tools import ALL_TOOL_SCHEMAS, execute_tool
from rag import init_retriever, get_retriever
from ui import (
    C, print_banner, print_help_menu, print_tools_info,
    print_memory_info, print_user_prompt, print_agent_prefix,
    print_error, print_info, print_success, print_warning,
    print_goodbye
)



def load_config():
    load_dotenv()

    api_key = os.getenv("DOUBAO_API_KEY", "")
    base_url = os.getenv("DOUBAO_BASE_URL", "https://ark.cn-beijing.volces.com/api/plan/v3")
    model = os.getenv("DOUBAO_MODEL", "doubao-seed-2.1-turbo")
    max_rounds = int(os.getenv("MAX_ROUNDS", "10"))
    max_tool_calls = int(os.getenv("MAX_TOOL_CALLS", "5"))

    if not api_key or api_key == "你的API密钥填这里":
        print_error("请先在 .env 文件中配置 DOUBAO_API_KEY")
        print("   复制 .env.example 为 .env，然后填入你的 API Key")
        sys.exit(1)

    return {
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
        "max_rounds": max_rounds,
        "max_tool_calls": max_tool_calls,
        "rag_mode": os.getenv("RAG_MODE", "tfidf"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "doubao-embedding-250128"),
        "embedding_dims": int(os.getenv("EMBEDDING_DIMENSIONS", "1536")),
        "pg_host": os.getenv("PG_HOST", "localhost"),
        "pg_port": int(os.getenv("PG_PORT", "5432")),
        "pg_user": os.getenv("PG_USER", "postgres"),
        "pg_password": os.getenv("PG_PASSWORD", ""),
        "pg_database": os.getenv("PG_DATABASE", "ali_code"),
    }

#  系统提示词
SYSTEM_PROMPT = """你是一个运行在 Windows 命令行终端的 AI 助手，名叫 Ali-code。
你专业、高效，能够调用工具来帮助用户解决问题。

【核心规则】
1. 优先使用工具：涉及计算、时间、系统信息、文档查询时，必须调用对应工具获取准确结果
2. 调用工具无需跟用户确认，直接调用即可
3. 拿到工具结果后，用自然语言整理成清晰的回答
4. 可以连续调用多个工具，也可以根据前一个工具的结果决定下一步
5. 回答简洁明了，重点突出

【知识库检索规则】
- 当用户询问关于文档、项目资料、或者需要根据已有文件回答的问题时，使用 search_knowledge_base 工具
- 根据检索到的内容回答用户问题，不要编造信息
- 如果检索结果不足以回答问题，如实告诉用户

【可用工具】
- calculator: 高精度数学计算器
- get_current_time: 获取当前日期和时间
- run_system_command: 执行系统命令（安全白名单限制）
- search_knowledge_base: 从知识库中检索文档内容
"""



#  流式 + 工具调用 混合引擎
def agent_stream_with_tools(client, model, memory, max_tool_calls):
    """
    流式输出 + 工具调用 的混合实现
    """
    for turn in range(max_tool_calls):
        messages = memory.get_messages()

        # 发起流式请求
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            tools=ALL_TOOL_SCHEMAS,
            tool_choice="auto",
            stream=True,
        )

        # 收集流式数据
        full_content = ""
        tool_calls = []

        for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # 收集文本内容并流式打印
            if delta.content:
                full_content += delta.content
                sys.stdout.write(f"{C.LBLUE}{delta.content}{C.RESET}")
                sys.stdout.flush()

            # 收集工具调用（流式拼接）
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    while len(tool_calls) <= idx:
                        tool_calls.append({
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""}
                        })

                    if tc.id:
                        tool_calls[idx]["id"] = tc.id
                    if tc.type:
                        tool_calls[idx]["type"] = tc.type
                    if tc.function and tc.function.name:
                        tool_calls[idx]["function"]["name"] += tc.function.name
                    if tc.function and tc.function.arguments:
                        tool_calls[idx]["function"]["arguments"] += tc.function.arguments

        print()  # 流式输出结束，换行

        #没有工具调用，回答完毕
        if not tool_calls:
            memory.add_assistant(full_content)
            print()
            return

        # 有工具调用
        tool_calls_for_memory = []
        for tc in tool_calls:
            tool_calls_for_memory.append({
                "id": tc["id"],
                "type": tc["type"],
                "function": {
                    "name": tc["function"]["name"],
                    "arguments": tc["function"]["arguments"]
                }
            })

        memory.add_assistant_with_tool_calls(
            content=full_content,
            tool_calls=tool_calls_for_memory
        )


        for tc in tool_calls:
            tool_name = tc["function"]["name"]
            try:
                tool_args = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                tool_args = {}


            args_str = ", ".join(f"{k}={v}" for k, v in tool_args.items())
            print(f"  {C.LYELLOW}[工具调用]{C.RESET} {C.CYAN}{tool_name}{C.RESET}({args_str})", end=" ", flush=True)


            result = execute_tool(tool_name, tool_args)

            print(f"{C.GREEN}完成{C.RESET}")


            memory.add_tool_result(
                tool_call_id=tc["id"],
                content=result
            )


        print_agent_prefix()

    print(f"{C.YELLOW}警告：工具调用次数达到上限，停止继续调用{C.RESET}")
    print()



def cmd_add_docs():
    """处理 /add-docs 命令：添加文档到知识库"""
    retriever = get_retriever()
    print(f"\n{C.CYAN}添加文档到知识库{C.RESET}")
    print(f"  支持格式: PDF, DOCX, TXT, MD, CSV, JSON, PY, JS")
    print(f"  可以输入文件路径或文件夹路径，多个路径用空格分隔")
    print(f"  输入空行取消\n")

    user_input = input(f"{C.GRAY}路径:{C.RESET} ").strip()

    if not user_input:
        print_info("已取消\n")
        return

    paths = user_input.split()
    total_docs = 0
    total_chunks = 0

    for path in paths:
        path = path.strip().strip('"').strip("'")
        if not path:
            continue

        try:
            if os.path.isfile(path):
                # 单个文件
                n = retriever.add_document(path)
                if n > 0:
                    total_docs += 1
                    total_chunks += n
                    print(f"  {C.GREEN}+{C.RESET} {os.path.basename(path)} ({n} 块)")
                else:
                    print(f"  {C.YELLOW}跳过{C.RESET} {os.path.basename(path)} (已加载或为空)")
            elif os.path.isdir(path):
                # 目录
                docs, chunks = retriever.add_directory(path)
                total_docs += docs
                total_chunks += chunks
                print(f"  {C.GREEN}+{C.RESET} 目录 '{path}' ({docs} 个文档, {chunks} 块)")
            else:
                print(f"  {C.RED}错误{C.RESET}: 路径不存在 - {path}")
        except Exception as e:
            print(f"  {C.RED}错误{C.RESET}: {path} - {e}")

    print()
    if total_docs > 0:
        print_success(f"添加完成：{total_docs} 个文档，共 {total_chunks} 个文本块\n")
    else:
        print_warning("没有添加任何文档\n")


def cmd_list_docs():
    """处理 /docs 命令：列出已加载的文档"""
    retriever = get_retriever()
    docs = retriever.list_documents()
    print(f"\n{C.BOLD}{C.LCYAN}知识库状态{C.RESET}")
    print(f"  文档数量: {C.YELLOW}{len(docs)}{C.RESET}")
    print(f"  文本块数: {C.YELLOW}{retriever.chunk_count}{C.RESET}")
    if docs:
        print(f"  文档列表:")
        for doc in docs:
            print(f"    - {doc}")
    print()


def main():
    # 加载配置
    config = load_config()

    # 初始化 LLM 客户端
    client = OpenAI(api_key=config["api_key"], base_url=config["base_url"])

    # 初始化记忆
    memory = Memory(
        system_prompt=SYSTEM_PROMPT,
        max_rounds=config["max_rounds"]
    )


    rag_mode = config["rag_mode"].lower()
    try:
        if rag_mode == "pgvector":
            init_retriever(
                mode="pgvector",
                db_config={
                    "host": config["pg_host"],
                    "port": config["pg_port"],
                    "user": config["pg_user"],
                    "password": config["pg_password"],
                    "dbname": config["pg_database"],
                },
                embedding_config={
                    "client": client,
                    "model": config["embedding_model"],
                    "dimensions": config["embedding_dims"],
                }
            )
            print_info(f"RAG 模式: pgvector (PostgreSQL 向量检索)\n")
        else:
            init_retriever(mode="tfidf")
            print_info(f"RAG 模式: TF-IDF (关键词检索)\n")
    except Exception as e:
        print_warning(f"初始化 {rag_mode} 检索器失败，回退到 TF-IDF 模式: {e}")
        init_retriever(mode="tfidf")

    # 打印启动界面
    os.system("cls")
    print_banner()

    # 主循环
    while True:
        try:
            user_input = print_user_prompt()

            # 空输入跳过
            if not user_input:
                continue

            # 命令处理
            if user_input.startswith("/"):
                cmd = user_input.lower()

                if cmd in ("/exit", "/quit", "/q"):
                    print_goodbye()
                    break

                elif cmd == "/help":
                    print_help_menu()

                elif cmd == "/clear":
                    os.system("cls")
                    memory.clear()
                    print_banner()

                elif cmd == "/model":
                    print(f"\n{C.CYAN}当前模型{C.RESET}: {config['model']}\n")

                elif cmd == "/status":
                    msgs = memory.get_messages()
                    print_memory_info(
                        rounds=memory.rounds,
                        msg_count=len(msgs),
                        est_tokens=memory.estimate_tokens(),
                        max_rounds=memory.max_rounds
                    )

                elif cmd == "/tools":
                    print_tools_info(ALL_TOOL_SCHEMAS)

                elif cmd == "/docs":
                    cmd_list_docs()

                elif cmd == "/add-docs":
                    cmd_add_docs()

                else:
                    print_error(f"未知命令: {user_input}，输入 /help 查看帮助")
                    print()

                continue

            # 正常对话：加入记忆，进入 Agent 循环
            memory.add_user(user_input)
            print_agent_prefix()

            try:
                agent_stream_with_tools(
                    client=client,
                    model=config["model"],
                    memory=memory,
                    max_tool_calls=config["max_tool_calls"]
                )
            except Exception as e:
                print_error(str(e))
                print()

        except KeyboardInterrupt:
            print_goodbye()
            break
        except EOFError:
            print_goodbye()
            break


if __name__ == "__main__":
    main()
