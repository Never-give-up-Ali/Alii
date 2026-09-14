from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(
    model="qwen-plus",
    openai_api_key="",
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.2
)

@tool
def get_weather(city: str) -> str:
    """查询指定城市当前天气
    Args:
        city: 城市名称
    """
    return f"{city} 当前天气：晴，26℃"

tools = [get_weather]

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是智能客服，有需要就调用天气工具，回答简洁"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

chat_history = []

# 对话1
res = agent_executor.invoke({
    "input": "上海天气",
    "chat_history": chat_history
})
print(res["output"])
chat_history.append(HumanMessage(content="上海天气"))

# 对话2
res = agent_executor.invoke({
    "input": "那北京呢？",
    "chat_history": chat_history
})
print(res["output"])
chat_history.append(HumanMessage(content="那北京呢？"))
