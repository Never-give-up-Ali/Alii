from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

llm = ChatOpenAI(
    model="qwen-plus",
    openai_api_key="",
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.2
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是智能客服，简洁回答用户问题"),
    MessagesPlaceholder(variable_name="chat_history"),  # 存放历史消息
    ("human", "{user_input}")
])

chain = prompt | llm | StrOutputParser()

chat_history = []

# 第一轮
user_input = "我叫小李"
resp = chain.invoke({
    "user_input": user_input,
    "chat_history": chat_history
})
print("AI:", resp)
chat_history.append(HumanMessage(content=user_input))
chat_history.append(AIMessage(content=resp))

# 第二轮，模型记得名字
user_input = "我叫什么？"
resp = chain.invoke({
    "user_input": user_input,
    "chat_history": chat_history
})
print("AI:", resp)
chat_history.append(HumanMessage(content=user_input))
chat_history.append(AIMessage(content=resp))
