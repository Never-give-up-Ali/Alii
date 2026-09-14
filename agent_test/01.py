from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


llm = ChatOpenAI(
    model="qwen-plus",
    openai_api_key="",
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.2
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是智能客服，简洁回答用户问题"),
    ("human", "{user_input}")
])

chain = prompt | llm | StrOutputParser()

res = chain.invoke({"user_input": "你好"})
print(res)
