import base64
import requests
import json
from typing import Dict, Any, List, TypedDict, Annotated, Optional, Type, ClassVar
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import operator


class SDText2ImgInput(BaseModel):
    prompt: str = Field(description="正向绘图提示词，描述画面内容")
    negative_prompt: Optional[str] = Field(default="", description="反向提示词")
    steps: Optional[int] = Field(default=20, description="采样步数")
    guidance_scale: Optional[float] = Field(default=7.5, description="引导系数")
    width: Optional[int] = Field(default=512, description="图片宽度")
    height: Optional[int] = Field(default=512, description="图片高度")


class SDText2ImgTool(BaseTool):
    name: str = "sd_text2image"
    description: str = "调用本地私有化Stable Diffusion服务生成图片"
    args_schema: Optional[Type[BaseModel]] = SDText2ImgInput
    api_url: ClassVar[str] = ""

    def _run(self, prompt: str,
             negative_prompt: str = "",
             steps: int = 20,
             guidance_scale: float = 7.5,
             width: int = 512,
             height: int = 512) -> Dict[str, Any]:
        """执行绘图，返回包含图片路径和元数据的结果"""
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height
        }
        resp = requests.post(self.api_url, json=payload, timeout=120)
        resp.raise_for_status()
        res_json = resp.json()
        img_b64 = res_json["images"][0]

        img_bytes = base64.b64decode(img_b64)
        save_path = f"agent_generated_{hash(prompt) % 10000}.png"
        with open(save_path, "wb") as f:
            f.write(img_bytes)

        return {
            "success": True,
            "image_path": save_path,
            "base64_length": len(img_b64),
            "params": payload
        }


class AgentState(TypedDict):

    user_query: str
    session_id: Optional[str]

    # Router 输出
    intent: Optional[str]  # "draw" 或 "chat"

    # Planner 输出
    plan: Optional[str]
    prompt: Optional[str]
    negative_prompt: Optional[str]
    params: Optional[Dict[str, Any]]

    # Worker 输出
    image_result: Optional[Dict[str, Any]]
    image_path: Optional[str]

    # Reviewer 输出
    review_score: Optional[int]
    review_feedback: Optional[str]
    is_passed: Optional[bool]
    retry_count: int

    # Responder 输出
    final_answer: Optional[str]

    # 上下文记忆
    history: Annotated[List[Dict[str, Any]], operator.add]



class RouterNode:
    """判断用户意图是绘图还是聊天"""

    def __init__(self, llm):
        self.llm = llm

    def __call__(self, state: AgentState) -> AgentState:
        user_input = state['user_query']

        # 关键词匹配（快速低成本）
        draw_keywords = ["画",  "绘制", "画一个", "做一张", "帮我画"]
        if any(kw in user_input for kw in draw_keywords):
            state['intent'] = "draw"
        else:
            state['intent'] = "chat"

        print(f"  识别意图: {state['intent']}")
        return state


class ResponderNode:

    def __init__(self, llm):
        self.llm = llm

    def __call__(self, state: AgentState) -> AgentState:
        #print("💬 [Responder] 直接回复用户...")
        response = self.llm.invoke(f"请以朋友的身份回复用户：{state['user_query']}")
        state['final_answer'] = response.content
        print(f" 回复: {state['final_answer'][:100]}...")
        return state


class PlannerNode:
   

    def __init__(self, llm):
        self.llm = llm

    def __call__(self, state: AgentState) -> AgentState:
        print(" [Planner] 正在解析绘图需求...")

        prompt = f"""你是一个专业的绘画策划师，需要根据用户需求制定绘图方案。

用户需求：{state['user_query']}

请输出JSON格式的绘图计划（只输出JSON，不要其他内容）：
{{
    "plan": "你对该绘图任务的简要分析",
    "prompt": "详细的正向提示词，描述画面内容、风格、构图",
    "negative_prompt": "反向提示词，列出不想要的内容",
    "params": {{
        "steps": 20,
        "guidance_scale": 7.5,
        "width": 512,
        "height": 512
    }}
}}

如果用户需求模糊，请在plan中说明你的补充假设。
"""
        response = self.llm.invoke(prompt)

        try:
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            plan_data = json.loads(content)

            state['plan'] = plan_data.get('plan', '')
            state['prompt'] = plan_data.get('prompt', state['user_query'])
            state['negative_prompt'] = plan_data.get('negative_prompt', '')
            state['params'] = plan_data.get('params', {})

            state['history'].append({
                "node": "Planner",
                "plan": state['plan'],
                "prompt": state['prompt']
            })

        except Exception as e:
            print(f"Planner解析失败: {e}，使用默认值")
            state['prompt'] = state['user_query']
            state['negative_prompt'] = ''
            state['params'] = {"steps": 20, "guidance_scale": 7.5, "width": 512, "height": 512}
            state['history'].append({"node": "Planner", "error": str(e)})

        state['retry_count'] = 0
        return state


class WorkerNode:
    """Worker：执行绘图任务"""

    def __init__(self, sd_tool: SDText2ImgTool):
        self.sd_tool = sd_tool

    def __call__(self, state: AgentState) -> AgentState:
        print("[Worker] 正在生成图片...")

        params = {
            "prompt": state.get('prompt', state['user_query']),
            "negative_prompt": state.get('negative_prompt', ''),
            "steps": state.get('params', {}).get('steps', 20),
            "guidance_scale": state.get('params', {}).get('guidance_scale', 7.5),
            "width": state.get('params', {}).get('width', 512),
            "height": state.get('params', {}).get('height', 512)
        }

        try:
            result = self.sd_tool._run(**params)
            state['image_result'] = result
            state['image_path'] = result.get('image_path')
            state['history'].append({
                "node": "Worker",
                "params": params,
                "success": result.get('success', False)
            })
            print(f"图片生成成功: {state['image_path']}")
        except Exception as e:
            print(f"Worker执行失败: {e}")
            state['image_result'] = {"success": False, "error": str(e)}
            state['history'].append({"node": "Worker", "error": str(e)})

        return state


class ReviewerNode:
    """Reviewer：评审图片质量，决定是否回溯"""

    def __init__(self, llm, max_retries: int = 3):
        self.llm = llm
        self.max_retries = max_retries

    def __call__(self, state: AgentState) -> AgentState:
        print("[Reviewer] 正在评审图片质量...")

        if not state.get('image_result', {}).get('success', False):
            state['is_passed'] = False
            state['review_feedback'] = "绘图服务执行失败"
            state['review_score'] = 0
            state['retry_count'] = state.get('retry_count', 0) + 1
            state['history'].append({"node": "Reviewer", "passed": False, "reason": "服务失败"})
            return state

        review_prompt = f"""你是一个专业的绘画评审师，请评审以下绘图任务的质量。

用户需求：{state['user_query']}
绘图计划：{state.get('plan', '无')}
生成的提示词：{state.get('prompt', '')}
图片参数：{json.dumps(state.get('params', {}))}

当前重试次数：{state.get('retry_count', 0)} / {self.max_retries}

请输出JSON格式的评审结果（只输出JSON）：
{{
    "score": 0-10的整数评分，
    "feedback": "具体的改进意见，指出哪里不足",
    "passed": true或false（7分及以上为通过）
}}

如果未通过，请给出具体的修改建议（比如如何调整prompt或参数）。
"""
        response = self.llm.invoke(review_prompt)

        try:
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            review_data = json.loads(content)
            state['review_score'] = review_data.get('score', 5)
            state['review_feedback'] = review_data.get('feedback', '')
            state['is_passed'] = review_data.get('passed', False)

            state['history'].append({
                "node": "Reviewer",
                "score": state['review_score'],
                "passed": state['is_passed'],
                "feedback": state['review_feedback']
            })

            print(f"评分: {state['review_score']}/10, {'通过' if state['is_passed'] else '❌ 不通过'}")

        except Exception as e:
            print(f" Reviewer解析失败: {e}，默认通过")
            state['is_passed'] = True
            state['review_score'] = 7
            state['review_feedback'] = "评审异常，默认通过"

        if not state['is_passed']:
            state['retry_count'] = state.get('retry_count', 0) + 1

        return state



def route_after_router(state: AgentState) -> str:
    
    if state.get('intent') == "draw":
        return "draw_flow"
    else:
        return "chat_flow"


def should_continue(state: AgentState) -> str:
    """Reviewer 节点后的条件路由"""
    if state.get('is_passed', False):
        print("评审通过，任务完成！")
        return "end"

    if state.get('retry_count', 0) >= 3:
        print(" 达到最大重试次数，强制结束")
        return "end"

    print(" 评审不通过，回溯到Worker重试...")
    return "retry"


def apply_review_feedback(state: AgentState) -> AgentState:
    """应用Reviewer的反馈，优化下一次绘图"""

    if not state.get('is_passed', False):
        feedback = state.get('review_feedback', '')
        print(f"应用反馈优化: {feedback}")

        llm = ChatOpenAI(
            model="qwen-plus",
            openai_api_key="sk-d6129fd1dc8c4e1683b8316deec5a57b",
            openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0.2
        )

        improve_prompt = f"""根据以下评审反馈改进绘图方案。

原提示词：{state.get('prompt', '')}
评审反馈：{feedback}

请输出改进后的绘图方案（JSON格式）：
{{
    "prompt": "改进后的正向提示词",
    "negative_prompt": "改进后的反向提示词",
    "params": {{"steps": 20, "guidance_scale": 7.5, "width": 512, "height": 512}}
}}
"""
        try:
            response = llm.invoke(improve_prompt)
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            improved = json.loads(content)
            state['prompt'] = improved.get('prompt', state['prompt'])
            state['negative_prompt'] = improved.get('negative_prompt', state.get('negative_prompt', ''))
            state['params'] = improved.get('params', state.get('params', {}))

            print(f" 已优化提示词: {state['prompt'][:50]}...")

        except Exception as e:
            print(f"反馈优化失败: {e}，保持原方案不变")

    return state



def build_agent_graph(llm, sd_tool) -> StateGraph:

    # 初始化节点
    router = RouterNode(llm)
    responder = ResponderNode(llm)
    planner = PlannerNode(llm)
    worker = WorkerNode(sd_tool)
    reviewer = ReviewerNode(llm, max_retries=3)

    graph = StateGraph(AgentState)

    # 添加所有节点
    graph.add_node("router", router)
    graph.add_node("responder", responder)
    graph.add_node("planner", planner)
    graph.add_node("worker", worker)
    graph.add_node("reviewer", reviewer)
    graph.add_node("apply_feedback", apply_review_feedback)

    graph.set_entry_point("router")

    # Router 后的条件分支
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "draw_flow": "planner",
            "chat_flow": "responder"
        }
    )

    # 绘图流程
    graph.add_edge("planner", "worker")
    graph.add_edge("worker", "reviewer")

    graph.add_conditional_edges(
        "reviewer",
        should_continue,
        {
            "end": END,
            "retry": "apply_feedback"
        }
    )

    graph.add_edge("apply_feedback", "worker")


    graph.add_edge("responder", END)

    return graph



def run_continuous_chat(compiled_graph, initial_state, config):
    """连续对话模式：保持会话状态，支持多轮交互"""

    print("=" * 60)
    print("连续对话模式已启动（输入 'exit' 或 'quit' 退出）")
    print("=" * 60)

    current_state = initial_state.copy()
    conversation_history = []
    round_num = 0

    while True:
        user_input = input("\n你: ").strip()

        if user_input.lower() in ['exit', 'quit', '退出']:
            print("再见！")
            break

        if not user_input:
            continue

        round_num += 1
        current_state['user_query'] = user_input

        print(f"\n处理第 {round_num} 轮对话...")

        try:
            final_state = compiled_graph.invoke(current_state, config)
        except Exception as e:
            print(f"执行出错: {e}")
            continue



        # 判断本轮走的哪个分支
        if final_state.get('intent') == "draw":
            print(f"用户需求: {final_state['user_query']}")
            print(f"最终提示词: {final_state.get('prompt', '')}")
            print(f"图片路径: {final_state.get('image_path', '未生成')}")
            print(f"评审分数: {final_state.get('review_score', 'N/A')}/10")
            print(f"是否通过: {'通过' if final_state.get('is_passed') else '未通过'}")
            print(f"重试次数: {final_state.get('retry_count', 0)}")
        else:
            # 聊天分支，展示 final_answer
            print(f"用户需求: {final_state['user_query']}")
            print(f"回复: {final_state.get('final_answer', '（无回复）')}")

        conversation_history.append({
            "round": round_num,
            "user": user_input,
            "intent": final_state.get('intent'),
            "image": final_state.get('image_path'),
            "score": final_state.get('review_score')
        })

        current_state = final_state.copy()




if __name__ == "__main__":

    llm = ChatOpenAI(
        model="q",
        openai_api_key="",
        openai_api_base="",
        temperature=0.2
    )


    sd_tool = SDText2ImgTool()
    graph = build_agent_graph(llm, sd_tool)

    memory = MemorySaver()
    compiled_graph = graph.compile(checkpointer=memory)

    initial_state = {
        "user_query": "",
        "session_id": "continuous_session_001",
        "history": [],
        "retry_count": 0,
        "intent": None,
        "final_answer": None
    }

    config = {"configurable": {"thread_id": "continuous_session_001"}}
    run_continuous_chat(compiled_graph, initial_state, config)
