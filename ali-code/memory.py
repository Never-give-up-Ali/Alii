# ==========================================
#  Step 3: 对话记忆管理模块
#  核心思路：滑动窗口 + 系统提示词常驻
# ==========================================


class Memory:
    """
    对话记忆管理器
    - 系统提示词永远保留在最前面
    - 用户和助手的对话按滑动窗口保留
    - 超过最大轮次时，丢掉最早的对话
    """

    def __init__(self, system_prompt: str, max_rounds: int = 10):
        """
        Args:
            system_prompt: 系统提示词
            max_rounds: 最多保留多少轮对话（一轮 = 用户+助手各一条）
        """
        self.system_prompt = system_prompt
        self.max_rounds = max_rounds
        # _history 存的是 user + assistant 的对话对，不含 system
        self._history: list[dict] = []

    def add_user(self, content: str):
        """添加用户消息"""
        self._history.append({"role": "user", "content": content})

    def add_assistant(self, content: str):
        """添加助手消息"""
        self._history.append({"role": "assistant", "content": content})

    def add_tool_result(self, tool_call_id: str, content: str):
        """添加工具调用结果（Step 5 会用到）"""
        self._history.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content
        })

    def add_assistant_with_tool_calls(self, content: str, tool_calls: list):
        """添加带工具调用的助手消息（Step 5 会用到）"""
        msg = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        self._history.append(msg)

    def get_messages(self) -> list[dict]:
        """
        获取发送给 LLM 的完整消息列表
        格式：[system_msg, ...最近的历史消息]
        """
        # 计算需要保留的历史消息条数（每轮 2 条：user + assistant）
        max_messages = self.max_rounds * 2
        if len(self._history) > max_messages:
            # 丢掉最早的，保留最新的
            # 注意：要从 user 消息开始截断，不能从中间截断
            excess = len(self._history) - max_messages
            # 确保截断后第一条是 user 消息
            start_idx = excess
            if self._history[start_idx]["role"] != "user":
                start_idx += 1
            self._history = self._history[start_idx:]

        # 组合 system + history
        return [{"role": "system", "content": self.system_prompt}] + self._history

    def clear(self):
        """清空对话历史（保留系统提示词）"""
        self._history = []

    def __len__(self) -> int:
        """返回历史消息条数（不含 system）"""
        return len(self._history)

    @property
    def rounds(self) -> int:
        """返回对话轮数"""
        return len(self._history) // 2

    def estimate_tokens(self) -> int:
        """
        粗略估算 token 数（按 1 个中文 ≈ 1.5 token，英文 1 词 ≈ 1 token）
        这只是估算，不准确但够用
        """
        total = 0
        for msg in self._history:
            content = msg.get("content", "")
            if content:
                # 简单估算：中文字符数 * 1.5 + 英文单词数
                import re
                chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', content))
                english_words = len(re.findall(r'[a-zA-Z]+', content))
                total += int(chinese_chars * 1.5 + english_words)
        # 加上 system prompt
        total += int(len(self.system_prompt) * 1.5)
        return total
