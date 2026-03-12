"""
Leader Agent Platform - LLM Client

本模块封装 LLM API 调用逻辑，提供：
- 多 profile 支持（default, fast, pro 等）
- 结构化输出解析
- 错误重试与异常处理
- 调用日志记录
"""

import json
import logging
import re
from typing import Any, Dict, Optional, Type, TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from ..config import settings
from ..models import (
    LLM_CALL_TIMEOUT_SECONDS,
    LLM_MAX_RETRIES,
    LLM6_CALL_TIMEOUT_SECONDS,
)
from ..models.exceptions import LLMCallError, LLMParseError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    """
    LLM API 客户端。

    支持多个 LLM profile，每个 profile 可以配置不同的模型、温度等参数。
    典型的 profile 包括：
    - llm.default: 默认配置，用于一般性任务
    - llm.fast: 快速模型，用于简单分类任务（如 LLM-1）
    - llm.pro: 高级模型，用于复杂推理任务
    """

    def __init__(self):
        """初始化 LLM 客户端。"""
        self._clients: Dict[str, OpenAI] = {}
        self._profiles: Dict[str, Dict[str, Any]] = {}
        self._init_profiles()

    def _init_profiles(self) -> None:
        """从配置初始化所有 LLM profiles。"""
        llm_config = settings.get("llm", {})

        for profile_name, profile_data in llm_config.items():
            if not isinstance(profile_data, dict):
                continue

            full_name = f"llm.{profile_name}"
            self._profiles[full_name] = profile_data

            # llm.pro 用于 LLM-6 (Aggregation)，需要更长超时时间
            timeout = (
                LLM6_CALL_TIMEOUT_SECONDS
                if profile_name == "pro"
                else LLM_CALL_TIMEOUT_SECONDS
            )

            # 创建 OpenAI 客户端
            self._clients[full_name] = OpenAI(
                api_key=profile_data.get("api_key", ""),
                base_url=profile_data.get("base_url"),
                timeout=timeout,
            )

            logger.info(
                f"Initialized LLM profile: {full_name} -> {profile_data.get('model')} (timeout={timeout}s)"
            )

    def get_profile(self, profile_name: str) -> Dict[str, Any]:
        """
        获取指定 profile 的配置。

        Args:
            profile_name: profile 名称（如 "llm.default" 或 "llm.fast"）

        Returns:
            profile 配置字典

        Raises:
            LLMCallError: profile 不存在
        """
        # 处理简写形式
        if not profile_name.startswith("llm."):
            profile_name = f"llm.{profile_name}"

        if profile_name not in self._profiles:
            raise LLMCallError(f"Unknown LLM profile: {profile_name}")

        return self._profiles[profile_name]

    def call(
        self,
        profile_name: str,
        system_prompt: str,
        user_message: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        调用 LLM 获取文本响应。

        Args:
            profile_name: LLM profile 名称
            system_prompt: 系统提示词
            user_message: 用户消息
            temperature: 覆盖默认温度（可选）
            max_tokens: 最大 token 数（可选）

        Returns:
            LLM 的响应文本

        Raises:
            LLMCallError: API 调用失败
        """
        # 规范化 profile 名称
        if not profile_name.startswith("llm."):
            profile_name = f"llm.{profile_name}"

        if profile_name not in self._clients:
            raise LLMCallError(f"Unknown LLM profile: {profile_name}")

        client = self._clients[profile_name]
        profile = self._profiles[profile_name]

        # 准备参数
        model = profile.get("model", "gpt-4")
        temp = (
            temperature if temperature is not None else profile.get("temperature", 0.7)
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        # 重试逻辑
        last_error = None
        for attempt in range(LLM_MAX_RETRIES):
            try:
                logger.debug(
                    f"LLM call [{profile_name}] attempt {attempt + 1}: "
                    f"model={model}, temp={temp}"
                )

                kwargs = {
                    "model": model,
                    "messages": messages,
                    "temperature": temp,
                }
                if max_tokens:
                    kwargs["max_tokens"] = max_tokens

                response = client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content

                logger.debug(f"LLM response received: {len(content)} chars")
                return content

            except Exception as e:
                last_error = e
                logger.warning(
                    f"LLM call failed (attempt {attempt + 1}/{LLM_MAX_RETRIES}): {e}"
                )

        raise LLMCallError(
            f"LLM call failed after {LLM_MAX_RETRIES} retries: {last_error}"
        )

    def call_structured(
        self,
        profile_name: str,
        system_prompt: str,
        user_message: str,
        response_model: Type[T],
        temperature: Optional[float] = None,
    ) -> T:
        """
        调用 LLM 并解析为结构化输出。

        使用 JSON 模式请求 LLM 返回符合指定 Pydantic 模型的响应。

        Args:
            profile_name: LLM profile 名称
            system_prompt: 系统提示词（应包含 JSON 格式要求）
            user_message: 用户消息
            response_model: 期望的响应 Pydantic 模型类
            temperature: 覆盖默认温度（可选）

        Returns:
            解析后的 Pydantic 模型实例

        Raises:
            LLMCallError: API 调用失败
            LLMParseError: 响应解析失败
        """
        # 调用 LLM
        raw_response = self.call(
            profile_name=profile_name,
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=temperature,
        )

        # 提取 JSON
        json_str = self._extract_json(raw_response)
        if not json_str:
            raise LLMParseError(
                f"No JSON found in LLM response: {raw_response[:200]}..."
            )

        # 解析并验证
        try:
            data = json.loads(json_str)
            return response_model.model_validate(data)
        except json.JSONDecodeError as e:
            # 添加详细日志以便排查
            logger.error(f"JSON decode error at position {e.pos}: {e.msg}")
            logger.error(f"Extracted JSON (first 500 chars): {json_str[:500]}...")
            raise LLMParseError(f"Invalid JSON in LLM response: {e}")
        except ValidationError as e:
            raise LLMParseError(f"Response validation failed: {e}")

    def _extract_json(self, text: str) -> Optional[str]:
        """
        从 LLM 响应中提取 JSON。

        支持以下格式：
        1. 纯 JSON 响应
        2. ```json ... ``` 代码块
        3. ``` ... ``` 代码块
        4. 文本中嵌入的 JSON 对象

        Args:
            text: LLM 响应文本

        Returns:
            提取的 JSON 字符串或 None
        """
        text = text.strip()

        # 1. 尝试直接解析
        if text.startswith("{") and text.endswith("}"):
            return text

        # 2. 尝试 ```json ... ``` 格式
        json_block_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if json_block_match:
            return json_block_match.group(1).strip()

        # 3. 尝试 ``` ... ``` 格式
        code_block_match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
        if code_block_match:
            content = code_block_match.group(1).strip()
            if content.startswith("{"):
                return content

        # 4. 尝试提取文本中的 JSON 对象
        json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
        if json_match:
            return json_match.group(0)

        return None


# 模块级单例
_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """获取 LLM 客户端单例。"""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
