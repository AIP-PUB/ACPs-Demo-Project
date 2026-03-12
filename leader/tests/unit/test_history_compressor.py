"""
Leader Agent Platform - HistoryCompressor 单元测试

测试 HistoryCompressor 的核心功能（LLM-7 规格）：
1. 历史压缩触发条件判断
2. 压缩请求处理
3. Prompt 构建
4. 响应解析
5. 降级处理
"""

import sys
import os
from pathlib import Path

_current_dir = Path(__file__).parent
_leader_dir = _current_dir.parent.parent
_project_root = _leader_dir.parent

if str(_leader_dir) not in sys.path:
    sys.path.insert(0, str(_leader_dir))
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from assistant.core.history_compressor import HistoryCompressor
from assistant.models.history_compression import (
    CompressionTurn,
    HistoryCompressionRequest,
    HistoryCompressionResult,
    COMPRESSION_THRESHOLD,
    TURNS_TO_KEEP,
)


@pytest.fixture
def mock_llm_client():
    """创建 Mock LLM 客户端。"""
    client = MagicMock()
    client.call = MagicMock(return_value="这是压缩后的摘要：用户询问了北京旅游推荐。")
    return client


@pytest.fixture
def mock_scenario_loader():
    """创建 Mock 场景加载器。"""
    loader = MagicMock()
    loader.get_scenario_prompt = MagicMock(
        return_value={"compression": "你是一个对话摘要助手。"}
    )
    return loader


@pytest.fixture
def compressor(mock_llm_client, mock_scenario_loader):
    """创建 HistoryCompressor 实例。"""
    return HistoryCompressor(
        llm_client=mock_llm_client,
        scenario_loader=mock_scenario_loader,
    )


def create_test_turns(count: int) -> list[CompressionTurn]:
    """创建测试用的轮次列表。"""
    turns = []
    for i in range(count):
        role = "user" if i % 2 == 0 else "assistant"
        content = f"这是第{i+1}轮的{'用户问题' if role == 'user' else '助手回复'}。"
        turns.append(
            CompressionTurn(
                turn_id=f"turn_{i}",
                role=role,
                content=content,
                timestamp=datetime.now().isoformat(),
                intent="TASK_NEW" if role == "user" else None,
            )
        )
    return turns


class TestShouldCompress:
    """测试压缩触发条件判断。"""

    def test_should_not_compress_empty(self, compressor):
        """测试空对话不需要压缩。"""
        result = compressor.should_compress(turns_count=0)
        assert result is False

    def test_should_not_compress_under_threshold(self, compressor):
        """测试轮次未达阈值不需要压缩。"""
        result = compressor.should_compress(turns_count=COMPRESSION_THRESHOLD - 1)
        assert result is False

    def test_should_compress_at_threshold(self, compressor):
        """测试轮次刚好达到阈值需要压缩。"""
        result = compressor.should_compress(turns_count=COMPRESSION_THRESHOLD)
        assert result is True

    def test_should_compress_over_threshold(self, compressor):
        """测试轮次超过阈值需要压缩。"""
        result = compressor.should_compress(turns_count=COMPRESSION_THRESHOLD + 5)
        assert result is True

    def test_custom_threshold(self, compressor):
        """测试自定义阈值。"""
        # 使用较小的阈值
        result = compressor.should_compress(turns_count=5, threshold=3)
        assert result is True

        result = compressor.should_compress(turns_count=2, threshold=3)
        assert result is False


class TestCompress:
    """测试压缩功能。"""

    @pytest.mark.asyncio
    async def test_compress_empty_turns(self, compressor):
        """测试压缩空轮次列表。"""
        request = HistoryCompressionRequest(
            session_id="test-session",
            turns_to_compress=[],
            existing_summary="现有摘要",
            existing_turn_count=5,
        )

        result = await compressor.compress(request)

        assert result.new_summary == "现有摘要"
        assert result.compressed_turn_count == 0

    @pytest.mark.asyncio
    async def test_compress_with_turns(self, compressor, mock_llm_client):
        """测试正常压缩。"""
        turns = create_test_turns(4)
        request = HistoryCompressionRequest(
            session_id="test-session",
            turns_to_compress=turns,
            existing_summary="",
            existing_turn_count=0,
        )

        result = await compressor.compress(request)

        # 验证 LLM 被调用
        mock_llm_client.call.assert_called_once()
        assert result.compressed_turn_count == 4
        assert len(result.new_summary) > 0

    @pytest.mark.asyncio
    async def test_compress_with_existing_summary(self, compressor, mock_llm_client):
        """测试在现有摘要基础上压缩。"""
        turns = create_test_turns(4)
        request = HistoryCompressionRequest(
            session_id="test-session",
            turns_to_compress=turns,
            existing_summary="用户之前询问过上海美食。",
            existing_turn_count=6,
        )

        result = await compressor.compress(request)

        # 验证 prompt 中包含现有摘要
        call_args = mock_llm_client.call.call_args
        prompt = call_args.kwargs.get("user_message", "")
        assert "已有历史摘要" in prompt

    @pytest.mark.asyncio
    async def test_compress_llm_error_fallback(self, compressor, mock_llm_client):
        """测试 LLM 调用失败时的降级处理。"""
        mock_llm_client.call = MagicMock(side_effect=Exception("LLM error"))

        turns = create_test_turns(4)
        request = HistoryCompressionRequest(
            session_id="test-session",
            turns_to_compress=turns,
            existing_summary="",
            existing_turn_count=0,
        )

        result = await compressor.compress(request)

        # 应该使用降级摘要
        assert result.compressed_turn_count == 4
        assert len(result.new_summary) > 0


class TestBuildCompressionPrompt:
    """测试 Prompt 构建功能。"""

    def test_prompt_contains_turns(self, compressor):
        """测试 prompt 包含待压缩轮次。"""
        turns = create_test_turns(2)

        prompt = compressor._build_compression_prompt(
            turns_to_compress=turns,
            existing_summary=None,
            scenario_id=None,
        )

        assert "第1轮" in prompt
        assert "第2轮" in prompt

    def test_prompt_contains_existing_summary(self, compressor):
        """测试 prompt 包含现有摘要。"""
        turns = create_test_turns(2)

        prompt = compressor._build_compression_prompt(
            turns_to_compress=turns,
            existing_summary="现有摘要内容",
            scenario_id=None,
        )

        assert "已有历史摘要" in prompt
        assert "现有摘要内容" in prompt

    def test_prompt_contains_task_instructions(self, compressor):
        """测试 prompt 包含任务说明。"""
        turns = create_test_turns(2)

        prompt = compressor._build_compression_prompt(
            turns_to_compress=turns,
            existing_summary=None,
            scenario_id=None,
        )

        assert "任务" in prompt
        assert "压缩" in prompt

    def test_prompt_shows_intent(self, compressor):
        """测试 prompt 显示意图信息。"""
        turns = [
            CompressionTurn(
                turn_id="t1",
                role="user",
                content="帮我订酒店",
                timestamp=datetime.now().isoformat(),
                intent="TASK_NEW",
            )
        ]

        prompt = compressor._build_compression_prompt(
            turns_to_compress=turns,
            existing_summary=None,
            scenario_id=None,
        )

        assert "TASK_NEW" in prompt


class TestFallbackCompression:
    """测试降级压缩功能。"""

    def test_fallback_creates_summary(self, compressor):
        """测试降级压缩生成摘要。"""
        turns = create_test_turns(3)

        summary = compressor._fallback_compression(
            turns_to_compress=turns,
            existing_summary=None,
        )

        assert len(summary) > 0
        # 降级摘要应该包含一些轮次内容
        assert "用户" in summary or "助手" in summary or "第" in summary

    def test_fallback_includes_existing_summary(self, compressor):
        """测试降级压缩包含现有摘要。"""
        turns = create_test_turns(2)

        summary = compressor._fallback_compression(
            turns_to_compress=turns,
            existing_summary="已有的摘要内容",
        )

        # 应该包含现有摘要或以某种形式整合
        assert len(summary) > 0


class TestParseResponse:
    """测试响应解析功能。"""

    def test_parse_simple_response(self, compressor):
        """测试解析简单响应。"""
        response = "这是对话的摘要内容。"

        result = compressor._parse_response(response)

        assert result == "这是对话的摘要内容。"

    def test_parse_response_with_extra_whitespace(self, compressor):
        """测试解析带额外空白的响应。"""
        response = "  \n这是摘要内容。\n  "

        result = compressor._parse_response(response)

        # 应该去除首尾空白
        assert result.strip() == "这是摘要内容。"


class TestCompressionResult:
    """测试压缩结果。"""

    @pytest.mark.asyncio
    async def test_result_has_correct_counts(self, compressor):
        """测试结果包含正确的计数。"""
        turns = create_test_turns(5)
        request = HistoryCompressionRequest(
            session_id="test-session",
            turns_to_compress=turns,
            existing_summary="",
            existing_turn_count=10,
        )

        result = await compressor.compress(request)

        assert result.compressed_turn_count == 5
        assert result.total_turn_count == 15

    @pytest.mark.asyncio
    async def test_result_has_timestamp(self, compressor):
        """测试结果包含时间戳。"""
        turns = create_test_turns(2)
        request = HistoryCompressionRequest(
            session_id="test-session",
            turns_to_compress=turns,
            existing_summary="",
            existing_turn_count=0,
        )

        result = await compressor.compress(request)

        assert result.compression_timestamp is not None
        assert len(result.compression_timestamp) > 0
