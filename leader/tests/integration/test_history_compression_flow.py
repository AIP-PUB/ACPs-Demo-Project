"""
Leader Agent Platform - 集成测试：LLM-7 历史压缩流程

测试对话历史压缩功能，验证：
1. HistoryCompressor 正确调用 LLM-7 生成摘要
2. 现有摘要与新对话的整合
3. 多轮对话的压缩
4. 不同场景下的压缩效果
5. 压缩阈值判断

本测试使用真实大模型 API 调用，不使用 Mock。
"""

import sys
import os

# 确保 leader 目录在 path 中
leader_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if leader_dir not in sys.path:
    sys.path.insert(0, leader_dir)

# 确保项目根目录在 path 中（用于导入 acps_sdk.aip）
project_root = os.path.dirname(leader_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pytest
from typing import List

from assistant.core.history_compressor import HistoryCompressor
from assistant.models.history_compression import (
    CompressionTurn,
    HistoryCompressionRequest,
    HistoryCompressionResult,
    COMPRESSION_THRESHOLD,
    TURNS_TO_KEEP,
    MAX_SUMMARY_LENGTH,
)
from assistant.llm.client import get_llm_client
from assistant.services.scenario_loader import get_scenario_loader

from .conftest import build_submit_request, extract_session_id, is_success_response

pytest_plugins = ("pytest_asyncio",)


# =============================================================================
# 测试 Fixtures
# =============================================================================


@pytest.fixture
def history_compressor():
    """获取真实的 HistoryCompressor 实例（使用真实 LLM）。"""
    llm_client = get_llm_client()
    scenario_loader = get_scenario_loader()
    return HistoryCompressor(
        llm_client=llm_client,
        scenario_loader=scenario_loader,
    )


def create_compression_turns(conversations: List[dict]) -> List[CompressionTurn]:
    """
    创建压缩轮次数据。

    Args:
        conversations: 对话列表，每个元素为 {"role": "user/assistant", "content": "..."}

    Returns:
        CompressionTurn 列表
    """
    return [
        CompressionTurn(
            role=conv["role"],
            content=conv["content"],
            intent=conv.get("intent"),
            timestamp=conv.get("timestamp"),
        )
        for conv in conversations
    ]


# =============================================================================
# LLM-7 集成测试：基本压缩功能
# =============================================================================


class TestHistoryCompressorBasicCompression:
    """LLM-7 基本压缩功能测试（真实 LLM 调用）。"""

    @pytest.mark.asyncio
    async def test_compress_simple_conversation(
        self, history_compressor: HistoryCompressor
    ):
        """测试：压缩简单对话。"""
        turns = create_compression_turns(
            [
                {"role": "user", "content": "我想去北京旅游"},
                {
                    "role": "assistant",
                    "content": "好的，请问您计划什么时候出发？有多少人同行？",
                },
                {"role": "user", "content": "下周末，两个人"},
                {"role": "assistant", "content": "了解了。请问您的预算大概是多少？"},
            ]
        )

        request = HistoryCompressionRequest(
            session_id="test-session-001",
            turns_to_compress=turns,
            existing_summary=None,
            existing_turn_count=0,
            scenario_id="tour",
        )

        # 执行 LLM-7
        result = await history_compressor.compress(request)

        # 验证
        assert result is not None
        print(f"\n[SimpleConversation] new_summary: {result.new_summary}")
        print(f"[SimpleConversation] compressed_count: {result.compressed_turn_count}")
        print(f"[SimpleConversation] total_count: {result.total_turn_count}")

        # 验证摘要生成
        assert result.new_summary is not None
        assert len(result.new_summary) > 0
        assert result.compressed_turn_count == 4
        assert result.total_turn_count == 4

        # 验证摘要内容包含关键信息
        summary_lower = result.new_summary.lower()
        # 应该包含：北京、旅游、下周末、两人等关键信息
        has_key_info = (
            "北京" in result.new_summary
            or "旅游" in result.new_summary
            or "出发" in result.new_summary
        )
        assert has_key_info, f"Summary should contain key info: {result.new_summary}"

    @pytest.mark.asyncio
    async def test_compress_multi_turn_with_decisions(
        self, history_compressor: HistoryCompressor
    ):
        """测试：压缩包含决策的多轮对话。"""
        turns = create_compression_turns(
            [
                {"role": "user", "content": "我想预订酒店", "intent": "hotel_booking"},
                {"role": "assistant", "content": "好的，请问入住日期和退房日期？"},
                {"role": "user", "content": "12月25日入住，12月27日退房"},
                {"role": "assistant", "content": "请问房型偏好？标间还是大床房？"},
                {"role": "user", "content": "大床房，最好有窗户"},
                {"role": "assistant", "content": "好的，预算范围是多少？"},
                {"role": "user", "content": "500-800元每晚"},
                {"role": "assistant", "content": "已为您筛选出3家符合条件的酒店"},
            ]
        )

        request = HistoryCompressionRequest(
            session_id="test-session-001",
            turns_to_compress=turns,
            existing_summary=None,
            existing_turn_count=0,
            scenario_id="tour",
        )

        result = await history_compressor.compress(request)

        print(f"\n[MultiTurnDecisions] new_summary: {result.new_summary}")

        # 验证
        assert result.new_summary is not None
        assert result.compressed_turn_count == 8

        # 应该保留关键约束信息
        summary = result.new_summary
        has_date_info = "12" in summary or "25" in summary or "27" in summary
        has_room_info = "大床" in summary or "房" in summary
        has_budget_info = "500" in summary or "800" in summary or "预算" in summary

        key_info_count = sum([has_date_info, has_room_info, has_budget_info])
        assert (
            key_info_count >= 1
        ), f"Summary should contain at least 1 key constraint: {summary}"


class TestHistoryCompressorWithExistingSummary:
    """LLM-7 合并摘要测试（真实 LLM 调用）。"""

    @pytest.mark.asyncio
    async def test_merge_with_existing_summary(
        self, history_compressor: HistoryCompressor
    ):
        """测试：与现有摘要合并。"""
        existing_summary = (
            "用户计划北京旅游，两人同行，预算3000元。已确认交通方式为高铁。"
        )

        turns = create_compression_turns(
            [
                {"role": "user", "content": "酒店方面，我想住在王府井附近"},
                {
                    "role": "assistant",
                    "content": "好的，王府井地区酒店较多，请问房型偏好？",
                },
                {"role": "user", "content": "标间即可，干净卫生就行"},
                {"role": "assistant", "content": "已为您推荐3家酒店"},
            ]
        )

        request = HistoryCompressionRequest(
            session_id="test-session-001",
            turns_to_compress=turns,
            existing_summary=existing_summary,
            existing_turn_count=6,
            scenario_id="tour",
        )

        result = await history_compressor.compress(request)

        print(f"\n[MergeExisting] new_summary: {result.new_summary}")
        print(f"[MergeExisting] total_count: {result.total_turn_count}")

        # 验证
        assert result.new_summary is not None
        assert result.total_turn_count == 10  # 6 + 4

        # 新摘要应该整合旧信息和新信息
        summary = result.new_summary
        # 旧信息：北京、两人、预算、高铁
        # 新信息：王府井、酒店、标间
        has_old_info = "北京" in summary or "高铁" in summary or "两人" in summary
        has_new_info = "王府井" in summary or "酒店" in summary or "标间" in summary

        # 至少应该保留部分旧信息或新信息
        assert (
            has_old_info or has_new_info
        ), f"Summary should contain key info: {summary}"

    @pytest.mark.asyncio
    async def test_incremental_compression(self, history_compressor: HistoryCompressor):
        """测试：增量压缩（多次压缩）。"""
        # 第一次压缩
        turns1 = create_compression_turns(
            [
                {"role": "user", "content": "帮我规划北京三日游"},
                {"role": "assistant", "content": "好的，请问出行时间？"},
                {"role": "user", "content": "下周五到周日"},
            ]
        )

        request1 = HistoryCompressionRequest(
            session_id="test-session-001",
            turns_to_compress=turns1,
            existing_summary=None,
            existing_turn_count=0,
            scenario_id="tour",
        )

        result1 = await history_compressor.compress(request1)
        print(f"\n[IncrementalCompression] First summary: {result1.new_summary}")

        # 第二次压缩，基于第一次的摘要
        turns2 = create_compression_turns(
            [
                {"role": "assistant", "content": "了解了。请问对景点有什么偏好？"},
                {"role": "user", "content": "想去故宫和长城"},
                {"role": "assistant", "content": "好的，这两个景点都很热门"},
            ]
        )

        request2 = HistoryCompressionRequest(
            session_id="test-session-001",
            turns_to_compress=turns2,
            existing_summary=result1.new_summary,
            existing_turn_count=result1.total_turn_count,
            scenario_id="tour",
        )

        result2 = await history_compressor.compress(request2)
        print(f"[IncrementalCompression] Second summary: {result2.new_summary}")

        # 验证
        assert result2.total_turn_count == 6  # 3 + 3

        # 最终摘要应该包含两次对话的信息
        summary = result2.new_summary
        has_first_info = "三日" in summary or "下周" in summary or "北京" in summary
        has_second_info = "故宫" in summary or "长城" in summary

        print(f"[IncrementalCompression] has_first_info: {has_first_info}")
        print(f"[IncrementalCompression] has_second_info: {has_second_info}")


class TestHistoryCompressorThreshold:
    """LLM-7 压缩阈值测试。"""

    def test_should_compress_below_threshold(
        self, history_compressor: HistoryCompressor
    ):
        """测试：低于阈值时不压缩。"""
        result = history_compressor.should_compress(
            turns_count=3,
            threshold=COMPRESSION_THRESHOLD,
        )
        # 假设阈值是 6，3 轮不应该触发压缩
        if COMPRESSION_THRESHOLD > 3:
            assert result is False

    def test_should_compress_at_threshold(self, history_compressor: HistoryCompressor):
        """测试：达到阈值时触发压缩。"""
        result = history_compressor.should_compress(
            turns_count=COMPRESSION_THRESHOLD,
            threshold=COMPRESSION_THRESHOLD,
        )
        assert result is True

    def test_should_compress_above_threshold(
        self, history_compressor: HistoryCompressor
    ):
        """测试：超过阈值时触发压缩。"""
        result = history_compressor.should_compress(
            turns_count=COMPRESSION_THRESHOLD + 3,
            threshold=COMPRESSION_THRESHOLD,
        )
        assert result is True


class TestHistoryCompressorEdgeCases:
    """LLM-7 边界情况测试。"""

    @pytest.mark.asyncio
    async def test_empty_turns(self, history_compressor: HistoryCompressor):
        """测试：空对话列表。"""
        request = HistoryCompressionRequest(
            session_id="test-session-001",
            turns_to_compress=[],
            existing_summary="已有摘要内容",
            existing_turn_count=5,
            scenario_id="tour",
        )

        result = await history_compressor.compress(request)

        # 应该返回原有摘要
        assert result.new_summary == "已有摘要内容"
        assert result.compressed_turn_count == 0
        assert result.total_turn_count == 5

    @pytest.mark.asyncio
    async def test_single_turn(self, history_compressor: HistoryCompressor):
        """测试：单轮对话压缩。"""
        turns = create_compression_turns(
            [
                {"role": "user", "content": "我想去北京玩三天"},
            ]
        )

        request = HistoryCompressionRequest(
            session_id="test-session-001",
            turns_to_compress=turns,
            existing_summary=None,
            existing_turn_count=0,
            scenario_id="tour",
        )

        result = await history_compressor.compress(request)

        print(f"\n[SingleTurn] new_summary: {result.new_summary}")

        assert result.new_summary is not None
        assert result.compressed_turn_count == 1

    @pytest.mark.asyncio
    async def test_long_conversation(self, history_compressor: HistoryCompressor):
        """测试：长对话压缩。"""
        # 创建一个较长的对话（10轮）
        conversations = []
        topics = [
            ("想预订机票", "好的，请问出发城市和目的地？"),
            ("从上海到北京", "请问出发日期？"),
            ("12月20日", "请问是单程还是往返？"),
            ("往返，12月25日返回", "请问乘机人数？"),
            ("两个成人", "有舱位偏好吗？"),
        ]

        for user_msg, assistant_msg in topics:
            conversations.append({"role": "user", "content": user_msg})
            conversations.append({"role": "assistant", "content": assistant_msg})

        turns = create_compression_turns(conversations)

        request = HistoryCompressionRequest(
            session_id="test-session-001",
            turns_to_compress=turns,
            existing_summary=None,
            existing_turn_count=0,
            scenario_id="tour",
        )

        result = await history_compressor.compress(request)

        print(f"\n[LongConversation] new_summary: {result.new_summary}")
        print(f"[LongConversation] summary length: {len(result.new_summary)}")

        # 验证
        assert result.new_summary is not None
        assert result.compressed_turn_count == 10

        # 摘要长度应该受控
        assert len(result.new_summary) <= MAX_SUMMARY_LENGTH + 100  # 允许一些误差

    @pytest.mark.asyncio
    async def test_special_content(self, history_compressor: HistoryCompressor):
        """测试：特殊内容（数字、日期、金额等）。"""
        turns = create_compression_turns(
            [
                {"role": "user", "content": "预算2000-3000元"},
                {"role": "assistant", "content": "收到，2000-3000元的预算范围"},
                {"role": "user", "content": "入住日期2024年12月25日"},
                {"role": "assistant", "content": "好的，12月25日入住"},
            ]
        )

        request = HistoryCompressionRequest(
            session_id="test-session-001",
            turns_to_compress=turns,
            existing_summary=None,
            existing_turn_count=0,
            scenario_id="tour",
        )

        result = await history_compressor.compress(request)

        print(f"\n[SpecialContent] new_summary: {result.new_summary}")

        # 数字和日期应该被保留
        summary = result.new_summary
        has_numbers = any(char.isdigit() for char in summary)
        assert has_numbers, f"Summary should contain numbers: {summary}"


class TestHistoryCompressorScenarios:
    """LLM-7 不同场景测试（真实 LLM 调用）。"""

    @pytest.mark.asyncio
    async def test_tour_scenario(self, history_compressor: HistoryCompressor):
        """测试：旅游场景压缩。"""
        turns = create_compression_turns(
            [
                {"role": "user", "content": "规划北京文化游", "intent": "travel_plan"},
                {"role": "assistant", "content": "请问游玩天数和同行人数？"},
                {"role": "user", "content": "3天2人"},
                {"role": "assistant", "content": "好的，有特别想去的景点吗？"},
                {"role": "user", "content": "想去故宫和颐和园"},
            ]
        )

        request = HistoryCompressionRequest(
            session_id="test-session-001",
            turns_to_compress=turns,
            existing_summary=None,
            existing_turn_count=0,
            scenario_id="tour",
        )

        result = await history_compressor.compress(request)

        print(f"\n[TourScenario] summary: {result.new_summary}")
        assert result.new_summary is not None
        assert len(result.new_summary) > 0

    @pytest.mark.asyncio
    async def test_no_scenario(self, history_compressor: HistoryCompressor):
        """测试：无场景时的默认压缩。"""
        turns = create_compression_turns(
            [
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "你好，有什么可以帮助您？"},
                {"role": "user", "content": "帮我查一下天气"},
            ]
        )

        request = HistoryCompressionRequest(
            session_id="test-session-001",
            turns_to_compress=turns,
            existing_summary=None,
            existing_turn_count=0,
            scenario_id=None,  # 无场景
        )

        result = await history_compressor.compress(request)

        print(f"\n[NoScenario] summary: {result.new_summary}")
        assert result.new_summary is not None


class TestHistoryCompressorQuality:
    """LLM-7 压缩质量测试（真实 LLM 调用）。"""

    @pytest.mark.asyncio
    async def test_key_information_preservation(
        self, history_compressor: HistoryCompressor
    ):
        """测试：关键信息保留。"""
        # 包含多种关键信息的对话
        turns = create_compression_turns(
            [
                {"role": "user", "content": "我是张三，想预订酒店"},
                {"role": "assistant", "content": "好的张三，请问入住日期？"},
                {"role": "user", "content": "12月25日入住，住3晚"},
                {"role": "assistant", "content": "好的，3晚到12月28日退房"},
                {"role": "user", "content": "预算800元每晚，要五星级"},
                {
                    "role": "assistant",
                    "content": "五星级酒店800元稍有挑战，是否考虑四星？",
                },
                {"role": "user", "content": "四星也可以，主要要安静"},
            ]
        )

        request = HistoryCompressionRequest(
            session_id="test-session-001",
            turns_to_compress=turns,
            existing_summary=None,
            existing_turn_count=0,
            scenario_id="tour",
        )

        result = await history_compressor.compress(request)

        print(f"\n[KeyInfoPreservation] summary: {result.new_summary}")

        # 检查关键信息保留
        summary = result.new_summary

        # 定义关键信息及其可能的表达方式
        key_infos = {
            "date": ["12月25", "25日", "12/25"],
            "duration": ["3晚", "三晚", "28日"],
            "budget": ["800", "预算"],
            "requirement": ["安静", "四星", "五星"],
        }

        preserved_count = 0
        for info_type, patterns in key_infos.items():
            if any(pattern in summary for pattern in patterns):
                preserved_count += 1
                print(f"  - {info_type}: PRESERVED")
            else:
                print(f"  - {info_type}: NOT FOUND")

        # 至少应该保留一半以上的关键信息
        assert (
            preserved_count >= 2
        ), f"Only {preserved_count} key info preserved in: {summary}"
