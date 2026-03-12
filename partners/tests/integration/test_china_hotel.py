"""
china_hotel Agent 集成测试

测试中国酒店预订智能体的完整功能
"""

import pytest
from acps_sdk.aip.aip_base_model import TaskState, TaskCommand
from partners.tests.integration.conftest import requires_llm


@requires_llm
class TestChinaHotelDecision:
    """测试意图识别"""

    @pytest.mark.integration
    def test_accept_hotel_request(self, rpc_call, wait_for_state):
        """测试：接受酒店预订请求"""
        agent_name = "china_hotel"

        result, task_id, session_id = rpc_call(
            agent_name, "帮我在北京找一家五星级酒店，入住三晚"
        )

        final_result = wait_for_state(
            agent_name,
            task_id,
            session_id,
            [
                TaskState.AwaitingInput,
                TaskState.AwaitingCompletion,
                TaskState.Rejected,
                TaskState.Failed,
            ],
            timeout=60,
        )

        state = final_result["result"]["status"]["state"]
        assert state != TaskState.Rejected.value

    @pytest.mark.integration
    def test_reject_travel_planning_request(self, rpc_call, wait_for_state):
        """测试：拒绝旅游规划请求"""
        agent_name = "china_hotel"

        result, task_id, session_id = rpc_call(agent_name, "帮我规划一个北京三日游行程")

        final_result = wait_for_state(
            agent_name,
            task_id,
            session_id,
            [
                TaskState.AwaitingInput,
                TaskState.AwaitingCompletion,
                TaskState.Rejected,
                TaskState.Failed,
            ],
            timeout=60,
        )

        state = final_result["result"]["status"]["state"]
        assert state == TaskState.Rejected.value


@requires_llm
class TestChinaHotelProduction:
    """测试内容生成"""

    @pytest.mark.integration
    def test_generates_hotel_recommendations(
        self, rpc_call, wait_for_state, assert_task_has_product, continue_task
    ):
        """测试：生成酒店推荐产出物"""
        agent_name = "china_hotel"

        # 提供尽可能完整的需求信息，减少被要求补充信息的可能
        result, task_id, session_id = rpc_call(
            agent_name,
            "推荐北京王府井附近的酒店，入住日期2026年2月1日，退房2月3日，"
            "预算每晚500-800元，需要含早餐，双床房，2位成人入住",
        )

        # 首先等待任务进入某个中间或终态
        final_result = wait_for_state(
            agent_name,
            task_id,
            session_id,
            [
                TaskState.AwaitingInput,
                TaskState.AwaitingCompletion,
                TaskState.Rejected,
                TaskState.Failed,
            ],
            timeout=120,
        )

        state = final_result["result"]["status"]["state"]

        # 如果任务在 AwaitingInput，说明 LLM 需要更多信息，继续提供
        if state == TaskState.AwaitingInput.value:
            # 继续提供补充信息
            continue_task(
                agent_name,
                task_id,
                session_id,
                "入住日期2026年2月1日，退房2月3日，2位成人，需要双床房",
            )

            # 再次等待
            final_result = wait_for_state(
                agent_name,
                task_id,
                session_id,
                [TaskState.AwaitingCompletion, TaskState.Rejected, TaskState.Failed],
                timeout=120,
            )
            state = final_result["result"]["status"]["state"]

        if state == TaskState.AwaitingCompletion.value:
            assert_task_has_product(final_result)
