"""
beijing_rural Agent 集成测试

测试北京郊区旅游智能体的完整功能
"""

import pytest
from acps_sdk.aip.aip_base_model import TaskState, TaskCommand
from partners.tests.integration.conftest import requires_llm


@requires_llm
class TestBeijingRuralDecision:
    """测试意图识别"""

    @pytest.mark.integration
    def test_accept_suburban_request(self, rpc_call, wait_for_state):
        """测试：接受郊区旅游请求"""
        agent_name = "beijing_rural"

        result, task_id, session_id = rpc_call(
            agent_name, "我想去慕田峪长城和十三陵玩两天"
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
    def test_reject_city_core_request(self, rpc_call, wait_for_state):
        """测试：拒绝城区旅游请求"""
        agent_name = "beijing_rural"

        result, task_id, session_id = rpc_call(agent_name, "我想去故宫和天坛玩")

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
class TestBeijingRuralProduction:
    """测试内容生成"""

    @pytest.mark.integration
    def test_generates_rural_itinerary(
        self, rpc_call, wait_for_state, assert_task_has_product
    ):
        """测试：生成郊区行程产出物"""
        agent_name = "beijing_rural"

        result, task_id, session_id = rpc_call(
            agent_name, "帮我规划一个去八达岭长城的一日游行程，想体验自然风光"
        )

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
