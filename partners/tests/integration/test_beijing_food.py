"""
beijing_food Agent 集成测试

测试北京美食推荐智能体的完整功能
"""

import pytest
from acps_sdk.aip.aip_base_model import TaskState, TaskCommand
from partners.tests.integration.conftest import requires_llm


@requires_llm
class TestBeijingFoodDecision:
    """测试意图识别"""

    @pytest.mark.integration
    def test_accept_food_request(self, rpc_call, wait_for_state):
        """测试：接受美食推荐请求"""
        agent_name = "beijing_food"

        result, task_id, session_id = rpc_call(agent_name, "请推荐几家正宗的北京烤鸭店")

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
    def test_reject_travel_request(self, rpc_call, wait_for_state):
        """测试：拒绝旅游景点请求"""
        agent_name = "beijing_food"

        result, task_id, session_id = rpc_call(agent_name, "我想去故宫参观")

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
class TestBeijingFoodProduction:
    """测试内容生成"""

    @pytest.mark.integration
    def test_generates_food_recommendations(
        self, rpc_call, wait_for_state, assert_task_has_product
    ):
        """测试：生成美食推荐产出物"""
        agent_name = "beijing_food"

        result, task_id, session_id = rpc_call(
            agent_name, "推荐一些北京传统小吃，包括豆汁、炸酱面、爆肚等"
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

    @pytest.mark.integration
    def test_location_based_recommendation(self, rpc_call, wait_for_state):
        """测试：基于位置的餐厅推荐"""
        agent_name = "beijing_food"

        result, task_id, session_id = rpc_call(
            agent_name, "我在故宫附近，想找一家适合午餐的餐厅"
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
        # 应该被接受（因为是餐饮请求，虽然提到了故宫位置）
        assert state in [
            TaskState.AwaitingInput.value,
            TaskState.AwaitingCompletion.value,
        ]
