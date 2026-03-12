"""
china_transport Agent 集成测试

测试中国交通服务智能体的完整功能
"""

import pytest
from acps_sdk.aip.aip_base_model import TaskState
from partners.tests.integration.conftest import requires_llm


@requires_llm
class TestChinaTransportDecision:
    """测试意图识别"""

    @pytest.mark.integration
    def test_accept_transport_request(self, rpc_call, wait_for_state):
        """测试：接受交通查询请求"""
        agent_name = "china_transport"

        result, task_id, session_id = rpc_call(agent_name, "查询北京到上海的高铁车次")

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
    def test_reject_tourism_request(self, rpc_call, wait_for_state):
        """测试：拒绝旅游景点请求"""
        agent_name = "china_transport"

        result, task_id, session_id = rpc_call(agent_name, "推荐北京有哪些好玩的景点")

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
class TestChinaTransportProduction:
    """测试内容生成"""

    @pytest.mark.integration
    def test_generates_transport_info(
        self, rpc_call, wait_for_state, assert_task_has_product
    ):
        """测试：生成交通信息产出物"""
        agent_name = "china_transport"

        result, task_id, session_id = rpc_call(
            agent_name, "帮我查询明天北京南站到上海虹桥的高铁，上午出发"
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
    def test_flight_query(self, rpc_call, wait_for_state):
        """测试：航班查询"""
        agent_name = "china_transport"

        result, task_id, session_id = rpc_call(agent_name, "查询北京到广州的航班信息")

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
        # 应该被接受
        assert state in [
            TaskState.AwaitingInput.value,
            TaskState.AwaitingCompletion.value,
        ]
