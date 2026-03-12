"""
beijing_urban Agent 集成测试

测试北京城区旅游智能体的完整功能：
- 意图识别（接受/拒绝）
- 需求分析与补全
- 行程生成
"""

import pytest
import time
from acps_sdk.aip.aip_base_model import TaskState, TaskCommandType
from partners.tests.integration.conftest import requires_llm


@requires_llm
class TestBeijingUrbanDecision:
    """测试意图识别与准入判断"""

    @pytest.mark.integration
    def test_accept_city_core_request(
        self, rpc_call, wait_for_state, assert_rpc_success
    ):
        """测试：接受城六区内的旅游请求"""
        agent_name = "beijing_urban"

        result, task_id, session_id = rpc_call(
            agent_name, "我想去北京故宫和天坛玩两天，预算中等，喜欢文化景点"
        )

        assert_rpc_success(result)

        # 等待处理完成
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

        # 应该被接受（不是 Rejected）
        state = final_result["result"]["status"]["state"]
        assert (
            state != TaskState.Rejected.value
        ), f"Request should be accepted but was rejected"

    @pytest.mark.integration
    def test_reject_suburban_request(self, rpc_call, wait_for_state):
        """测试：拒绝郊区旅游请求"""
        agent_name = "beijing_urban"

        result, task_id, session_id = rpc_call(agent_name, "我想去八达岭长城和十三陵玩")

        # 等待处理完成
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
        # 可能是 Rejected 或者在 AwaitingInput 状态说明需要澄清
        # 具体行为取决于 LLM 的判断
        assert state in [TaskState.Rejected.value, TaskState.AwaitingInput.value]

    @pytest.mark.integration
    def test_reject_food_request(self, rpc_call, wait_for_state):
        """测试：拒绝美食推荐请求（超出职责）"""
        agent_name = "beijing_urban"

        result, task_id, session_id = rpc_call(agent_name, "请推荐北京好吃的烤鸭店")

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
class TestBeijingUrbanAnalysis:
    """测试需求分析阶段"""

    @pytest.mark.integration
    def test_complete_requirements_flow(
        self, rpc_call, wait_for_state, get_task_context
    ):
        """测试：完整需求信息直接进入生产阶段"""
        agent_name = "beijing_urban"

        result, task_id, session_id = rpc_call(
            agent_name,
            "我想去北京城区玩两天，预算中等，主要想看故宫、天坛、颐和园这些文化景点，"
            "偏好轻体力行程，有老人同行",
        )

        final_result = wait_for_state(
            agent_name,
            task_id,
            session_id,
            [TaskState.AwaitingInput, TaskState.AwaitingCompletion],
            timeout=90,
        )

        # 信息完整应该进入 AwaitingCompletion
        state = final_result["result"]["status"]["state"]

        if state == TaskState.AwaitingCompletion.value:
            # 验证内部 requirements 被填充
            ctx = get_task_context(agent_name, task_id)
            if ctx:
                assert ctx.requirements is not None

    @pytest.mark.integration
    def test_missing_info_requests_clarification(self, rpc_call, wait_for_state):
        """测试：信息缺失时请求补充"""
        agent_name = "beijing_urban"

        result, task_id, session_id = rpc_call(
            agent_name, "我想去北京玩"  # 信息很不完整
        )

        final_result = wait_for_state(
            agent_name,
            task_id,
            session_id,
            [TaskState.AwaitingInput, TaskState.AwaitingCompletion, TaskState.Rejected],
            timeout=60,
        )

        state = final_result["result"]["status"]["state"]

        # 应该进入 AwaitingInput 请求更多信息
        if state == TaskState.AwaitingInput.value:
            # 验证有追问信息
            data_items = final_result["result"]["status"].get("dataItems", [])
            assert len(data_items) > 0

    @pytest.mark.integration
    def test_continue_with_additional_info(self, rpc_call, wait_for_state):
        """测试：通过 Continue 补充信息"""
        agent_name = "beijing_urban"

        # 第一次请求（不完整）
        result, task_id, session_id = rpc_call(agent_name, "我想去北京城区玩")

        # 等待进入 AwaitingInput
        result = wait_for_state(
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

        state = result["result"]["status"]["state"]

        if state == TaskState.AwaitingInput.value:
            # 补充信息
            result, _, _ = rpc_call(
                agent_name,
                "两天时间，想去故宫和天坛，预算中等",
                TaskCommandType.Continue,
                task_id,
                session_id,
            )

            # 等待处理
            final_result = wait_for_state(
                agent_name,
                task_id,
                session_id,
                [TaskState.AwaitingInput, TaskState.AwaitingCompletion],
                timeout=90,
            )

            # 应该继续处理
            final_state = final_result["result"]["status"]["state"]
            assert final_state in [
                TaskState.AwaitingInput.value,
                TaskState.AwaitingCompletion.value,
            ]


@requires_llm
class TestBeijingUrbanProduction:
    """测试内容生成阶段"""

    @pytest.mark.integration
    def test_generates_itinerary_product(
        self, rpc_call, wait_for_state, assert_task_has_product
    ):
        """测试：生成行程规划产出物"""
        agent_name = "beijing_urban"

        result, task_id, session_id = rpc_call(
            agent_name,
            "帮我规划北京城区两日游，第一天去故宫和景山公园，"
            "第二天去天坛和国家博物馆，预算1000元，轻体力行程",
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
            # 验证有产出物
            assert_task_has_product(final_result)

            # 验证产出物内容
            products = final_result["result"].get("products", [])
            assert len(products) > 0

            # 产出物应包含文本内容
            product_texts = []
            for prod in products:
                for item in prod.get("dataItems", []):
                    if item.get("type") == "text":
                        product_texts.append(item.get("text", ""))

            assert len(product_texts) > 0
            # 产出物应该是有意义的内容
            total_text = " ".join(product_texts)
            assert len(total_text) > 100  # 至少有一定长度

    @pytest.mark.integration
    def test_complete_task_flow(
        self, rpc_call, wait_for_state, assert_task_state, continue_task
    ):
        """测试：完整任务流程到 Completed"""
        agent_name = "beijing_urban"

        # 提供详细的需求信息
        result, task_id, session_id = rpc_call(
            agent_name,
            "规划一个北京城区一日游，日期2026年2月1日，上午去故宫，下午去天坛，"
            "预算500元，1人出行，步行加地铁出行",
        )

        # 等待生产完成，也接受 AwaitingInput 状态
        result = wait_for_state(
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

        state = result["result"]["status"]["state"]

        # 如果任务在 AwaitingInput，继续提供信息
        if state == TaskState.AwaitingInput.value:
            continue_task(
                agent_name,
                task_id,
                session_id,
                "日期2026年2月1日，1人出行，预算500元，不需要住宿",
            )
            result = wait_for_state(
                agent_name,
                task_id,
                session_id,
                [TaskState.AwaitingCompletion, TaskState.Rejected, TaskState.Failed],
                timeout=120,
            )
            state = result["result"]["status"]["state"]

        if state == TaskState.AwaitingCompletion.value:
            # 发送 Complete 确认
            result, _, _ = rpc_call(
                agent_name,
                "方案很好，确认完成",
                TaskCommandType.Complete,
                task_id,
                session_id,
            )

            # 验证任务完成
            assert_task_state(result, TaskState.Completed)


@requires_llm
class TestBeijingUrbanInternalState:
    """测试内部状态验证"""

    @pytest.mark.integration
    def test_task_context_stores_requirements(
        self, rpc_call, wait_for_state, get_task_context
    ):
        """测试：任务上下文正确存储 requirements"""
        agent_name = "beijing_urban"

        result, task_id, session_id = rpc_call(
            agent_name, "北京城区两日游，游览故宫天坛，预算1000元，文化深度游"
        )

        wait_for_state(
            agent_name,
            task_id,
            session_id,
            [TaskState.AwaitingInput, TaskState.AwaitingCompletion],
            timeout=90,
        )

        # 获取内部任务上下文
        ctx = get_task_context(agent_name, task_id)

        if ctx and ctx.requirements:
            # 验证 requirements 结构
            reqs = ctx.requirements

            # 应该有一些基本字段被提取
            # 具体字段取决于 LLM 的分析结果
            assert isinstance(reqs, dict)

    @pytest.mark.integration
    def test_message_history_tracked(self, rpc_call, wait_for_state, get_task_context):
        """测试：消息历史被正确记录"""
        agent_name = "beijing_urban"

        result, task_id, session_id = rpc_call(agent_name, "北京城区一日游")

        result = wait_for_state(
            agent_name,
            task_id,
            session_id,
            [TaskState.AwaitingInput, TaskState.AwaitingCompletion, TaskState.Rejected],
            timeout=60,
        )

        state = result["result"]["status"]["state"]

        if state == TaskState.AwaitingInput.value:
            # 发送 Continue
            rpc_call(
                agent_name,
                "去故宫和天坛",
                TaskCommandType.Continue,
                task_id,
                session_id,
            )

            time.sleep(1)

            # 验证消息历史
            ctx = get_task_context(agent_name, task_id)

            if ctx:
                assert ctx.task.messageHistory is not None
                assert len(ctx.task.messageHistory) >= 2
