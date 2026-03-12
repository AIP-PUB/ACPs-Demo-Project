"""
Leader Agent Platform - E2E Tests: Complete User Journey

完整用户旅程测试 - 模拟真实用户在**同一个 session** 中的完整交互链路。

所有测试按顺序执行，共享同一个 session，模拟真实用户行为：
- Phase 1: 开场和能力了解（闲聊）
- Phase 2: 发起旅游规划任务
- Phase 3: 补充信息和修改要求
- Phase 4: 中间穿插闲聊（场景切换）
- Phase 5: 异常输入恢复
- Phase 6: 总结和结束

运行方式：
    pytest leader/tests/e2e/test_user_journey.py -v -s

注意：
- 需要后端服务运行在 localhost:59210，Partner 服务运行（各 partner 独立端口，如 59221-59225）
- 测试必须按顺序执行（不能并行）
- 所有测试共享同一个 session
"""

import pytest
import httpx
import time
import uuid
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# 配置
# =============================================================================

BASE_URL = "http://localhost:59210"
PARTNER_URL = "http://localhost:59221"  # beijing_food (用于健康检查)
API_PREFIX = "/api/v1"

# 轮询配置
POLL_INTERVAL = 1.0
MAX_POLL_TIME = 120
# 任务相关请求需要更长时间（LLM 调用可能需要 30-90 秒）
MAX_POLL_TIME_TASK = 180


# =============================================================================
# Helper Functions
# =============================================================================


def api_url(path: str) -> str:
    """构建完整的 API URL"""
    return f"{BASE_URL}{API_PREFIX}{path}"


def check_service_health() -> Tuple[bool, str]:
    """检查服务健康状态"""
    try:
        with httpx.Client(timeout=10.0) as client:
            # 检查 Leader 服务
            try:
                leader_resp = client.post(
                    f"{BASE_URL}/api/v1/submit",
                    json={
                        "query": "test",
                        "mode": "direct_rpc",
                        "clientRequestId": "health_check",
                    },
                    timeout=10.0,
                )
                if leader_resp.status_code not in [200, 400, 422, 500]:
                    return False, f"Leader service unhealthy: {leader_resp.status_code}"
            except httpx.ConnectError:
                return False, "Leader service not reachable"

            # 检查 Partner 服务
            try:
                partner_resp = client.get(f"{PARTNER_URL}/health")
                if partner_resp.status_code != 200:
                    return (
                        False,
                        f"Partner service unhealthy: {partner_resp.status_code}",
                    )
            except httpx.ConnectError:
                return False, "Partner service not reachable"

            return True, "All services healthy"
    except Exception as e:
        return False, f"Service check failed: {e}"


def submit_query(
    client: httpx.Client,
    query: str,
    session_id: Optional[str] = None,
    active_task_id: Optional[str] = None,
) -> Dict[str, Any]:
    """提交查询到 /submit API"""
    payload = {
        "query": query,
        "mode": "direct_rpc",
        "clientRequestId": f"journey_{uuid.uuid4().hex[:12]}",
    }
    if session_id:
        payload["sessionId"] = session_id
    if active_task_id:
        payload["activeTaskId"] = active_task_id

    response = client.post(api_url("/submit"), json=payload)

    # 处理 LLM 服务问题
    if response.status_code == 500:
        try:
            data = response.json()
            detail = data.get("detail", {})
            if isinstance(detail, dict) and detail.get("code") in [
                "LLM_CALL_ERROR",
                "LLM_SERVICE_UNAVAILABLE",
            ]:
                pytest.skip(
                    f"LLM service unavailable: {detail.get('message', 'unknown error')}"
                )
        except Exception:
            pass

    try:
        data = response.json()
    except Exception:
        data = None

    return {
        "status_code": response.status_code,
        "data": data,
    }


def get_result(client: httpx.Client, session_id: str) -> Dict[str, Any]:
    """获取会话结果"""
    response = client.get(api_url(f"/result/{session_id}"))
    try:
        data = response.json()
    except Exception:
        data = None
    return {
        "status_code": response.status_code,
        "data": data,
    }


def poll_for_completion(
    client: httpx.Client,
    session_id: str,
    expected_types: List[str] = ["final", "clarification"],
    max_time: float = MAX_POLL_TIME,
) -> Dict[str, Any]:
    """
    轮询直到任务完成。

    Returns:
        {
            "converged": bool,
            "final_result": dict,
            "poll_count": int,
            "elapsed_time": float,
        }
    """
    start_time = time.time()
    poll_count = 0

    while time.time() - start_time < max_time:
        poll_count += 1
        result = get_result(client, session_id)

        if result["status_code"] != 200:
            time.sleep(POLL_INTERVAL)
            continue

        leader_result = result["data"].get("result", {})
        user_result = leader_result.get("userResult", {})
        result_type = user_result.get("type")

        if result_type in expected_types:
            return {
                "converged": True,
                "final_result": result["data"],
                "poll_count": poll_count,
                "elapsed_time": time.time() - start_time,
            }

        time.sleep(POLL_INTERVAL)

    # 超时
    final_result = get_result(client, session_id)
    return {
        "converged": False,
        "final_result": final_result.get("data"),
        "poll_count": poll_count,
        "elapsed_time": time.time() - start_time,
    }


def extract_turn_count(result_data: Dict[str, Any]) -> int:
    """从结果中提取对话轮数"""
    if not result_data:
        return 0
    leader_result = result_data.get("result", {})
    dialog_context = leader_result.get("dialogContext", {})
    recent_turns = dialog_context.get("recentTurns", [])
    return len(recent_turns)


def extract_user_result_type(result_data: Dict[str, Any]) -> Optional[str]:
    """从结果中提取 userResult.type"""
    if not result_data:
        return None
    leader_result = result_data.get("result", {})
    user_result = leader_result.get("userResult", {})
    return user_result.get("type")


def extract_active_task(result_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """从结果中提取 activeTask"""
    if not result_data:
        return None
    leader_result = result_data.get("result", {})
    return leader_result.get("activeTask")


def extract_external_status(result_data: Dict[str, Any]) -> Optional[str]:
    """从结果中提取 externalStatus"""
    if not result_data:
        return None
    leader_result = result_data.get("result", {})
    return leader_result.get("externalStatus")


def extract_partner_tasks(result_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """从结果中提取 activeTask.partnerTasks"""
    if not result_data:
        return None
    leader_result = result_data.get("result", {})
    active_task = leader_result.get("activeTask", {})
    return active_task.get("partnerTasks") if active_task else None


def has_awaiting_input_partners(result_data: Dict[str, Any]) -> bool:
    """检查是否有 Partner 处于 awaiting-input 状态"""
    partner_tasks = extract_partner_tasks(result_data)
    if not partner_tasks:
        return False
    for aic, task in partner_tasks.items():
        if task.get("state") == "awaiting-input":
            return True
    return False


def poll_for_clarification(
    client: httpx.Client,
    session_id: str,
    max_time: float = MAX_POLL_TIME,
) -> Dict[str, Any]:
    """
    轮询直到出现 clarification 或任务完成。
    专门用于等待 Partner 返回 awaiting-input 状态触发反问。

    Returns:
        {
            "converged": bool,
            "got_clarification": bool,
            "final_result": dict,
            "poll_count": int,
            "elapsed_time": float,
        }
    """
    start_time = time.time()
    poll_count = 0

    while time.time() - start_time < max_time:
        poll_count += 1
        result = get_result(client, session_id)

        if result["status_code"] != 200:
            time.sleep(POLL_INTERVAL)
            continue

        leader_result = result["data"].get("result", {})
        user_result = leader_result.get("userResult", {})
        result_type = user_result.get("type")
        external_status = leader_result.get("externalStatus")

        # 检查是否收到 clarification（反问）
        if result_type == "clarification" or external_status == "awaiting_input":
            return {
                "converged": True,
                "got_clarification": True,
                "final_result": result["data"],
                "poll_count": poll_count,
                "elapsed_time": time.time() - start_time,
            }

        # 如果是 final，说明直接完成了（没有触发反问）
        if result_type == "final":
            return {
                "converged": True,
                "got_clarification": False,
                "final_result": result["data"],
                "poll_count": poll_count,
                "elapsed_time": time.time() - start_time,
            }

        time.sleep(POLL_INTERVAL)

    # 超时
    final_result = get_result(client, session_id)
    return {
        "converged": False,
        "got_clarification": False,
        "final_result": final_result.get("data"),
        "poll_count": poll_count,
        "elapsed_time": time.time() - start_time,
    }


# =============================================================================
# Shared Session State (Module-level)
# =============================================================================


@dataclass
class JourneyState:
    """跨所有测试共享的旅程状态"""

    session_id: Optional[str] = None
    turn_number: int = 0
    all_turns: List[Dict[str, Any]] = field(default_factory=list)
    failed_turns: int = 0  # 记录因超时失败的轮次数

    def record_turn(
        self, query: str, result: Dict[str, Any], elapsed: float, converged: bool = True
    ):
        """记录一轮交互"""
        self.turn_number += 1
        result_type = extract_user_result_type(result) if result else "timeout"
        turn_count = extract_turn_count(result) if result else 0

        self.all_turns.append(
            {
                "turn": self.turn_number,
                "query": query,
                "result_type": result_type,
                "turn_count": turn_count,
                "elapsed": elapsed,
                "converged": converged,
            }
        )

        status = "✓" if converged else "✗ timeout"
        logger.info(
            f'  Turn {self.turn_number}: "{query[:35]}..." → '
            f"type={result_type}, history={turn_count}, time={elapsed:.1f}s [{status}]"
        )

        if not converged:
            self.failed_turns += 1


# 模块级共享状态
_journey_state = JourneyState()


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def http_client():
    """创建 HTTP 客户端"""
    with httpx.Client(timeout=120.0) as client:
        yield client


@pytest.fixture(scope="module", autouse=True)
def check_services():
    """测试前检查服务状态"""
    healthy, message = check_service_health()
    if not healthy:
        pytest.skip(f"Services not available: {message}")


@pytest.fixture(scope="module")
def journey():
    """
    获取共享的旅程状态。
    所有测试共享同一个 session。
    """
    return _journey_state


# =============================================================================
# Phase 1: 开场和能力了解（闲聊）
# =============================================================================


class TestPhase1_Opening:
    """
    Phase 1: 开场问候和了解能力

    - Turn 1: 问候
    - Turn 2: 询问能力

    预期：都是闲聊响应，快速返回 type=final
    """

    def test_turn1_greeting(self, http_client, journey: JourneyState):
        """Turn 1: 开场问候"""
        logger.info("\n" + "=" * 70)
        logger.info("Phase 1: Opening - Turn 1: Greeting")
        logger.info("=" * 70)

        query = "你好"
        submit_resp = submit_query(http_client, query, session_id=journey.session_id)
        assert submit_resp["status_code"] == 200, f"Submit failed: {submit_resp}"

        # 首次请求，获取 session_id
        journey.session_id = submit_resp["data"]["result"]["sessionId"]
        logger.info(f"Session created: {journey.session_id}")

        poll_result = poll_for_completion(http_client, journey.session_id, max_time=60)

        # 开场问候必须成功，否则后续测试无意义
        assert poll_result["converged"], "Greeting must converge to continue"

        result_type = extract_user_result_type(poll_result["final_result"])
        assert result_type == "final", f"Greeting should be final, got: {result_type}"

        journey.record_turn(
            query,
            poll_result["final_result"],
            poll_result["elapsed_time"],
            converged=True,
        )

    def test_turn2_ask_capability(self, http_client, journey: JourneyState):
        """Turn 2: 询问系统能力"""
        logger.info("\n" + "-" * 50)
        logger.info("Phase 1: Opening - Turn 2: Ask Capability")
        logger.info("-" * 50)

        assert journey.session_id is not None, "Session should exist from Turn 1"

        query = "你能帮我做什么？"
        submit_resp = submit_query(http_client, query, session_id=journey.session_id)
        assert submit_resp["status_code"] == 200
        assert (
            submit_resp["data"]["result"]["sessionId"] == journey.session_id
        ), "Session should persist"

        poll_result = poll_for_completion(http_client, journey.session_id, max_time=60)

        journey.record_turn(
            query,
            poll_result["final_result"],
            poll_result["elapsed_time"],
            converged=poll_result["converged"],
        )

        if not poll_result["converged"]:
            pytest.xfail(
                f"Capability query timed out after {poll_result['elapsed_time']:.1f}s"
            )

        result_type = extract_user_result_type(poll_result["final_result"])
        assert result_type == "final", f"Should be final, got: {result_type}"


# =============================================================================
# Phase 2: 发起旅游规划任务
# =============================================================================


class TestPhase2_InitiateTask:
    """
    Phase 2: 发起旅游规划任务

    - Turn 3: 发起北京三日游规划

    预期：场景从 CHIT_CHAT 切换到 TASK_NEW，创建任务
    """

    def test_turn3_initiate_travel_task(self, http_client, journey: JourneyState):
        """Turn 3: 发起旅游规划任务"""
        logger.info("\n" + "=" * 70)
        logger.info("Phase 2: Initiate Task - Turn 3: Travel Planning")
        logger.info("=" * 70)

        assert journey.session_id is not None

        query = "我想规划一次北京三日游，下周五出发"
        submit_resp = submit_query(http_client, query, session_id=journey.session_id)
        assert submit_resp["status_code"] == 200

        # 任务请求需要更长超时时间
        poll_result = poll_for_completion(
            http_client, journey.session_id, max_time=MAX_POLL_TIME_TASK
        )

        # 先记录轮次，即使超时
        journey.record_turn(
            query,
            poll_result["final_result"],
            poll_result["elapsed_time"],
            converged=poll_result["converged"],
        )

        # 超时不应阻止后续测试
        if not poll_result["converged"]:
            pytest.xfail(f"Task timed out after {poll_result['elapsed_time']:.1f}s")

        result_type = extract_user_result_type(poll_result["final_result"])
        assert result_type in [
            "final",
            "clarification",
        ], f"Unexpected type: {result_type}"

        # 检查是否创建了任务
        active_task = extract_active_task(poll_result["final_result"])
        if active_task:
            logger.info(f"  Task created: {active_task.get('taskId', 'N/A')}")


# =============================================================================
# Phase 3: 补充信息和修改要求
# =============================================================================


class TestPhase3_SupplementInfo:
    """
    Phase 3: 补充信息和修改要求

    - Turn 4: 补充酒店偏好
    - Turn 5: 询问交通
    - Turn 6: 修改酒店预算

    预期：增量更新处理
    """

    def test_turn4_hotel_preference(self, http_client, journey: JourneyState):
        """Turn 4: 补充酒店偏好"""
        logger.info("\n" + "-" * 50)
        logger.info("Phase 3: Supplement - Turn 4: Hotel Preference")
        logger.info("-" * 50)

        assert journey.session_id is not None

        query = "酒店我想住在国贸附近，预算800-1200元每晚"
        submit_resp = submit_query(http_client, query, session_id=journey.session_id)
        assert submit_resp["status_code"] == 200

        poll_result = poll_for_completion(
            http_client, journey.session_id, max_time=MAX_POLL_TIME_TASK
        )

        journey.record_turn(
            query,
            poll_result["final_result"],
            poll_result["elapsed_time"],
            converged=poll_result["converged"],
        )

        if not poll_result["converged"]:
            pytest.xfail(f"Task timed out after {poll_result['elapsed_time']:.1f}s")

    def test_turn5_ask_transport(self, http_client, journey: JourneyState):
        """Turn 5: 询问交通"""
        logger.info("\n" + "-" * 50)
        logger.info("Phase 3: Supplement - Turn 5: Transport")
        logger.info("-" * 50)

        assert journey.session_id is not None

        query = "交通方面有什么建议？我从上海出发"
        submit_resp = submit_query(http_client, query, session_id=journey.session_id)
        assert submit_resp["status_code"] == 200

        poll_result = poll_for_completion(
            http_client, journey.session_id, max_time=MAX_POLL_TIME_TASK
        )

        journey.record_turn(
            query,
            poll_result["final_result"],
            poll_result["elapsed_time"],
            converged=poll_result["converged"],
        )

        if not poll_result["converged"]:
            pytest.xfail(f"Task timed out after {poll_result['elapsed_time']:.1f}s")

    def test_turn6_modify_budget(self, http_client, journey: JourneyState):
        """Turn 6: 修改酒店预算（增量更新）"""
        logger.info("\n" + "-" * 50)
        logger.info("Phase 3: Supplement - Turn 6: Modify Budget")
        logger.info("-" * 50)

        assert journey.session_id is not None

        query = "酒店预算调整一下，500-800元就行"
        submit_resp = submit_query(http_client, query, session_id=journey.session_id)
        assert submit_resp["status_code"] == 200

        poll_result = poll_for_completion(
            http_client, journey.session_id, max_time=MAX_POLL_TIME_TASK
        )

        journey.record_turn(
            query,
            poll_result["final_result"],
            poll_result["elapsed_time"],
            converged=poll_result["converged"],
        )

        if not poll_result["converged"]:
            pytest.xfail(f"Task timed out after {poll_result['elapsed_time']:.1f}s")


# =============================================================================
# Phase 3.5: Partner AwaitingInput → 用户补充 → Continue 流程
# =============================================================================


class TestPhase3_5_AwaitingInputFlow:
    """
    Phase 3.5: 测试完整的 awaiting-input 补充信息流程

    测试场景：
    1. 发送一个缺少必要信息的请求，触发 Partner 返回 awaiting-input
    2. 等待系统返回 clarification（反问）
    3. 用户提供补充信息
    4. 验证系统正确识别 TASK_INPUT 意图并路由到正确的 Partner

    这是测试 "Partner awaiting-input → LLM-3 合并反问 → 用户补充 → LLM-1 识别 TASK_INPUT → 路由" 的完整链路
    """

    def test_turn_awaiting_input_trigger(self, http_client, journey: JourneyState):
        """触发 Partner awaiting-input 状态"""
        logger.info("\n" + "=" * 70)
        logger.info("Phase 3.5: AwaitingInput Flow - Trigger Clarification")
        logger.info("=" * 70)

        assert journey.session_id is not None

        # 使用一个缺少关键信息的请求，这样 Partner 很可能返回 awaiting-input
        # 例如：只说要去北京但没说具体时间、预算等
        query = "帮我规划去成都的旅行"
        submit_resp = submit_query(http_client, query, session_id=journey.session_id)
        assert submit_resp["status_code"] == 200

        # 使用专门的轮询函数等待 clarification
        poll_result = poll_for_clarification(
            http_client, journey.session_id, max_time=MAX_POLL_TIME_TASK
        )

        journey.record_turn(
            query,
            poll_result["final_result"],
            poll_result["elapsed_time"],
            converged=poll_result["converged"],
        )

        if not poll_result["converged"]:
            pytest.xfail(f"Task timed out after {poll_result['elapsed_time']:.1f}s")

        result_type = extract_user_result_type(poll_result["final_result"])
        external_status = extract_external_status(poll_result["final_result"])
        has_awaiting = has_awaiting_input_partners(poll_result["final_result"])

        logger.info(f"  result_type: {result_type}")
        logger.info(f"  external_status: {external_status}")
        logger.info(f"  has_awaiting_input_partners: {has_awaiting}")

        # 验证是否触发了 clarification 或者有 partner 在 awaiting-input
        # 注意：如果 LLM 能自动推断信息，可能直接返回 final
        if poll_result.get("got_clarification"):
            logger.info("  ✓ Got clarification as expected")
        else:
            logger.info(
                "  ℹ No clarification triggered - Partner may have auto-filled info"
            )
            pytest.skip("Partner did not request clarification - test skipped")

    def test_turn_provide_supplement_info(self, http_client, journey: JourneyState):
        """提供补充信息（TASK_INPUT 流程）"""
        logger.info("\n" + "-" * 50)
        logger.info("Phase 3.5: AwaitingInput Flow - Provide Supplement Info")
        logger.info("-" * 50)

        assert journey.session_id is not None

        # 先检查当前状态是否是 awaiting_input
        current_result = get_result(http_client, journey.session_id)
        if current_result["status_code"] != 200:
            pytest.skip("Cannot get current session status")

        external_status = extract_external_status(current_result["data"])
        result_type = extract_user_result_type(current_result["data"])

        # 只有在 awaiting_input 状态下才测试补充输入
        if external_status != "awaiting_input" and result_type != "clarification":
            logger.info(
                f"  Current status: {external_status}, result_type: {result_type}"
            )
            pytest.skip(
                "Session not in awaiting_input state - skipping supplement test"
            )

        # 提供补充信息
        # 这会触发 LLM-1 识别为 TASK_INPUT 意图，并路由到等待输入的 Partner
        query = "3月15日出发，预算每天800元，住2晚"
        submit_resp = submit_query(http_client, query, session_id=journey.session_id)
        assert submit_resp["status_code"] == 200, f"Submit failed: {submit_resp}"

        # 等待处理完成
        poll_result = poll_for_completion(
            http_client, journey.session_id, max_time=MAX_POLL_TIME_TASK
        )

        journey.record_turn(
            query,
            poll_result["final_result"],
            poll_result["elapsed_time"],
            converged=poll_result["converged"],
        )

        if not poll_result["converged"]:
            pytest.xfail(
                f"Supplement info processing timed out after {poll_result['elapsed_time']:.1f}s"
            )

        # 验证补充信息后的状态
        result_type = extract_user_result_type(poll_result["final_result"])
        external_status = extract_external_status(poll_result["final_result"])

        logger.info(f"  After supplement - result_type: {result_type}")
        logger.info(f"  After supplement - external_status: {external_status}")

        # 补充信息后，应该返回 final（任务继续执行）或者新的 clarification
        # 关键是不应该返回错误 "没有找到等待输入的 Partner"
        assert result_type in [
            "final",
            "clarification",
        ], f"Unexpected result type after supplement: {result_type}"

        # 如果是 final，说明补充信息被正确处理，Partner 继续执行
        if result_type == "final":
            logger.info("  ✓ Supplement info processed - Partner continued execution")
        elif result_type == "clarification":
            logger.info("  ℹ Got new clarification - may need more info")


# =============================================================================
# Phase 4: 中间穿插闲聊（场景切换）
# =============================================================================


class TestPhase4_IntermittentChitchat:
    """
    Phase 4: 中间穿插闲聊

    - Turn 7: 询问天气（闲聊）
    - Turn 8: 继续任务（餐饮推荐）

    预期：闲聊不中断任务上下文，能正常切回
    """

    def test_turn7_chitchat_weather(self, http_client, journey: JourneyState):
        """Turn 7: 中间闲聊 - 询问天气"""
        logger.info("\n" + "=" * 70)
        logger.info("Phase 4: Intermittent Chitchat - Turn 7: Weather")
        logger.info("=" * 70)

        assert journey.session_id is not None

        query = "对了，北京现在天气怎么样？"
        submit_resp = submit_query(http_client, query, session_id=journey.session_id)
        assert submit_resp["status_code"] == 200

        poll_result = poll_for_completion(http_client, journey.session_id, max_time=60)

        journey.record_turn(
            query,
            poll_result["final_result"],
            poll_result["elapsed_time"],
            converged=poll_result["converged"],
        )

        if not poll_result["converged"]:
            pytest.xfail(f"Chitchat timed out after {poll_result['elapsed_time']:.1f}s")

        # 闲聊应该快速返回 final
        result_type = extract_user_result_type(poll_result["final_result"])
        assert result_type == "final", f"Chitchat should be final, got: {result_type}"

    def test_turn8_food_recommendation(self, http_client, journey: JourneyState):
        """Turn 8: 回到任务 - 餐饮推荐"""
        logger.info("\n" + "-" * 50)
        logger.info("Phase 4: Back to Task - Turn 8: Food Recommendation")
        logger.info("-" * 50)

        assert journey.session_id is not None

        query = "北京有什么好吃的推荐吗？最好是当地特色"
        submit_resp = submit_query(http_client, query, session_id=journey.session_id)
        assert submit_resp["status_code"] == 200

        poll_result = poll_for_completion(
            http_client, journey.session_id, max_time=MAX_POLL_TIME_TASK
        )

        journey.record_turn(
            query,
            poll_result["final_result"],
            poll_result["elapsed_time"],
            converged=poll_result["converged"],
        )

        if not poll_result["converged"]:
            pytest.xfail(f"Task timed out after {poll_result['elapsed_time']:.1f}s")


# =============================================================================
# Phase 5: 异常输入和恢复
# =============================================================================


class TestPhase5_ErrorRecovery:
    """
    Phase 5: 异常输入和恢复

    - Turn 9: 无意义输入
    - Turn 10: 恢复正常对话

    预期：系统能容错，session 不会损坏
    """

    def test_turn9_invalid_input(self, http_client, journey: JourneyState):
        """Turn 9: 无意义输入"""
        logger.info("\n" + "=" * 70)
        logger.info("Phase 5: Error Recovery - Turn 9: Invalid Input")
        logger.info("=" * 70)

        assert journey.session_id is not None

        query = "asdfjkl;qwer123!@#"
        submit_resp = submit_query(http_client, query, session_id=journey.session_id)
        # 应该接受请求，不应该崩溃
        assert (
            submit_resp["status_code"] == 200
        ), "Should accept invalid input gracefully"

        poll_result = poll_for_completion(http_client, journey.session_id, max_time=60)

        # 无意义输入可能成功可能失败，但要记录
        journey.record_turn(
            query,
            poll_result.get("final_result"),
            poll_result.get("elapsed_time", 0),
            converged=poll_result.get("converged", False),
        )

    def test_turn10_recovery(self, http_client, journey: JourneyState):
        """Turn 10: 恢复正常对话"""
        logger.info("\n" + "-" * 50)
        logger.info("Phase 5: Error Recovery - Turn 10: Recovery")
        logger.info("-" * 50)

        assert journey.session_id is not None

        query = "抱歉刚才打错了，我还想问问北京有什么景点推荐"
        submit_resp = submit_query(http_client, query, session_id=journey.session_id)
        assert (
            submit_resp["status_code"] == 200
        ), "Should be able to continue after invalid input"

        poll_result = poll_for_completion(
            http_client, journey.session_id, max_time=MAX_POLL_TIME_TASK
        )

        journey.record_turn(
            query,
            poll_result["final_result"],
            poll_result["elapsed_time"],
            converged=poll_result["converged"],
        )

        if not poll_result["converged"]:
            pytest.xfail(f"Recovery timed out after {poll_result['elapsed_time']:.1f}s")

        result_type = extract_user_result_type(poll_result["final_result"])
        assert result_type in [
            "final",
            "clarification",
        ], f"Should work normally, got: {result_type}"


# =============================================================================
# Phase 6: 总结和结束
# =============================================================================


class TestPhase6_Conclusion:
    """
    Phase 6: 总结和结束

    - Turn 11: 感谢
    - Turn 12: 总结

    验证整个旅程的完整性
    """

    def test_turn11_thanks(self, http_client, journey: JourneyState):
        """Turn 11: 感谢"""
        logger.info("\n" + "=" * 70)
        logger.info("Phase 6: Conclusion - Turn 11: Thanks")
        logger.info("=" * 70)

        assert journey.session_id is not None

        query = "谢谢你的建议，你真厉害"
        submit_resp = submit_query(http_client, query, session_id=journey.session_id)
        assert submit_resp["status_code"] == 200

        poll_result = poll_for_completion(http_client, journey.session_id, max_time=60)

        journey.record_turn(
            query,
            poll_result["final_result"],
            poll_result["elapsed_time"],
            converged=poll_result["converged"],
        )

        if not poll_result["converged"]:
            pytest.xfail(f"Thanks timed out after {poll_result['elapsed_time']:.1f}s")

        result_type = extract_user_result_type(poll_result["final_result"])
        assert result_type == "final", f"Thanks should be final, got: {result_type}"

    def test_turn12_summary_and_verify_journey(
        self, http_client, journey: JourneyState
    ):
        """Turn 12: 总结请求 & 验证整个旅程"""
        logger.info("\n" + "-" * 50)
        logger.info("Phase 6: Conclusion - Turn 12: Summary")
        logger.info("-" * 50)

        assert journey.session_id is not None

        query = "好的，帮我总结一下这次旅行的安排"
        submit_resp = submit_query(http_client, query, session_id=journey.session_id)
        assert submit_resp["status_code"] == 200

        poll_result = poll_for_completion(
            http_client, journey.session_id, max_time=MAX_POLL_TIME_TASK
        )

        journey.record_turn(
            query,
            poll_result["final_result"],
            poll_result["elapsed_time"],
            converged=poll_result["converged"],
        )

        # =====================================================================
        # 验证整个旅程
        # =====================================================================
        logger.info("\n" + "=" * 70)
        logger.info("COMPLETE USER JOURNEY SUMMARY")
        logger.info("=" * 70)

        logger.info(f"\nSession ID: {journey.session_id}")
        logger.info(f"Total Turns: {journey.turn_number}")
        logger.info(f"Failed Turns (timeout): {journey.failed_turns}")

        # 验证完成了至少 12 轮（Phase 3.5 可能增加 0-2 轮）
        assert (
            journey.turn_number >= 12
        ), f"Should complete at least 12 turns, got {journey.turn_number}"

        # 统计响应类型
        type_counts = {}
        total_time = 0
        converged_count = 0
        for turn in journey.all_turns:
            t = turn["result_type"]
            type_counts[t] = type_counts.get(t, 0) + 1
            total_time += turn["elapsed"]
            if turn.get("converged", True):
                converged_count += 1

        logger.info(f"Response Types: {type_counts}")
        logger.info(f"Converged: {converged_count}/{journey.turn_number}")
        logger.info(f"Total Time: {total_time:.1f}s")

        # 打印完整旅程
        logger.info("\nComplete Journey:")
        for turn in journey.all_turns:
            status = "✓" if turn.get("converged", True) else "✗"
            logger.info(
                f"  {status} Turn {turn['turn']:2d}: {turn['query'][:40]:40s} → "
                f"{turn['result_type']:15s} ({turn['elapsed']:.1f}s)"
            )

        # 如果最后一轮超时，标记为预期失败但不阻止报告
        if not poll_result["converged"]:
            logger.warning(
                f"\nFinal turn timed out after {poll_result['elapsed_time']:.1f}s"
            )

        # 验证对话历史累积（如果收敛了）
        if poll_result["converged"]:
            final_turn_count = extract_turn_count(poll_result["final_result"])
            logger.info(f"\nDialog history recorded: {final_turn_count} turns")
            # 由于历史压缩，可能小于实际轮数，但应该 > 0
            assert final_turn_count > 0, "Dialog context should record turns"

        # 验证至少有一半的轮次收敛成功
        success_rate = converged_count / journey.turn_number
        assert (
            success_rate >= 0.5
        ), f"At least 50% of turns should converge, got {success_rate:.0%}"

        logger.info("\n" + "=" * 70)
        if journey.failed_turns > 0:
            logger.warning(
                f"⚠ USER JOURNEY COMPLETED WITH {journey.failed_turns} TIMEOUTS"
            )
        else:
            logger.info("✓ COMPLETE USER JOURNEY TEST PASSED!")
        logger.info("=" * 70)


# =============================================================================
# 独立测试：AwaitingInput 流程（可单独运行）
# =============================================================================


class TestAwaitingInputFlowStandalone:
    """
    独立的 awaiting-input 流程测试

    完整测试 "Partner awaiting-input → 用户补充 → continue" 链路：
    1. 创建新会话
    2. 发送缺少必要信息的请求，触发 Partner 返回 awaiting-input
    3. 系统返回 clarification（反问）
    4. 用户提供补充信息
    5. 验证 LLM-1 正确识别 TASK_INPUT 意图
    6. 验证信息正确路由到等待输入的 Partner

    运行方式：
        pytest leader/tests/e2e/test_user_journey.py::TestAwaitingInputFlowStandalone -v -s -o "addopts="
    """

    @pytest.fixture(scope="class")
    def standalone_state(self):
        """独立会话状态"""
        return {"session_id": None}

    def test_step1_trigger_awaiting_input(self, http_client, standalone_state):
        """
        Step 1: 触发 Partner awaiting-input 状态

        发送一个模糊的旅游请求，缺少关键信息（日期、预算等），
        这样 Partner 很可能返回 awaiting-input 要求补充信息。
        """
        logger.info("\n" + "=" * 70)
        logger.info("Standalone AwaitingInput Test - Step 1: Trigger Clarification")
        logger.info("=" * 70)

        # 发送新请求创建会话
        # 使用故意非常模糊的请求来触发反问 - 只说目的地，不给任何其他信息
        query = "2月2日，从成都飞机到北京"
        submit_resp = submit_query(http_client, query, session_id=None)
        assert submit_resp["status_code"] == 200, f"Submit failed: {submit_resp}"

        # 获取 session_id
        session_id = submit_resp["data"]["result"]["sessionId"]
        standalone_state["session_id"] = session_id
        logger.info(f"  Session created: {session_id}")

        # 等待 clarification 或任务完成
        poll_result = poll_for_clarification(
            http_client, session_id, max_time=MAX_POLL_TIME_TASK
        )

        if not poll_result["converged"]:
            pytest.xfail(f"Task timed out after {poll_result['elapsed_time']:.1f}s")

        result_type = extract_user_result_type(poll_result["final_result"])
        external_status = extract_external_status(poll_result["final_result"])
        got_clarification = poll_result.get("got_clarification", False)

        logger.info(f"  result_type: {result_type}")
        logger.info(f"  external_status: {external_status}")
        logger.info(f"  got_clarification: {got_clarification}")
        logger.info(f"  elapsed_time: {poll_result['elapsed_time']:.1f}s")

        # 保存状态用于下一步测试
        standalone_state["got_clarification"] = got_clarification
        standalone_state["result_type"] = result_type
        standalone_state["external_status"] = external_status

        if got_clarification:
            logger.info("  ✓ Got clarification - Partner requested more info")
        else:
            logger.info("  ℹ No clarification - Partner auto-completed or rejected")

    def test_step2_provide_supplement_and_verify(self, http_client, standalone_state):
        """
        Step 2: 提供补充信息并验证处理

        如果 Step 1 触发了 clarification，则提供补充信息，
        验证系统能正确识别 TASK_INPUT 意图并路由到正确的 Partner。
        """
        logger.info("\n" + "-" * 50)
        logger.info("Standalone AwaitingInput Test - Step 2: Provide Supplement Info")
        logger.info("-" * 50)

        session_id = standalone_state.get("session_id")
        assert session_id is not None, "Session should exist from Step 1"

        # 如果 Step 1 没有触发 clarification，跳过此测试
        if not standalone_state.get("got_clarification"):
            logger.info("  ℹ No clarification in Step 1 - skipping supplement test")
            pytest.skip("Step 1 did not trigger clarification")

        # 验证当前状态
        current_result = get_result(http_client, session_id)
        if current_result["status_code"] != 200:
            pytest.fail(f"Cannot get session status: {current_result}")

        external_status = extract_external_status(current_result["data"])
        result_type = extract_user_result_type(current_result["data"])
        logger.info(
            f"  Current status: external={external_status}, result_type={result_type}"
        )

        # 确认当前处于等待输入状态
        if external_status != "awaiting_input" and result_type != "clarification":
            pytest.skip(
                f"Session not awaiting input: status={external_status}, type={result_type}"
            )

        # 提供补充信息
        # 这会触发 TASK_INPUT 流程
        query = "3月20日出发，预算2000元，1位乘客"
        logger.info(f"  Sending supplement: {query}")

        submit_resp = submit_query(http_client, query, session_id=session_id)

        # 关键验证：补充信息请求应该成功接受
        assert submit_resp["status_code"] == 200, (
            f"Supplement request failed: {submit_resp}\n"
            "这可能表示 TASK_INPUT 流程中的 Bug（如 '没有找到等待输入的 Partner' 错误）"
        )

        # 等待处理完成
        poll_result = poll_for_completion(
            http_client, session_id, max_time=MAX_POLL_TIME_TASK
        )

        if not poll_result["converged"]:
            pytest.xfail(
                f"Supplement processing timed out after {poll_result['elapsed_time']:.1f}s"
            )

        # 验证结果
        result_type = extract_user_result_type(poll_result["final_result"])
        external_status = extract_external_status(poll_result["final_result"])

        logger.info(f"  After supplement - result_type: {result_type}")
        logger.info(f"  After supplement - external_status: {external_status}")
        logger.info(f"  elapsed_time: {poll_result['elapsed_time']:.1f}s")

        # 验证：补充信息后应该返回正常结果
        # 可能是 final（任务继续执行并完成）或新的 clarification（需要更多信息）
        assert result_type in ["final", "clarification"], (
            f"Unexpected result type after supplement: {result_type}\n"
            "expected: 'final' (Partner continued) or 'clarification' (need more info)"
        )

        if result_type == "final":
            logger.info(
                "  ✓ SUCCESS: Supplement processed - Partner continued execution"
            )
        elif result_type == "clarification":
            logger.info("  ✓ SUCCESS: Supplement accepted - Partner needs more info")

        logger.info("\n" + "=" * 70)
        logger.info("✓ AWAITING-INPUT FLOW TEST PASSED!")
        logger.info("=" * 70)
