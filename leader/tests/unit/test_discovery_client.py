"""
Leader Agent Platform - Discovery Client 单元测试

测试内容：
1. DiscoveryClient 初始化与配置
2. discover 方法的请求构建与响应解析
3. discover_for_dimension 维度查询方法
4. 错误处理（超时、HTTP 错误、服务未配置等）
"""

import sys
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

import httpx

from assistant.services.discovery_client import (
    DiscoveryClient,
    DiscoveryClientError,
)
from acps_sdk.adp import (
    DiscoveryResponse,
    DiscoveryResult,
    DiscoveryAgentGroup,
    DiscoveryAgentSkill,
)


# =============================================================================
# 基础配置测试
# =============================================================================


class TestDiscoveryClientConfig:
    """测试 DiscoveryClient 配置。"""

    def test_is_configured_with_url(self):
        """有 server_base_url 时 is_configured 为 True。"""
        client = DiscoveryClient(server_base_url="https://ds.example.com")
        assert client.is_configured is True

    def test_is_not_configured_without_url(self):
        """无 server_base_url 时 is_configured 为 False。"""
        client = DiscoveryClient(server_base_url="")
        assert client.is_configured is False

    def test_trailing_slash_stripped(self):
        """server_base_url 尾部斜杠会被去掉。"""
        client = DiscoveryClient(server_base_url="https://ds.example.com/")
        assert client._server_base_url == "https://ds.example.com"


# =============================================================================
# discover 方法测试
# =============================================================================


class TestDiscoverMethod:
    """测试 discover 方法。"""

    @pytest.mark.asyncio
    async def test_discover_not_configured_raises(self):
        """未配置 server_base_url 时调用 discover 应抛出异常。"""
        client = DiscoveryClient(server_base_url="")
        with pytest.raises(DiscoveryClientError, match="未配置"):
            await client.discover(query="测试查询")

    @pytest.mark.asyncio
    async def test_discover_success(self):
        """成功的发现请求应返回 DiscoveryResponse。"""
        client = DiscoveryClient(
            server_base_url="https://ds.example.com",
            timeout=5,
            default_limit=3,
        )

        mock_response_data = {
            "result": {
                "acsMap": {
                    "AIC-001": {
                        "name": "FoodAgent",
                        "description": "美食推荐智能体",
                    }
                },
                "agents": [
                    {
                        "group": "美食推荐",
                        "agentSkills": [
                            {
                                "aic": "AIC-001",
                                "skillId": "food-rec",
                                "ranking": 1,
                            }
                        ],
                    }
                ],
            }
        }

        mock_http_response = MagicMock()
        mock_http_response.status_code = 200
        mock_http_response.json.return_value = mock_response_data
        mock_http_response.raise_for_status = MagicMock()

        with patch("assistant.services.discovery_client.httpx.AsyncClient") as mock_cls:
            mock_client_instance = AsyncMock()
            mock_client_instance.post.return_value = mock_http_response
            mock_client_instance.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client_instance

            response = await client.discover(query="我需要美食推荐")

        assert response.is_success()
        assert len(response.result.agents) == 1
        assert response.result.agents[0].agent_skills[0].aic == "AIC-001"

    @pytest.mark.asyncio
    async def test_discover_http_error(self):
        """HTTP 错误应抛出 DiscoveryClientError。"""
        client = DiscoveryClient(server_base_url="https://ds.example.com")

        mock_request = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 500

        with patch("assistant.services.discovery_client.httpx.AsyncClient") as mock_cls:
            mock_client_instance = AsyncMock()
            mock_client_instance.post.side_effect = httpx.HTTPStatusError(
                "Server Error",
                request=mock_request,
                response=mock_resp,
            )
            mock_client_instance.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client_instance

            with pytest.raises(DiscoveryClientError, match="返回错误"):
                await client.discover(query="测试")

    @pytest.mark.asyncio
    async def test_discover_timeout(self):
        """超时应抛出 DiscoveryClientError。"""
        client = DiscoveryClient(server_base_url="https://ds.example.com", timeout=1)

        with patch("assistant.services.discovery_client.httpx.AsyncClient") as mock_cls:
            mock_client_instance = AsyncMock()
            mock_client_instance.post.side_effect = httpx.TimeoutException("timeout")
            mock_client_instance.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client_instance

            with pytest.raises(DiscoveryClientError, match="超时"):
                await client.discover(query="测试")


# =============================================================================
# discover_for_dimension 方法测试
# =============================================================================


class TestDiscoverForDimension:
    """测试 discover_for_dimension 方法。"""

    @pytest.mark.asyncio
    async def test_builds_query_from_dimension(self):
        """应基于维度名称和描述构建查询文本。"""
        client = DiscoveryClient(server_base_url="https://ds.example.com")

        mock_response_data = {"result": {"agents": []}}
        mock_http_response = MagicMock()
        mock_http_response.status_code = 200
        mock_http_response.json.return_value = mock_response_data
        mock_http_response.raise_for_status = MagicMock()

        with patch("assistant.services.discovery_client.httpx.AsyncClient") as mock_cls:
            mock_client_instance = AsyncMock()
            mock_client_instance.post.return_value = mock_http_response
            mock_client_instance.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client_instance

            await client.discover_for_dimension(
                dimension_name="住宿",
                dimension_description="酒店预订和推荐",
            )

            # 验证 POST 被调用，且 payload 中 query 包含维度信息
            call_args = mock_client_instance.post.call_args
            payload = call_args.kwargs.get("json") or call_args[1].get("json")
            assert "住宿" in payload["query"]
            assert "酒店预订和推荐" in payload["query"]
