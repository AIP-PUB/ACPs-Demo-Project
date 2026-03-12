"""
配置加载单元测试

测试 Partner Agent 的配置加载功能：
- ACS (acs.json) 加载
- Config (config.toml) 加载
- Prompts (prompts.toml) 加载
- LLM 客户端配置
"""

import pytest
import os
import json
import tempfile
import shutil
from unittest.mock import patch, MagicMock, AsyncMock


class TestConfigLoading:
    """测试配置文件加载"""

    def test_load_acs(self, test_agent_dir, test_agent_acs):
        """测试：正确加载 ACS 配置"""
        from partners.generic_runner import GenericRunner

        with patch("partners.generic_runner.AsyncOpenAI"):
            runner = GenericRunner("test_agent", test_agent_dir)

            assert runner.acs["aic"] == test_agent_acs["aic"]
            assert runner.acs["name"] == test_agent_acs["name"]
            assert runner.acs["description"] == test_agent_acs["description"]
            assert len(runner.acs["skills"]) == len(test_agent_acs["skills"])

    def test_load_config(self, test_agent_dir, test_agent_config):
        """测试：正确加载 config.toml"""
        from partners.generic_runner import GenericRunner

        with patch("partners.generic_runner.AsyncOpenAI"):
            runner = GenericRunner("test_agent", test_agent_dir)

            assert "llm" in runner.config
            assert "default" in runner.config["llm"]
            assert (
                runner.config["llm"]["default"]["model"]
                == test_agent_config["llm"]["default"]["model"]
            )
            assert runner.config["concurrency"]["max_concurrent_tasks"] == 5

    def test_load_prompts(self, test_agent_dir, test_agent_prompts):
        """测试：正确加载 prompts.toml"""
        from partners.generic_runner import GenericRunner

        with patch("partners.generic_runner.AsyncOpenAI"):
            runner = GenericRunner("test_agent", test_agent_dir)

            assert "decision" in runner.prompts
            assert "analysis" in runner.prompts
            assert "production" in runner.prompts

            assert runner.prompts["decision"]["llm_profile"] == "fast"
            assert "{{output_schema}}" in runner.prompts["decision"]["system"]

    def test_missing_acs_raises_error(self, test_agent_dir):
        """测试：缺少 acs.json 时抛出错误"""
        from partners.generic_runner import GenericRunner

        # 删除 acs.json
        acs_path = os.path.join(test_agent_dir, "acs.json")
        os.remove(acs_path)

        with pytest.raises(FileNotFoundError):
            with patch("partners.generic_runner.AsyncOpenAI"):
                GenericRunner("test_agent", test_agent_dir)

    def test_missing_config_raises_error(self, test_agent_dir):
        """测试：缺少 config.toml 时抛出错误"""
        from partners.generic_runner import GenericRunner

        # 删除 config.toml
        config_path = os.path.join(test_agent_dir, "config.toml")
        os.remove(config_path)

        with pytest.raises(FileNotFoundError):
            with patch("partners.generic_runner.AsyncOpenAI"):
                GenericRunner("test_agent", test_agent_dir)

    def test_missing_prompts_raises_error(self, test_agent_dir):
        """测试：缺少 prompts.toml 时抛出错误"""
        from partners.generic_runner import GenericRunner

        # 删除 prompts.toml
        prompts_path = os.path.join(test_agent_dir, "prompts.toml")
        os.remove(prompts_path)

        with pytest.raises(FileNotFoundError):
            with patch("partners.generic_runner.AsyncOpenAI"):
                GenericRunner("test_agent", test_agent_dir)


class TestLLMClientSetup:
    """测试 LLM 客户端配置"""

    def test_multiple_llm_profiles(self, test_agent_dir):
        """测试：支持多个 LLM Profile"""
        from partners.generic_runner import GenericRunner

        mock_openai = MagicMock()
        with patch("partners.generic_runner.AsyncOpenAI", mock_openai):
            runner = GenericRunner("test_agent", test_agent_dir)

            assert "default" in runner.llm_clients
            assert "fast" in runner.llm_clients

    def test_get_llm_client_by_profile(self, test_agent_dir):
        """测试：按 Profile 名称获取 LLM 客户端"""
        from partners.generic_runner import GenericRunner

        mock_client_default = MagicMock()
        mock_client_fast = MagicMock()

        call_count = [0]

        def create_client(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_client_fast
            return mock_client_default

        with patch("partners.generic_runner.AsyncOpenAI", side_effect=create_client):
            runner = GenericRunner("test_agent", test_agent_dir)

            fast_client = runner._get_llm_client("fast")
            default_client = runner._get_llm_client("default")

            # 应该返回不同的客户端
            assert fast_client is not None
            assert default_client is not None

    def test_fallback_to_default_profile(self, test_agent_dir):
        """测试：未知 Profile 回退到 default"""
        from partners.generic_runner import GenericRunner

        with patch("partners.generic_runner.AsyncOpenAI"):
            runner = GenericRunner("test_agent", test_agent_dir)

            # 请求不存在的 Profile
            client = runner._get_llm_client("nonexistent")

            # 应该回退到 default
            assert client == runner.llm_clients.get("default")

    def test_get_model_name(self, test_agent_dir, test_agent_config):
        """测试：获取模型名称"""
        from partners.generic_runner import GenericRunner

        with patch("partners.generic_runner.AsyncOpenAI"):
            runner = GenericRunner("test_agent", test_agent_dir)

            model = runner._get_model_name("default")
            assert model == test_agent_config["llm"]["default"]["model"]

            fast_model = runner._get_model_name("fast")
            assert fast_model == test_agent_config["llm"]["fast"]["model"]

    def test_api_key_from_env(self, test_agent_acs, test_agent_prompts):
        """测试：从环境变量读取 API Key"""
        import tomli_w

        temp_dir = tempfile.mkdtemp(prefix="test_env_")

        try:
            # 写入配置，使用环境变量
            config = {
                "llm": {
                    "default": {
                        "api_key": "TEST_API_KEY_ENV",  # 环境变量名
                        "base_url": "https://api.test.com/v1/",
                        "model": "test-model",
                    }
                },
                "concurrency": {"max_concurrent_tasks": 5},
                "log": {"level": "DEBUG"},
            }

            with open(os.path.join(temp_dir, "acs.json"), "w") as f:
                json.dump(test_agent_acs, f)
            with open(os.path.join(temp_dir, "config.toml"), "wb") as f:
                tomli_w.dump(config, f)
            with open(os.path.join(temp_dir, "prompts.toml"), "wb") as f:
                tomli_w.dump(test_agent_prompts, f)

            # 设置环境变量
            os.environ["TEST_API_KEY_ENV"] = "actual-api-key-from-env"

            from partners.generic_runner import GenericRunner

            captured_key = [None]

            def capture_openai(*args, **kwargs):
                captured_key[0] = kwargs.get("api_key")
                return MagicMock()

            with patch(
                "partners.generic_runner.AsyncOpenAI", side_effect=capture_openai
            ):
                runner = GenericRunner("test_agent", temp_dir)

            assert captured_key[0] == "actual-api-key-from-env"

        finally:
            del os.environ["TEST_API_KEY_ENV"]
            shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_call_llm_uses_stage_temperature_without_max_tokens(
        self, test_agent_dir
    ):
        """测试：按阶段写死 temperature，且不再传 max_tokens"""
        from partners.generic_runner import GenericRunner

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ok"
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("partners.generic_runner.AsyncOpenAI", return_value=mock_client):
            runner = GenericRunner("test_agent", test_agent_dir)
            await runner._call_llm("decision", "fast", "system", "user")
            await runner._call_llm("production", "default", "system", "user")

        decision_kwargs = mock_client.chat.completions.create.await_args_list[0].kwargs
        production_kwargs = mock_client.chat.completions.create.await_args_list[1].kwargs

        assert decision_kwargs["temperature"] == 0.2
        assert production_kwargs["temperature"] == 0.6
        assert "max_tokens" not in decision_kwargs
        assert "max_tokens" not in production_kwargs


class TestResponsibilitiesPrompt:
    """测试职责描述生成"""

    def test_build_responsibilities_prompt(self, test_agent_dir, test_agent_acs):
        """测试：从 ACS 生成职责描述"""
        from partners.generic_runner import GenericRunner

        with patch("partners.generic_runner.AsyncOpenAI"):
            runner = GenericRunner("test_agent", test_agent_dir)

            responsibilities = runner._build_responsibilities_prompt()

            # 应包含 description
            assert test_agent_acs["description"] in responsibilities

            # 应包含 skills
            for skill in test_agent_acs["skills"]:
                assert skill["name"] in responsibilities

    def test_responsibilities_without_skills(
        self, test_agent_config, test_agent_prompts
    ):
        """测试：没有 skills 时只返回 description"""
        import tomli_w

        temp_dir = tempfile.mkdtemp(prefix="test_no_skills_")

        try:
            # 创建没有 skills 的 ACS
            acs = {
                "aic": "test-001",
                "name": "无技能智能体",
                "description": "这是一个没有定义技能的测试智能体。",
                "skills": [],
            }

            with open(os.path.join(temp_dir, "acs.json"), "w") as f:
                json.dump(acs, f)
            with open(os.path.join(temp_dir, "config.toml"), "wb") as f:
                tomli_w.dump(test_agent_config, f)
            with open(os.path.join(temp_dir, "prompts.toml"), "wb") as f:
                tomli_w.dump(test_agent_prompts, f)

            from partners.generic_runner import GenericRunner

            with patch("partners.generic_runner.AsyncOpenAI"):
                runner = GenericRunner("test_agent", temp_dir)

                responsibilities = runner._build_responsibilities_prompt()

                assert acs["description"] in responsibilities
                assert "Skills" not in responsibilities

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
