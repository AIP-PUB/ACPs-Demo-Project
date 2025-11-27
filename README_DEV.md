# 开发教程

本文档基于 `demo-apps` 示例工程，总结了如何按照 ACPs 体系开发 **Partner Agent** 与 **Leader Agent**。阅读顺序建议：先了解公共组件，再动手实现 Partner，最后扩展 Leader 的编排逻辑。

---

# 1. 开发准备

- **运行环境**：Python 3.10+、FastAPI、httpx、OpenAI SDK。建议使用 `python3 -m venv venv && source venv/bin/activate` 创建虚拟环境。
- **依赖安装**：`pip install -r requirements.txt`
- **环境变量**：复制 `.env.example` 为 `.env`，填入以下关键项：
  - `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL`
  - `DISCOVERY_BASE_URL`（Leader 动态发现北京各 Partner 使用）
  - 各 Partner 的 `*_AGENT_ID`、`*_AIP_ENDPOINT`、`*_LOG_LEVEL`（保留默认即可）
- **证书准备**：`certs/` 目录下已给出示例 mTLS 证书。自定义部署时请替换成生产证书，并在对应 `*.json` 中更新路径。
- **启动脚本**：`./start.sh` 会按顺序启动所有 Partner、Leader 以及本地静态 Web。开发时可以单独运行某个 Agent，例如 `python3 beijing_urban/beijing_urban.py`。

---

# 2. ACPs AIP 公共组件

项目将与 AIP 协议相关的共性逻辑集中在 `acps_aip/` 目录，关键模块如下：

- `aip_base_model.py`：定义 `Task`、`TaskState`、`Message`、`Product`、`DataItem` 等核心模型。参照 [AIP-DESIGN.md](../acps-design/AIP-DESIGN.md) 可了解完整字段。
- `aip_rpc_server.py`：封装 Partner 侧 JSON-RPC 服务端，提供 `add_aip_rpc_router()`、`TaskManager`、`CommandHandlers` 等工具，帮助快速实现 `Start/Continue/Cancel/Complete` 处理流程。
- `aip_rpc_client.py`：Leader 侧的 JSON-RPC 客户端，包装了 `start_task()`、`continue_task()`、`complete_task()` 等异步调用方法。
- `mtls_config.py`：加载 mTLS 配置 JSON，生成 `ssl.SSLContext`。Partner 与 Leader 的互信依赖该配置。

通用辅助函数见 `base.py`：

- `get_agent_logger()` 标准化日志输出（含结构化字段）。
- `truncate()` 用于日志打印长文本。
- `call_openai_chat()` 针对 OpenAI Chat Completion 做了重试和统一参数处理。
- `load_capabilities_snippet_from_json()`/`extract_text_from_message()` 等便于处理 ACS 描述与消息体。

在开发新 Agent 前，可直接复用这些组件，避免重复实现协议或通用逻辑。

---

# 3. 实现 Partner Agent

Partner 以 FastAPI + JSON-RPC 形式提供服务，核心流程可以参考 `beijing_urban/`、`beijing_rural/`、`beijing_catering/`、`china_transport/`、`china_hotel/` 五个示例。以下以 `beijing_urban` 为例讲解关键步骤。

## 3.1 目录结构

```
beijing_urban/
├── beijing_urban.py        # 主入口（FastAPI 应用 + AIP CommandHandlers）
├── beijing_urban.json      # ACS 能力描述
└── __init__.py
```

## 3.2 FastAPI 启动骨架

```python
app = FastAPI(title="北京城区景点规划师 Agent", ...)
add_aip_rpc_router(app, AIP_ENDPOINT, handlers)

if __name__ == "__main__":
	 uvicorn.run(..., ssl_keyfile=..., ssl_certfile=..., ssl_ca_certs=...)
```

`handlers` 由 `CommandHandlers` 构建，包含 `on_start`、`on_continue`、`on_cancel`。AIP 服务端会自动解析 JSON-RPC 请求并回调这些函数。

## 3.3 Start/Continue 流程

1. **Start**：

   - 解析 `Message` 文本，调用 `decide_accept()` 做门卫判断。
   - 若范围不匹配（示例中城六区以外请求），直接创建 `TaskState.Rejected`。
   - 若通过门卫，`TaskManager.create_task(..., initial_state=TaskState.Accepted)`，并挂载等待超时参数，随后委派后台流水线执行（`UrbanJobManager.start_job()`）。

2. **后台流水线 `_run_urban_pipeline()`**：

   - 先将任务置为 `Working`。
   - `analyze_requirements()` 调用 LLM 生成结构化需求。若 `decision=reject` 或 `missingFields` 非空，则更新为 `AwaitingInput`，并记录提示文案。
   - 信息完整后，`produce_plan()` 生成行程文本，写入 `Product`，任务进入 `AwaitingCompletion`。
   - 通过 `schedule_await_timeout()` 设定 `AwaitingInput`/`AwaitingCompletion` 的超时行为。

3. **Continue**：

   - 仅当任务处于 `AwaitingInput` 或 `AwaitingCompletion` 时处理。
   - 重新调度流水线，合并补充信息与前一次 `requirements`，重复上面步骤。

4. **Cancel**：
   - 取消后台任务，交给 `DefaultHandlers.cancel()` 做收尾。

## 3.4 LLM 提示词

每个 Partner 都将提示模板放在模块顶端：

- `DECIDE_PROMPT`：门卫层，用于判断请求是否属于本 Agent。
- `ANALYZE_PROMPT`：需求结构化，返回 `decision`、`requirements`、`missingFields`。
- `PRODUCE_PROMPT`：生成最终产出文本。

开发新的 Partner 时，可以复用现有的流水线，参考这三类提示词与兜底逻辑。

## 3.5 常见扩展点

- **自定义字段校验**：在 `analyze_requirements()` 后检查特定字段，如预算范围、交通方式等。
- **多轮交互记忆**：`TaskManager` 会把 `Message` 存入历史，可在继续阶段复用。
- **产出格式**：`Product` 中可以放结构化 JSON 或富文本，只需选择合适的 `DataItem` 类型。

---

# 4. 实现 Leader Agent

`tour_assistant/tour_assistant.py` 提供了完整的 Leader 示例。关键职责：

1. **环境初始化**：加载 `.env`，设置 `LEADER_AGENT_ID`、`OPENAI_*`，并准备日志、mTLS、FastAPI 应用。
2. **动态发现**：
   - `DISCOVERY_BASE_URL` 指向 discovery-server（见 `../discovery-server` 工程）。
   - `find_agent_service()` 对北京的 3 个 Partner 先调用 REST 发现接口，失败时回退到本地 JSON。
3. **会话管理**：`sessions` 字典缓存最近 5 条消息、partner task 状态、待回答问题等。
4. **多段流程**：
   - `llm_analysis()`：将用户请求拆解为 5 个维度（交通/酒店/城区/郊区/美食），生成 `dimensions`、`required_agents`、`task_priority` 等结构化结果。若 LLM 失效则使用启发式兜底。
   - `call_partner()`：根据维度调用 `AipRpcClient.start_task()`，轮询任务状态，处理 `AwaitingInput`/`AwaitingCompletion`。
   - `continue_partner()` / `complete_partner()`：在用户补充或复查时继续推进打开的任务。
   - `_evaluate_product_satisfaction()`：使用 LLM 快速评估 Partner 产出是否满足需求，为 `AwaitingCompletion` 提供自动反馈。
   - `llm_integrate()`：整合各 Partner 结果，生成面向用户的最终回答。
5. **API 接口**：
   - `POST /user_api`：统一入口。根据是否存在待补充任务，决定是发起新需求还是继续任务。
   - `GET /`：健康检查。

## 4.1 新增 Partner 的接入步骤

1. 将 Partner ACS 描述部署到 discovery-server 或本地 JSON。
2. 在 `llm_analysis()` 的 `mapping` 与 `ANALYSIS_PROMPT_TEMPLATE` 中补充维度名称与可用 Agent。
3. 在 `call_partner()` / `continue_partner()` 的 lookup 映射中加入新 Agent ID。
4. 根据 Partner 的业务，调整 `llm_integrate()` 的输出模板或格式。

---

# 5. 开发新 Agent 的建议流程

1. **复制模板**：以现有 Partner 目录为模板，替换 `PROMPT`、日志标签、产出格式。
2. **编写 ACS**：完成 `*.json` 文件，包括 AIC、endPoints、skills。
3. **注册认证**：使用 ACS 的 json 文件完成注册、审批和认证，确保证书与实际部署一致。
4. **本地调试**：`python3 <agent>/<agent>.py` 启动，使用 `acps_aip/aip_rpc_client.py` 或 Leader 进行测试。
5. **接入 Leader**：更新 Leader 的发现/路由映射，确保 `ANALYSIS_PROMPT_TEMPLATE` 能触发该 Agent。
6. **补充测试**：在 `tests/` 中新增场景，验证多轮交互是否满足预期。
7. **部署上线**：配置 discovery-server 与证书，Leader 即可自动发现新的 Agent。

通过阅读与实践以上内容，即可快速搭建符合 ACPs 标准的多智能体协同系统，并在此基础上扩展更多垂直领域的专业 Agent。
