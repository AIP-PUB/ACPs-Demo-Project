# ACPs Demo Apps

本项目是基于 ACPs 协议族的示例应用程序，通过北京旅游场景模拟，展示多 Agent 基于 AIP（智能体交互协议）进行协同工作的完整流程。

## 1. 架构概览

系统由一个 **Leader Agent** 和多个 **Partner Agent** 协作组成，通过 Web 前端与用户交互：

- **Leader（智能助理）**：接收用户请求，通过大模型理解意图、拆解任务，协调各 Partner 完成工作后整合结果返回用户。采用"通用底座 + 场景插件"架构，场景行为完全由配置驱动。
- **Partner（专业智能体）**：各自负责特定领域的专业任务。采用"通用运行时 + 配置驱动"架构，新增 Agent 只需编写配置文件。
- **Web App**：基于原生 JavaScript 的前端界面，通过轮询机制与 Leader 交互。

### 当前 Partner 角色

| Agent           | 角色               | 服务范围                                             |
| --------------- | ------------------ | ---------------------------------------------------- |
| beijing_urban   | 北京城区景点规划师 | 东城、西城、朝阳、海淀、丰台、石景山区景点及城区交通 |
| beijing_rural   | 北京郊区景点规划师 | 北京郊区景点及郊区交通                               |
| beijing_food    | 北京美食推荐师     | 北京全境美食推荐                                     |
| china_transport | 全国交通预定师     | 全国范围城市间交通规划及市内接驳                     |
| china_hotel     | 全国酒店预订师     | 全国范围酒店推荐与预订                               |

### 通信模式

支持两种 AIP 通信模式：

- **Direct RPC**：Leader 与 Partner 点对点交互，适用于小规模、低延迟场景。
- **Group**：通过 RabbitMQ 消息队列进行群组协作，群内消息全局可见。

## 2. 目录结构

```
demo-apps/
├── leader/              # Leader Agent（旅游助理）
│   ├── main.py          # 应用入口（FastAPI）
│   ├── config.toml      # 平台级系统配置（LLM、端口、发现服务等）
│   ├── acs.json         # Leader 自身的 ACS 定义
│   ├── assistant/       # 平台核心代码（编排器、Session、意图分析、LLM 封装等）
│   └── scenario/        # 场景配置集合（base 基础人设 + expert 专业场景插件）
├── partners/            # Partner Agent 通用框架
│   ├── main.py          # 主入口：扫描 online/ 目录，启动 FastAPI 服务
│   ├── generic_runner.py # 通用运行时（三阶段处理逻辑）
│   ├── group_handler.py # Group 模式消息队列处理
│   ├── online/          # 活跃 Agent 配置（beijing_urban, beijing_food 等）
│   └── offline/         # 编辑区（开发/暂停的 Agent）
├── web_app/             # 前端 Web 界面
│   ├── index.html       # 主页面
│   ├── app.js           # 前端逻辑（API 调用、轮询、消息渲染）
│   ├── config.js        # 运行时配置（后端地址、轮询间隔等）
│   ├── styles.css       # 样式表
│   └── webserver.py     # 静态文件服务器（Python 标准库）
├── logs/                # 运行时日志和 PID 文件
├── pyproject.toml       # Python 项目配置及依赖
└── run.sh               # 服务管理脚本 (start/stop/restart/status)
```

各子目录的详细说明：

- [leader/README.md](leader/README.md) — Leader Agent 平台结构、配置与 API
- [partners/README.md](partners/README.md) — Partner Agent 框架、三阶段处理流程与新增 Agent 方法
- [web_app/README.md](web_app/README.md) — 前端界面与配置

## 3. 环境准备

### 前置条件

- **Python 3.13+**
- **Poetry**（包管理工具）
- **RabbitMQ**（仅 Group 模式需要）
- **OpenAI 兼容的 LLM API**（如 OpenAI、豆包、通义千问等）

### 安装依赖

```bash
cd demo-apps
python3.13 -m venv venv
source venv/bin/activate
pip install poetry
poetry install
```

> `poetry install` 会自动安装 `pyproject.toml` 中定义的所有依赖，包括本地路径依赖 `acps-sdk`。

### 配置

1. **Leader 配置**：

   ```bash
   cd leader
   cp config.example.toml config.toml
   ```

   编辑 `config.toml`，配置 LLM API Key 和 Base URL（`[llm.*]` 段）。

2. **Partner 配置**：

   各 Partner 目录（`partners/online/<agent_name>/`）下均有 `config.toml`，需配置对应的 LLM 参数。可参考 `config.example.toml`。

3. **Web App 配置**（可选）：

   如果 Leader 不在默认地址 `http://127.0.0.1:59210`，编辑 `web_app/config.js` 修改 `backendBase`。

## 4. 运行

### 一键启动所有服务

```bash
./run.sh start          # 启动 Partner + Leader + Web 前端
```

### 分步启动

```bash
./run.sh start partner  # 仅启动 Partner 服务
./run.sh start leader   # 仅启动 Leader 服务（需 Partner 已启动）
./run.sh start web      # 仅启动 Web 前端
```

### 其他操作

```bash
./run.sh stop           # 停止所有服务
./run.sh restart        # 重启所有服务
./run.sh status         # 查看服务状态和端口
```

### 停止服务

```bash
./run.sh stop           # 停止所有服务
./run.sh stop partner   # 仅停止 Partner
./run.sh stop leader    # 仅停止 Leader
```

## 5. 验证

1. 启动所有服务后，浏览器访问 `http://localhost:59200`。
2. 在输入框中输入旅游请求，例如："帮我规划一个北京三日游"。
3. 观察 Leader 协调各 Partner 的工作过程：意图分析 → 任务编排 → 并行调度 → 结果整合。
4. 如果 Agent 需要补充信息（反问），界面会显示问题，用户回答后继续推进。

### 查看日志

```bash
tail -f logs/*.log              # 查看所有服务日志
tail -f logs/leader_base.log    # 仅查看 Leader 日志
tail -f logs/partners_base.log  # 仅查看 Partner 日志
```

## 6. 证书管理脚本

根目录下的 `manage-certs.sh` 用于统一管理 Leader 和所有在线 Partner 的 mTLS 证书。脚本会读取各 Agent 的 `acs.json` 中的 `aic` 字段，调用外部 `ca-client` 申请或续签证书，并将生成的证书文件写入对应目录。

支持的操作包括：

- `./manage-certs.sh new`：为全部 Agent 申请新证书
- `./manage-certs.sh renew`：为全部 Agent 续签证书
- `./manage-certs.sh trust-bundle`：更新并分发 `trust-bundle.pem`
- `./manage-certs.sh new leader`：仅为指定 Agent 操作

其中，Leader 的证书输出到 `leader/atr/`，Partner 的证书输出到各自目录。对于 Partner，脚本还会自动更新其 `config.toml` 中 `[server.mtls]` 段的证书路径。各 Agent 的处理彼此独立，单个 Agent 失败不会阻塞其他 Agent，脚本结束后会输出汇总结果。
