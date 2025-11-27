本项目是一个基于 ACPs 架构的示例应用程序，通过对北京旅游的场景模拟，展示了如何使用 ACPs 架构进行多 Agent 协同工作。

# 1. 各个 Agent 的角色

1. 旅游助理。负责接收和回应用户请求，通过自然语言与多个 Agent 进行交互。它是用户与系统之间的桥梁，能够理解用户的意图并协调各个 Agent 的工作。它在 ACPs 协议中扮演 Leader 角色。
2. 北京城区景点规划师。负责根据旅游助理提供的需求和偏好，制定个性化的旅游路线和景点推荐。它需要综合考虑交通、时间和用户兴趣等因素，为用户提供最佳的旅游方案。它只支持北京城六区的请求。城区的范围定义是：东城区、西城区、朝阳区、海淀区、丰台区和石景山区，其它区县定义为郊区。它明确拒绝北京郊区和其它城市的景点推荐请求，明确拒绝北京的美食推荐请求。如果是关于交通的请求，它只提供北京城区的交通建议，明确拒绝北京郊区以及进出北京的交通建议。它在 ACPs 协议中扮演 Partner 角色。
3. 北京郊区景点规划师。负责根据旅游助理提供的需求和偏好，制定个性化的郊区旅游路线和景点推荐。它需要考虑郊区的自然景观和人文景点，为用户提供最佳的郊区旅游方案。它只支持北京郊区的请求。它明确拒绝北京城区和其它城市的景点推荐请求，明确拒绝北京的美食推荐请求。如果是关于交通的请求，它只提供北京郊区的交通建议，明确拒绝北京城区以及进出北京的交通建议。它在 ACPs 协议中扮演 Partner 角色。
4. 北京美食推荐师。负责根据旅游助理提供的的口味偏好和旅游路线，推荐北京的特色美食。它需要了解北京的美食文化和各类餐厅信息，为用户提供丰富的美食选择。它提供北京全境的美食推荐服务，包括城区和郊区。如果请求中包含交通信息，它可以提供交通信息周边的美食推荐，而拒绝提供交通建议。它明确拒绝北京的景点推荐请求，明确拒绝进出北京的交通建议。它在 ACPs 协议中扮演 Partner 角色。
5. 全国交通预定师。负责根据旅游助理提供的旅游日程，规划最佳的全国范围的不同城市之间的交通方式和路线，并协助完成预定。它需要考虑交通工具、时间和费用等因素，为用户提供便捷的出行方案。它可以提供进出城市的市内交通接驳的建议，比如机场或高铁站与市内某个地址之间的交通建议，但明确拒绝与接驳无关的城市内部的交通建议。它明确拒绝城市的景点和美食推荐请求。它在 ACPs 协议中扮演 Partner 角色。
6. 全国酒店预订师。负责根据旅游助理提供的旅游日程和预算，推荐合适的酒店，并协助用户完成预订。它需要了解用户目的地城市的酒店信息和用户的住宿需求，为用户提供舒适的住宿选择。它可以提供全国范围内的酒店推荐和预订服务。它明确拒绝城市的景点、美食和交通建议请求。它在 ACPs 协议中扮演 Partner 角色。

# 2. ACPs 协议族的实践

ACPs 协议族是一个用于智能体协作的协议集合，包含了智能体身份码（AIC）、智能体能力描述（ACS）、智能体可信注册（ATR）、智能体身份认证过程（AIA）、智能体发现流程（ADP）和智能体交互协议（AIP）等规范。它们定义了智能体之间的身份、能力、注册、认证、发现和交互方式。

本项目中，旅游助理使用 AIP 协议与各个专业 Agent 进行对话和数据交换。旅游助理 Agent 通过自然语言与用户交互，使用大模型来理解用户的需求，然后分解用户需求。Leader 会通过发现服务查询有哪些 Agent 可用，并通过各个 Agent 的能力描述（ACS）理解它们的能力范围和工作方式，然后把分解后的任务发送给各个 Partner 进行工作。最后，Leader 会收集各个 Partner 的结果，整合成一个完整的回复，返回给用户。Leader 和 Partner 之间的通信都遵循 AIP 协议，并且会涉及到多轮对话。

# 3. 目录结构和文件说明

```
demo-apps/
├── base.py                 # 公共工具函数（日志、字符串处理等）
├── requirements.txt        # Python 依赖列表
├── start.sh                # 启动各 Agent/服务的脚本示例
├── acps_aip/               # AIP 协议交互的基础设施 & RPC 客户端/服务端实现
├── tour_assistant/         # 旅游助理Leader服务
├── beijing_urban/          # 北京城区景点 Partner（基础实现示例）
├── beijing_rural/          # 北京郊区景点 Partner（TODO： SSE示例）
├── beijing_catering/       # 北京美食 Partner（TODO：notification示例）
├── china_transport/        # 全国交通 Partner（多个Skills选择示例）
├── china_hotel/            # 全国酒店 Partner（多个Skills选择示例）
├── certs/                  # mTLS 所需的示例证书与密钥
├── tests/                  # Pytest 测试用例（单测 & E2E）
├── web_app/                # 前端静态站点（含 webserver）
└── logs/                   # 运行时 PID 文件与日志输出目录
```

# 4. 技术概要说明

本项目包含基于 Python 的各个 Agent 的实现，和以 Javascript 为主的 WebApp 的实现。

## 4.1. 基于 Python 的各个 Agent

### 技术栈

- **Web 框架**: FastAPI
- **数据验证与解析** Pydantic V2（避免使用 V1 版本的风格）
- **大模型集成**: OpenAI API

## 4.2. 基于 JavaScript 的 WebApp

### 技术栈

当前实现采取“零依赖”策略：不引入构建工具与第三方运行时库，方便快速浏览与嵌入。

- **框架**: 原生 JavaScript (ES2020)
- **样式**: 轻量手写 CSS，无 Tailwind
- **HTTP**: 原生 `fetch`
- **Markdown 渲染**: 仅做标题/列表的简单替换（后续可换用 marked.js）
- **构建**: 无（直接静态服务）

如果后续需要增强体验，可逐步引入：Tailwind、marked.js、Prism.js、组件化框架 (React/Vue) 等。

# 4. 快速开始

**创建虚拟环境并安装依赖**

```bash
python3.18 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**准备环境变量：**

```bash
cp .env.example .env
# 根据实际部署修改 openai/discovery 相关配置
```

**启动全部 Agent 与演示服务（首次运行可先确认证书路径等）：**

```bash
./start.sh
```

**在浏览器验证 Leader：**

用浏览器访问 `http://localhost:3000`。

**查看运行日志：**

- 每个子服务在 `logs/` 下记录日志，可用 `tail -f logs/tour_assistant.log` 等命令查看。
- 若需查看所有服务的日志，可运行 `tail -f logs/*.log`。

# 5. 注册与认证

每个 Agent 在接入 ACPs 网络之前都要拿到两个成果：注册成功后分配的唯一 AIC 编码，以及认证完成后颁发的证书与私钥。本章节给出一个最小可运行的教程，帮助你从零完成注册与认证。

## 5.1. 注册

**目标**：将新的 Agent 注册到`registry-server`，并在审批通过后拿到 AIC。

### 5.1.1. 启动注册服务

1. 打开一个新终端进入项目根目录。
2. 切换到注册服务目录并启动 API：

```bash
cd registry-server
python3.18 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

3. 保持该终端运行，`registry-server`会监听 `http://localhost:8001`。

### 5.1.2. 使用注册脚本

1. 在新的终端中切换到注册服务目录并激活虚拟环境：

```bash
cd registry-server
source venv/bin/activate
```

2. 先确保演示用账号可用（如已存在会直接登录）：

```bash
python demo_register.py ensure-accounts
```

3. 使用示例 ACS 文件`../demo-apps/beijing_urban/beijing_urban.json`提交注册并自动进入待审状态：

```bash
python demo_register.py register --acs-path ../demo-apps/beijing_urban/beijing_urban.json
```

    - 终端会返回 Agent 的内部 `id` 以及 `approval_status=PENDING`。
    - 如服务地址非默认 `http://localhost:8001/api`，可通过`--base-url`覆盖。

### 5.1.3. 审批并获取 AIC

1. 保持当前终端（仍在`registry-server`虚拟环境中），以管理员身份审批刚刚提交的 Agent：

```bash
python demo_register.py approve --acs-path ../demo-apps/beijing_urban/beijing_urban.json
```

2. 审批成功后，脚本会打印分配好的 `aic`，并将该编号写回同一个 ACS 文件，便于后续认证阶段使用。

### 5.1.4. （可选）删除演示 Agent

若需要重复练习或清理测试数据，可执行：

```bash
python demo_register.py delete --acs-path ../demo-apps/beijing_urban/beijing_urban.json
```

删除完成后可重新执行注册 → 审批流程。

## 5.2. 认证

**目标**：让新注册的 Agent 获取 mTLS 所需的证书与私钥。

### 5.2.1. 启动认证服务与挑战服务

1. 确认注册服务仍在运行。
2. 在新终端中启动认证服务：

```bash
cd ca-server
python3.18 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

3. 再打开一个终端启动挑战服务（模拟 ACME HTTP-01）：

```bash
cd ca-client
./challenge-server.sh
```

4. 认证服务监听 `http://localhost:8003`，挑战服务监听 `http://localhost:8004`，保持两者运行直至认证完成。

### 5.2.2. 运行认证脚本

1. 在`ca-client`目录启用脚本：

```bash
cd ca-client
./acme-client.sh new-cert --agent-id AIC --config ./acme-client.conf
```

2. 将`AIC`替换为注册阶段记录的 AIC 编码。
3. 命令会走完 ACME 账户创建、挑战验证与证书签发流程，最终生成：
   - 证书文件：`ca-client/certs/AIC.crt`
   - 私钥文件：`ca-client/private/AIC.key`

### 5.2.3. 完成配置

将上述证书/私钥复制到目标 Agent 的`certs/`目录中完成配置。

```bash
cp ca-client/certs/AIC.crt demo-apps/certs/
cp ca-client/private/AIC.key demo-apps/certs/
```

此外，还需要将根证书`ca-server/certs/ca.crt`复制到目标 Agent 的`certs/ca.crt`，以便启用 mTLS。

```bash
cp ca-server/certs/ca.crt demo-apps/certs/
```

这样，目标 Agent 就可以通过 mTLS 与其他 Agent 安全通信了。
