"""
Partner Agent 基础工具函数

提供 SSL 上下文构建、Agent 发现与端口校验等基础能力，
供 main.py 和测试使用。
"""

import os
import ssl
import logging
import multiprocessing
import tomllib
from typing import Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("partners.utils")

ONLINE_DIR = os.path.join(os.path.dirname(__file__), "online")
CONFIG_FILENAME = "config.toml"


# ---------------------------------------------------------------------------
# mTLS / SSL
# ---------------------------------------------------------------------------


def build_ssl_context(agent_path: str, server_cfg: dict) -> Optional[ssl.SSLContext]:
    """
    根据 config.toml 中 [server] 和 [server.mtls] 的配置构建 SSL 上下文。

    返回 None 表示使用纯 HTTP。
    """
    mtls_cfg = server_cfg.get("mtls", {})
    tls_enabled = mtls_cfg.get("tls_enabled", False)

    if not tls_enabled:
        return None

    def resolve(p: str) -> str:
        if os.path.isabs(p):
            return p
        return os.path.join(agent_path, p)

    cert_file = resolve(mtls_cfg["cert_file"])
    key_file = resolve(mtls_cfg["key_file"])
    ca_file = resolve(mtls_cfg["ca_file"])

    for f, desc in [
        (cert_file, "cert_file"),
        (key_file, "key_file"),
        (ca_file, "ca_file"),
    ]:
        if not os.path.isfile(f):
            raise FileNotFoundError(f"mTLS {desc} not found: {f}")

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile=cert_file, keyfile=key_file)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2

    verify_client = mtls_cfg.get("verify_client", False)
    if verify_client:
        ctx.load_verify_locations(cafile=ca_file)
        ctx.verify_mode = ssl.CERT_REQUIRED
    else:
        ctx.verify_mode = ssl.CERT_NONE  # noqa: S504

    return ctx


def build_uvicorn_ssl_kwargs(agent_path: str, server_cfg: dict) -> dict:
    """
    根据 config.toml 中 [server.mtls] 的配置，返回 uvicorn.run() 所需的 SSL 关键字参数。

    返回空字典表示使用纯 HTTP。
    """
    mtls_cfg = server_cfg.get("mtls", {})
    tls_enabled = mtls_cfg.get("tls_enabled", False)

    if not tls_enabled:
        return {}

    def resolve(p: str) -> str:
        if os.path.isabs(p):
            return p
        return os.path.join(agent_path, p)

    cert_file = resolve(mtls_cfg["cert_file"])
    key_file = resolve(mtls_cfg["key_file"])
    ca_file = resolve(mtls_cfg["ca_file"])

    for f, desc in [
        (cert_file, "cert_file"),
        (key_file, "key_file"),
        (ca_file, "ca_file"),
    ]:
        if not os.path.isfile(f):
            raise FileNotFoundError(f"mTLS {desc} not found: {f}")

    kwargs = {
        "ssl_certfile": cert_file,
        "ssl_keyfile": key_file,
        "ssl_ca_certs": ca_file,
    }

    verify_client = mtls_cfg.get("verify_client", False)
    if verify_client:
        kwargs["ssl_cert_reqs"] = ssl.CERT_REQUIRED
    else:
        kwargs["ssl_cert_reqs"] = ssl.CERT_NONE

    return kwargs


# ---------------------------------------------------------------------------
# Agent 发现与端口校验
# ---------------------------------------------------------------------------


def discover_agents(filter_names: Optional[list] = None) -> Dict[str, str]:
    """扫描 online 目录，返回 {agent_name: agent_path} 字典。"""
    agents = {}
    if not os.path.exists(ONLINE_DIR):
        os.makedirs(ONLINE_DIR, exist_ok=True)
        return agents

    for name in sorted(os.listdir(ONLINE_DIR)):
        agent_path = os.path.join(ONLINE_DIR, name)
        if not os.path.isdir(agent_path):
            continue
        if not os.path.exists(os.path.join(agent_path, "acs.json")):
            logger.warning(f"Skipping {name}: acs.json missing")
            continue
        if not os.path.exists(os.path.join(agent_path, CONFIG_FILENAME)):
            logger.warning(f"Skipping {name}: {CONFIG_FILENAME} missing")
            continue
        if filter_names and name not in filter_names:
            continue
        agents[name] = agent_path

    return agents


def read_agent_port(agent_path: str) -> int:
    """从 agent 的 config.toml 读取端口号。"""
    config_path = os.path.join(agent_path, CONFIG_FILENAME)
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)
    return cfg.get("server", {}).get("port", 59221)


def validate_ports(agents: Dict[str, str]) -> Dict[int, str]:
    """验证所有 agent 端口无冲突，返回 {port: agent_name} 映射。"""
    port_map: Dict[int, str] = {}
    for name, path in agents.items():
        port = read_agent_port(path)
        if port in port_map:
            raise ValueError(
                f"Port conflict: {name} and {port_map[port]} both use port {port}"
            )
        port_map[port] = name
    return port_map


# ---------------------------------------------------------------------------
# 进程管理
# ---------------------------------------------------------------------------


def terminate_processes(processes: Dict[str, multiprocessing.Process]):
    """优雅终止所有子进程。"""
    for name, p in processes.items():
        if p.is_alive():
            logger.info(f"[{name}] Terminating (PID: {p.pid})")
            p.terminate()
    for name, p in processes.items():
        p.join(timeout=5)
        if p.is_alive():
            logger.warning(f"[{name}] Force killing (PID: {p.pid})")
            p.kill()


def check_process_health(processes: Dict[str, multiprocessing.Process], shutdown_fn):
    """检查所有子进程状态，若任一退出则触发关闭。"""
    import sys

    for name, p in processes.items():
        if not p.is_alive():
            logger.warning(f"[{name}] Process exited (exit code: {p.exitcode})")
            shutdown_fn()
            sys.exit(1)
