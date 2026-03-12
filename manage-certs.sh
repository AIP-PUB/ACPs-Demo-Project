#!/usr/bin/env bash
#
# manage-certs.sh — 自动为所有 agent（leader + partners）申请 / 续签证书
#
# 用法:
#   ./manage-certs.sh new          # 为所有 agent 申请新证书
#   ./manage-certs.sh renew        # 为所有 agent 续签证书
#   ./manage-certs.sh trust-bundle # 仅更新 trust-bundle 并分发
#   ./manage-certs.sh new   beijing_food   # 只为指定 agent 操作
#   ./manage-certs.sh renew leader         # 只为 leader 续签
#
# 每个 agent 的操作互不阻塞，出错只记录，不影响其余 agent。
# 结束后会输出汇总报告。
#
set -uo pipefail

# ---------- 配置 ----------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CA_CLIENT_DIR="${CA_CLIENT_DIR:-$(cd "$SCRIPT_DIR/../ca-client" && pwd)}"
CA_CLIENT="${CA_CLIENT_DIR}/venv/bin/ca-client"
CA_CLIENT_CONF="${CA_CLIENT_DIR}/.ca-client.conf"

PARTNERS_DIR="$SCRIPT_DIR/partners/online"
LEADER_DIR="$SCRIPT_DIR/leader"
LEADER_ACS_JSON="$LEADER_DIR/atr/acs.json"

# ca-client 默认输出目录（仅用于 trust-bundle 子命令分发）
CA_CERTS_DIR="$CA_CLIENT_DIR/certs"

# ---------- 颜色 ----------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# ---------- 计数器 ----------
SUCCESS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0
declare -a FAIL_AGENTS=()

# ---------- 工具函数 ----------
log_info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

extract_aic() {
    local acs_json="$1"
    python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['aic'])" "$acs_json" 2>/dev/null
}

usage() {
    echo "用法: $0 <new|renew|trust-bundle> [agent_name ...]"
    echo ""
    echo "  new           为 agent 申请新证书"
    echo "  renew         为 agent 续签证书"
    echo "  trust-bundle  仅更新 trust-bundle 并分发到各 agent"
    echo ""
    echo "  不指定 agent_name 则处理所有 agent（leader + 所有 partners）"
    echo "  agent_name 可以是 'leader' 或 partners/online/ 下的目录名"
    exit 1
}

# ---------- 证书操作：单个 agent ----------
# process_agent <action> <agent_name> <acs_json_path> <cert_dest_dir>
process_agent() {
    local action="$1"
    local agent_name="$2"
    local acs_json="$3"
    local cert_dest="$4"

    log_info "--- [$agent_name] 开始处理 (action=$action) ---"

    # 1) 提取 AIC
    local aic
    aic=$(extract_aic "$acs_json")
    if [[ -z "$aic" ]]; then
        log_error "[$agent_name] 无法从 $acs_json 中提取 AIC，跳过"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAIL_AGENTS+=("$agent_name(无法提取AIC)")
        return 1
    fi
    log_info "[$agent_name] AIC = $aic"

    # 2) 确定输出路径
    mkdir -p "$cert_dest"
    local dest_cert="$cert_dest/${aic}.pem"
    local dest_key="$cert_dest/${aic}.key"
    local dest_trust="$cert_dest/trust-bundle.pem"

    # 3) 调用 ca-client，通过 --cert-path / --key-path / --trust-bundle-path 直接输出到目标目录
    local ca_cmd
    if [[ "$action" == "new" ]]; then
        ca_cmd=("$CA_CLIENT" -c "$CA_CLIENT_CONF" new-cert --aic "$aic"
                --cert-path "$dest_cert" --key-path "$dest_key" --trust-bundle-path "$dest_trust")
    elif [[ "$action" == "renew" ]]; then
        ca_cmd=("$CA_CLIENT" -c "$CA_CLIENT_CONF" renew-cert --aic "$aic"
                --cert-path "$dest_cert" --key-path "$dest_key" --trust-bundle-path "$dest_trust")
    else
        log_error "[$agent_name] 未知 action: $action"
        return 1
    fi

    log_info "[$agent_name] 执行: ${ca_cmd[*]}"
    # 必须在 ca-client 目录执行，因为配置使用相对路径
    local output
    if ! output=$(cd "$CA_CLIENT_DIR" && "${ca_cmd[@]}" 2>&1); then
        log_error "[$agent_name] ca-client 失败:"
        echo "$output" | sed 's/^/    /'
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAIL_AGENTS+=("$agent_name(ca-client失败)")
        return 1
    fi
    log_info "[$agent_name] ca-client 成功"
    echo "$output" | sed 's/^/    /'

    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    log_info "[$agent_name] 证书已输出到 $cert_dest ✓"
}

# ---------- 更新 trust-bundle 并分发 ----------
distribute_trust_bundle() {
    log_info "=== 更新 trust-bundle ==="

    # 收集所有 agent 的 trust-bundle 目标路径
    local dest_paths=()
    mkdir -p "$LEADER_DIR/atr"
    dest_paths+=("$LEADER_DIR/atr/trust-bundle.pem")
    for partner_dir in "$PARTNERS_DIR"/*/; do
        [[ -d "$partner_dir" ]] || continue
        dest_paths+=("${partner_dir}trust-bundle.pem")
    done

    # 用第一个路径调用 update-trust-bundle，再复制到其余路径
    local first_dest="${dest_paths[0]}"
    local output
    if ! output=$(cd "$CA_CLIENT_DIR" && "$CA_CLIENT" -c "$CA_CLIENT_CONF" update-trust-bundle 2>&1); then
        log_error "update-trust-bundle 失败:"
        echo "$output" | sed 's/^/    /'
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAIL_AGENTS+=("trust-bundle(更新失败)")
        return 1
    fi
    log_info "trust-bundle 已更新"
    echo "$output" | sed 's/^/    /'

    # 从默认位置分发到各 agent
    local src_trust="$CA_CERTS_DIR/trust-bundle.pem"
    if [[ ! -f "$src_trust" ]]; then
        log_error "trust-bundle.pem 不存在: $src_trust"
        return 1
    fi

    for dest in "${dest_paths[@]}"; do
        cp "$src_trust" "$dest" && \
            log_info "已分发: $dest"
    done

    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
}

# ---------- 更新 partner config.toml 中的证书路径 ----------
update_partner_config() {
    local agent_name="$1"
    local aic="$2"
    local config_file="$PARTNERS_DIR/$agent_name/config.toml"

    [[ -f "$config_file" ]] || return 0

    # 检查 config.toml 中的 cert_file 是否已经指向正确的 AIC
    local current_cert
    current_cert=$(grep 'cert_file' "$config_file" | head -1 | sed 's/.*= *"\(.*\)"/\1/')

    if [[ "$current_cert" == "${aic}.pem" ]]; then
        return 0  # 已经是正确路径
    fi

    log_info "[$agent_name] 更新 config.toml 证书路径..."
    # 使用 python 精确替换 [server.mtls] 下的路径
    python3 - "$config_file" "$aic" <<'PYEOF'
import sys, re

config_file = sys.argv[1]
aic = sys.argv[2]

with open(config_file, 'r') as f:
    content = f.read()

# 替换 [server.mtls] 段下的 cert_file / key_file / ca_file
in_mtls = False
lines = content.split('\n')
new_lines = []
for line in lines:
    stripped = line.strip()
    if stripped == '[server.mtls]':
        in_mtls = True
    elif stripped.startswith('[') and stripped.endswith(']'):
        in_mtls = False
    if in_mtls:
        if stripped.startswith('cert_file'):
            line = f'cert_file = "{aic}.pem"'
        elif stripped.startswith('key_file'):
            line = f'key_file = "{aic}.key"'
        elif stripped.startswith('ca_file'):
            line = f'ca_file = "trust-bundle.pem"'
    new_lines.append(line)

with open(config_file, 'w') as f:
    f.write('\n'.join(new_lines))
PYEOF
    log_info "[$agent_name] config.toml 已更新"
}

# ========== 主流程 ==========

if [[ $# -lt 1 ]]; then
    usage
fi

ACTION="$1"
shift

if [[ "$ACTION" != "new" && "$ACTION" != "renew" && "$ACTION" != "trust-bundle" ]]; then
    log_error "未知操作: $ACTION"
    usage
fi

# 检查 ca-client 是否存在
if [[ ! -x "$CA_CLIENT" ]]; then
    log_error "找不到 ca-client: $CA_CLIENT"
    exit 1
fi

# trust-bundle 操作
if [[ "$ACTION" == "trust-bundle" ]]; then
    distribute_trust_bundle
    echo ""
    log_info "========== 汇总 =========="
    log_info "成功: $SUCCESS_COUNT  失败: $FAIL_COUNT"
    [[ ${#FAIL_AGENTS[@]} -gt 0 ]] && log_error "失败列表: ${FAIL_AGENTS[*]}"
    exit $FAIL_COUNT
fi

# 确定要处理的 agent 列表
AGENTS=("$@")
if [[ ${#AGENTS[@]} -eq 0 ]]; then
    # 未指定则处理所有
    AGENTS=("leader")
    for d in "$PARTNERS_DIR"/*/; do
        [[ -d "$d" ]] && AGENTS+=("$(basename "$d")")
    done
fi

log_info "========== 开始证书${ACTION} =========="
log_info "待处理 agent: ${AGENTS[*]}"
echo ""

for agent in "${AGENTS[@]}"; do
    if [[ "$agent" == "leader" ]]; then
        if [[ ! -f "$LEADER_ACS_JSON" ]]; then
            log_error "[leader] 找不到 $LEADER_ACS_JSON，跳过"
            SKIP_COUNT=$((SKIP_COUNT + 1))
            continue
        fi
        process_agent "$ACTION" "leader" "$LEADER_ACS_JSON" "$LEADER_DIR/atr"
    else
        partner_dir="$PARTNERS_DIR/$agent"
        partner_acs="$partner_dir/acs.json"
        if [[ ! -d "$partner_dir" ]]; then
            log_error "[$agent] 目录不存在: $partner_dir，跳过"
            SKIP_COUNT=$((SKIP_COUNT + 1))
            continue
        fi
        if [[ ! -f "$partner_acs" ]]; then
            log_error "[$agent] 找不到 $partner_acs，跳过"
            SKIP_COUNT=$((SKIP_COUNT + 1))
            continue
        fi
        process_agent "$ACTION" "$agent" "$partner_acs" "$partner_dir"

        # 更新 partner 的 config.toml 证书路径
        if [[ $? -eq 0 ]]; then
            local_aic=$(extract_aic "$partner_acs")
            [[ -n "$local_aic" ]] && update_partner_config "$agent" "$local_aic"
        fi
    fi
    echo ""
done


# ---------- 汇总报告 ----------
echo ""
log_info "================================================"
log_info "  汇总报告"
log_info "================================================"
log_info "  成功: $SUCCESS_COUNT"
[[ $FAIL_COUNT -gt 0 ]] && log_error "  失败: $FAIL_COUNT — ${FAIL_AGENTS[*]}"
[[ $SKIP_COUNT -gt 0 ]] && log_warn "  跳过: $SKIP_COUNT"
log_info "================================================"

exit $FAIL_COUNT
