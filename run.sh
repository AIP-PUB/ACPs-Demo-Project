#!/usr/bin/env bash
set -euo pipefail

# =============================
# run.sh — 统一服务管理脚本
#
# 用法:
#   ./run.sh start     [partner|leader|web|all]   # 启动服务（若已运行则报错）
#   ./run.sh stop      [partner|leader|web|all]   # 停止服务（默认 all）
#   ./run.sh restart   [partner|leader|web|all]   # 重启服务（先停后启，默认 all）
#   ./run.sh status                               # 查看所有服务状态
#   ./run.sh kill-port <port> [port2 ...]          # 按端口号杀死监听进程（PID文件丢失时的恢复手段）
#
# 端口由各服务自身配置决定（partners/config.toml, leader/config.toml, webserver.py），
# 本脚本不指定端口，启动后通过 lsof 发现进程实际监听的端口。
#
# 重复执行 start 等同于 restart：会先停掉同名旧进程再启动新进程。
# 每个进程日志输出到 logs/<name>.log，PID 记录在 logs/<name>.pid。
# =============================

ACTION="${1:-}"
SCOPE="${2:-all}"

# 验证参数
case "${ACTION}" in
  start|stop|restart|status|kill-port) ;;
  *)
    echo "用法: $0 {start|stop|restart|status|kill-port} [partner|leader|web|all|<port>...]"
    echo ""
    echo "  start     [scope]          启动服务（若已运行则报错，默认 all）"
    echo "  stop      [scope]          停止服务（默认 all）"
    echo "  restart   [scope]          重启服务（先停后启，默认 all）"
    echo "  status                     查看所有服务状态"
    echo "  kill-port <port> [port2 ...] 按端口号杀死监听进程（PID文件丢失时的恢复手段）"
    echo ""
    echo "  scope: partner | leader | web | all"
    exit 1
    ;;
esac

if [[ "$ACTION" != "status" && "$ACTION" != "kill-port" ]]; then
  case "$SCOPE" in
    partner|leader|web|all) ;;
    *)
      echo "无效的服务范围: ${SCOPE}（可选: partner | leader | web | all）"
      exit 1
      ;;
  esac
fi

# ---------------------------------------------------------------------------
# 基础设置
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# 服务注册表: "服务名  启动目标  [额外PYTHONPATH子目录]"
#   启动目标为点分模块名 (如 partners.main) → python -m 执行
#   启动目标以 .py 结尾 (如 web_app/webserver.py) → python 脚本执行
#   PID/日志文件自动以服务名命名: logs/<服务名>.pid, logs/<服务名>.log
SVC_PARTNER="partners_base  partners.main"
SVC_LEADER="leader_base  leader.main  leader"
SVC_WEB="static_web  web_app/webserver.py"

ALL_SERVICES=("$SVC_PARTNER" "$SVC_LEADER" "$SVC_WEB")

# 各服务的已知端口（用于 kill-port 无参数时的缺省模式）
# partner 的端口为各子进程端口；主进程不监听端口，杀子进程后会自动退出
PORTS_PARTNER="59221 59222 59223 59224 59225"
PORTS_LEADER="59210"
PORTS_WEB="59200"

if [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
  VENV_DIR="$SCRIPT_DIR/.venv"
elif [ -x "$SCRIPT_DIR/venv/bin/python" ]; then
  VENV_DIR="$SCRIPT_DIR/venv"
else
  echo "ERROR: 未找到虚拟环境 (.venv 或 venv)。"
  echo "请先创建虚拟环境并通过 Poetry 安装依赖。例如:"
  echo "  python3.13 -m venv .venv"
  echo "  source .venv/bin/activate"
  echo "  pip install poetry"
  echo "  poetry install"
  exit 1
fi
PYTHON_BIN="$VENV_DIR/bin/python"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-15}"

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

is_pid_alive() {
  local pid="$1"
  [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null
}

# 获取进程（含子进程）监听的所有端口，逗号分隔
# 优先使用 ss（Linux），回退到 lsof（macOS/通用）
get_listen_ports() {
  local pid="$1"
  local all_pids="$pid"
  local child_pids
  child_pids=$(pgrep -P "$pid" 2>/dev/null || true)
  if [ -n "${child_pids:-}" ]; then
    child_pids=$(echo "$child_pids" | tr '\n' ',' | sed 's/,$//')
    all_pids="$pid,$child_pids"
  fi

  if command -v ss >/dev/null 2>&1; then
    local p
    for p in $(echo "$all_pids" | tr ',' ' '); do
      ss -tlnp 2>/dev/null | awk -v pid="$p" '
        $0 ~ "pid="pid"[^0-9]" || $0 ~ "pid="pid"$" {
          n = split($4, a, ":"); print a[n]
        }'
    done | sort -un | tr '\n' ',' | sed 's/,$//'
    return 0
  elif command -v lsof >/dev/null 2>&1; then
    lsof -a -p "$all_pids" -iTCP -sTCP:LISTEN -n -P 2>/dev/null \
      | awk 'NR>1 { split($9, a, ":"); print a[length(a)] }' \
      | sort -un \
      | tr '\n' ',' \
      | sed 's/,$//' \
      || true
    return 0
  fi
  return 0
}

# 按 scope 筛选服务定义（每行输出一条服务定义）
resolve_services() {
  local scope="$1"
  case "$scope" in
    partner) printf '%s\n' "$SVC_PARTNER" ;;
    leader)  printf '%s\n' "$SVC_LEADER" ;;
    web)     printf '%s\n' "$SVC_WEB" ;;
    all)     printf '%s\n' "${ALL_SERVICES[@]}" ;;
  esac
}

# ---------------------------------------------------------------------------
# stop 操作
# ---------------------------------------------------------------------------

do_stop() {
  local scope="$1"
  echo "=== 停止服务: $scope ==="

  local ok_count=0 fail_count=0
  declare -a active_names=()
  declare -a active_pids=()
  declare -a active_pid_files=()

  while IFS= read -r svc_line; do
    [ -z "$svc_line" ] && continue
    # shellcheck disable=SC2086
    set -- $svc_line
    local name="$1"

    local pid_file="$LOG_DIR/${name}.pid"
    [ -f "$pid_file" ] || continue

    local pid
    pid=$(cat "$pid_file" 2>/dev/null || true)

    if [ -z "${pid:-}" ] || ! [[ "$pid" =~ ^[0-9]+$ ]]; then
      echo "[WARN] $name: PID 文件无效（内容: '${pid:-}' ），清理"
      rm -f "$pid_file"
      ok_count=$((ok_count+1))
      continue
    fi

    if ! is_pid_alive "$pid"; then
      echo "[INFO] $name: PID:$pid 未运行，清理 PID 文件"
      rm -f "$pid_file"
      ok_count=$((ok_count+1))
      continue
    fi

    echo "发送 SIGTERM -> $name (PID:$pid)"
    kill "$pid" 2>/dev/null || true
    active_names+=("$name")
    active_pids+=("$pid")
    active_pid_files+=("$pid_file")
  done < <(resolve_services "$scope")

  if [ ${#active_names[@]} -eq 0 ]; then
    if [ $ok_count -eq 0 ]; then
      echo "未发现运行中的服务。"
    fi
    return 0
  fi

  # 批量等待退出
  echo "等待进程退出..."
  local waited=0
  while [ $waited -lt 10 ]; do
    local all_stopped=true
    for pid in "${active_pids[@]}"; do
      if is_pid_alive "$pid"; then
        all_stopped=false
        break
      fi
    done
    $all_stopped && break
    sleep 1
    waited=$((waited+1))
  done

  # 逐个检查，必要时 SIGKILL
  for i in "${!active_names[@]}"; do
    local name="${active_names[$i]}"
    local pid="${active_pids[$i]}"
    local pid_file="${active_pid_files[$i]}"

    if is_pid_alive "$pid"; then
      echo "[WARN] $name (PID:$pid) 未响应 SIGTERM，发送 SIGKILL"
      kill -9 "$pid" 2>/dev/null || true
      sleep 0.5
    fi

    if is_pid_alive "$pid"; then
      echo "[FAIL] 无法停止 $name (PID:$pid)"
      fail_count=$((fail_count+1))
    else
      echo "[OK] $name (PID:$pid) 已停止"
      rm -f "$pid_file"
      ok_count=$((ok_count+1))
    fi
  done

  printf "\n停止结果: OK=%s, FAIL=%s\n" "$ok_count" "$fail_count"
  [ $fail_count -eq 0 ]
}

# ---------------------------------------------------------------------------
# start 操作
# ---------------------------------------------------------------------------

# 停止单个已有同名服务（仅 restart 调用）
stop_one() {
  local name="$1"
  local pid_file="$LOG_DIR/${name}.pid"
  [ -f "$pid_file" ] || return 0
  local old_pid
  old_pid=$(cat "$pid_file" 2>/dev/null || true)
  if [ -z "${old_pid:-}" ] || ! is_pid_alive "$old_pid"; then
    rm -f "$pid_file"
    return 0
  fi

  echo "检测到 $name 旧进程 (PID:$old_pid) 仍在运行，正在停止..."
  kill "$old_pid" 2>/dev/null || true
  local waited=0
  while is_pid_alive "$old_pid" && [ $waited -lt 5 ]; do
    sleep 1
    waited=$((waited+1))
  done
  if is_pid_alive "$old_pid"; then
    echo "$name (PID:$old_pid) 未在 5s 内退出，发送 SIGKILL"
    kill -9 "$old_pid" 2>/dev/null || true
    sleep 1
  fi
  rm -f "$pid_file"
  echo "$name 旧进程已停止"
}

# 检查服务是否已在运行，若是则报错退出（start 模式使用）
check_not_running() {
  local name="$1"
  local pid_file="$LOG_DIR/${name}.pid"
  [ -f "$pid_file" ] || return 0
  local old_pid
  old_pid=$(cat "$pid_file" 2>/dev/null || true)
  if [ -n "${old_pid:-}" ] && is_pid_alive "$old_pid"; then
    echo "错误: $name 已在运行 (PID:$old_pid)。若要重启请使用: $0 restart"
    exit 1
  fi
  # PID 文件存在但进程已死，清理
  rm -f "$pid_file"
}

launch() {
  local name="$1"
  local target="$2"
  local extra_path="${3:-}"
  local log_file="$LOG_DIR/${name}.log"

  if [ "${RESTART_MODE:-}" = true ]; then
    stop_one "$name"
  else
    check_not_running "$name"
  fi

  local pythonpath="$SCRIPT_DIR"
  if [ -n "$extra_path" ]; then
    pythonpath="$SCRIPT_DIR/$extra_path:$SCRIPT_DIR"
  fi

  if [[ "$target" == *.py ]]; then
    # 脚本模式
    if [ ! -f "$target" ]; then
      echo "WARN: 未找到 ${target}，跳过 $name 启动"
      return
    fi
    (PYTHONPATH="$pythonpath" nohup "$PYTHON_BIN" -u "$target" \
        >"$log_file" 2>&1 & echo $! >"$LOG_DIR/${name}.pid")
  else
    # 模块模式: python -m
    (PYTHONPATH="$pythonpath" nohup "$PYTHON_BIN" -m "$target" \
        >"$log_file" 2>&1 & echo $! >"$LOG_DIR/${name}.pid")
  fi

  if [ -f "$LOG_DIR/${name}.pid" ]; then
    local pid
    pid=$(cat "$LOG_DIR/${name}.pid")
    echo "启动 $name ($target), PID: $pid, 日志: $log_file"
  fi
}

# 轮询等待单个服务就绪：进程存活且端口已监听
# 返回: 0=OK, 1=FAIL(进程死亡), 2=WARN(超时但存活)
wait_for_ready() {
  local name="$1"
  local pid_file="$LOG_DIR/${name}.pid"
  local pid=""
  [ -f "$pid_file" ] && pid=$(cat "$pid_file" 2>/dev/null || true)

  if [ -z "${pid:-}" ]; then
    echo "[FAIL] $name: 未找到 PID 文件"
    return 1
  fi

  local waited=0
  while [ $waited -lt "$STARTUP_TIMEOUT" ]; do
    # 进程已死 → 立即报错
    if ! is_pid_alive "$pid"; then
      echo "[FAIL] $name: 进程 PID:$pid 启动失败（请查看 $LOG_DIR/${name}.log）"
      return 1
    fi
    # 检测到端口 → 成功
    local ports
    ports=$(get_listen_ports "$pid")
    if [ -n "${ports:-}" ]; then
      echo "[OK] $name: PID:$pid, 端口: $ports ($((waited))s)"
      return 0
    fi
    sleep 1
    waited=$((waited+1))
  done

  # 超时但进程还活着
  echo "[WARN] $name: PID:$pid 存活，但 ${STARTUP_TIMEOUT}s 内未检测到端口（lsof 不可用或启动较慢）"
  return 0
}

do_start() {
  local scope="$1"
  echo "=== 启动服务: $scope ==="

  declare -a svc_entries=()
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    svc_entries+=("$line")
  done < <(resolve_services "$scope")

  for entry in "${svc_entries[@]+"${svc_entries[@]}"}"; do
    # shellcheck disable=SC2086
    launch $entry
    sleep 0.2
  done

  echo "等待服务就绪（超时 ${STARTUP_TIMEOUT}s）..."

  local ok_count=0 fail_count=0

  for entry in "${svc_entries[@]+"${svc_entries[@]}"}"; do
    # shellcheck disable=SC2086
    set -- $entry
    local local_name="$1"
    if wait_for_ready "$local_name"; then
      ok_count=$((ok_count+1))
    else
      fail_count=$((fail_count+1))
    fi
  done

  printf "\n启动完成: OK=%s, FAIL=%s\n" "$ok_count" "$fail_count"
  if [ $fail_count -gt 0 ]; then
    echo "有服务未成功启动，详见上述 [FAIL] 日志以及对应日志文件。"
  fi
}

# ---------------------------------------------------------------------------
# status 操作
# ---------------------------------------------------------------------------

do_status() {
  echo "=== 服务状态 ==="
  local found=false
  for svc in "${ALL_SERVICES[@]}"; do
    # shellcheck disable=SC2086
    set -- $svc
    local name="$1"
    local pid_file="$LOG_DIR/${name}.pid"
    if [ ! -f "$pid_file" ]; then
      echo "[--] $name: 未启动（无 PID 文件）"
      continue
    fi
    found=true
    local pid
    pid=$(cat "$pid_file" 2>/dev/null || true)

    if [ -z "${pid:-}" ] || ! [[ "$pid" =~ ^[0-9]+$ ]]; then
      echo "[--] $name: PID 文件无效（内容: '${pid:-}' ）"
      continue
    fi

    if ! is_pid_alive "$pid"; then
      echo "[STOP] $name: PID:$pid 未存活"
      continue
    fi

    local ports
    ports=$(get_listen_ports "$pid")
    if [ -n "${ports:-}" ]; then
      echo "[RUN] $name: PID:$pid, 端口: $ports"
    else
      echo "[RUN] $name: PID:${pid}（端口未知）"
    fi
  done
}

# ---------------------------------------------------------------------------
# kill-port 操作
# ---------------------------------------------------------------------------

# 按端口号查找监听进程的 PID（优先 ss，回退 lsof）
find_pids_by_port() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -tlnp 2>/dev/null \
      | awk -v port="$port" '{
          n = split($4, a, ":");
          if (a[n] == port) {
            gsub(/.*pid=/, "", $0); gsub(/[^0-9].*/, "", $0);
            if ($0 ~ /^[0-9]+$/) print $0
          }
        }' | sort -un || true
  elif command -v lsof >/dev/null 2>&1; then
    lsof -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null | sort -un || true
  fi
}

do_kill_port() {
  shift  # 跳过 ACTION 参数

  # 无参数时使用注册表中的所有端口
  # shellcheck disable=SC2206
  local port_list=()
  if [ $# -eq 0 ]; then
    port_list=($PORTS_WEB $PORTS_LEADER $PORTS_PARTNER)
    echo "未指定端口，使用注册表端口: ${port_list[*]}"
  else
    port_list=("$@")
  fi

  local ok_count=0 fail_count=0
  # 收集所有被杀的 PID，用于后续清理主进程
  declare -a killed_pids=()

  for port in "${port_list[@]}"; do
    # 校验端口号格式
    if ! [[ "$port" =~ ^[0-9]+$ ]] || [ "$port" -lt 1 ] || [ "$port" -gt 65535 ]; then
      echo "[ERROR] 无效端口号: ${port}（需为 1-65535 的整数）"
      fail_count=$((fail_count+1))
      continue
    fi

    echo "--- 端口 $port ---"
    # 查找监听该端口的进程
    local pids
    pids=$(find_pids_by_port "$port")

    if [ -z "${pids:-}" ]; then
      echo "[INFO] 端口 $port 上未发现监听进程"
      ok_count=$((ok_count+1))
      continue
    fi

    for pid in $pids; do
      local cmd_info
      cmd_info=$(ps -p "$pid" -o pid=,args= 2>/dev/null || echo "$pid (信息不可用)")
      echo "发现进程: $cmd_info"
      echo "发送 SIGTERM -> PID:$pid"
      kill "$pid" 2>/dev/null || true
      killed_pids+=("$pid")
    done

    # 等待进程退出
    local waited=0
    while [ $waited -lt 5 ]; do
      local still_alive=false
      for pid in $pids; do
        if is_pid_alive "$pid"; then
          still_alive=true
          break
        fi
      done
      $still_alive || break
      sleep 1
      waited=$((waited+1))
    done

    # 检查并强杀
    local port_ok=true
    for pid in $pids; do
      if is_pid_alive "$pid"; then
        echo "[WARN] PID:$pid 未响应 SIGTERM，发送 SIGKILL"
        kill -9 "$pid" 2>/dev/null || true
        sleep 0.5
        if is_pid_alive "$pid"; then
          echo "[FAIL] 无法杀死 PID:$pid"
          port_ok=false
        else
          echo "[OK] PID:$pid 已强制停止"
        fi
      else
        echo "[OK] PID:$pid 已停止"
      fi
    done

    if $port_ok; then
      ok_count=$((ok_count+1))
    else
      fail_count=$((fail_count+1))
    fi
  done

  # --- 清理不监听端口的主进程（如 partners.main 监控进程） ---
  # partner 主进程本身不监听端口，仅监控子进程；子进程被杀后主进程通常在 ~1s 内
  # 通过健康检查自动退出，但这里做一次兜底清理以确保可靠。
  if [ ${#killed_pids[@]} -gt 0 ]; then
    echo ""
    echo "等待主进程自动退出..."
    sleep 2

    local straggler_pids
    straggler_pids=$(pgrep -f "python.*-m (partners|leader)\.main" 2>/dev/null || true)
    if [ -n "${straggler_pids:-}" ]; then
      echo "检测到残留主进程，正在清理:"
      for spid in $straggler_pids; do
        local scmd
        scmd=$(ps -p "$spid" -o pid=,args= 2>/dev/null || echo "$spid")
        echo "  发送 SIGTERM -> $scmd"
        kill "$spid" 2>/dev/null || true
      done
      sleep 2
      for spid in $straggler_pids; do
        if is_pid_alive "$spid"; then
          echo "  [WARN] PID:$spid 未退出，发送 SIGKILL"
          kill -9 "$spid" 2>/dev/null || true
        fi
      done
      echo "主进程清理完成"
    else
      echo "无残留主进程"
    fi
  fi

  printf "\n结果: OK=%s, FAIL=%s\n" "$ok_count" "$fail_count"
  [ $fail_count -eq 0 ]
}

# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

case "$ACTION" in
  start)     do_start "$SCOPE" ;;
  stop)      do_stop "$SCOPE" ;;
  restart)   RESTART_MODE=true; do_start "$SCOPE" ;;
  status)    do_status ;;
  kill-port) do_kill_port "$@" ;;
esac
