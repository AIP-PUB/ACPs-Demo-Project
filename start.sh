#!/usr/bin/env bash
set -euo pipefail

# =============================
# start.sh
# 启动 5 个 Partner Agents + 1 Leader (tour_assistant)
# 每个进程日志输出到 logs/<name>.log
# 启动成功后脚本直接退出，停止由 stop.sh 负责
# =============================

# 进入脚本所在目录（保证相对路径一致）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# 可配置端口（与各模块 Python 文件中的 uvicorn.run() 保持一致）
# 各服务启动配置（包括 mTLS、reload 等）在各自的 Python 文件 __main__ 块中定义
# 当前端口分配：
#   beijing_urban.py          8011
#   beijing_rural.py          8012
#   beijing_catering.py       8013
#   china_hotel.py            8015
#   china_transport.py        8016
#   tour_assistant.py         8019

# 格式: "目录名 端口 服务名"
NAMES=(
  "beijing_urban 8011 beijing_urban"
  "beijing_rural 8012 beijing_rural"
  "beijing_catering 8013 beijing_catering"
  "china_hotel 8015 china_hotel"
  "china_transport 8016 china_transport"
  "tour_assistant 8019 tour_assistant"
)

# 额外：静态前端 web 服务器 (web_app/webserver.py) 
WEB_STATIC_SCRIPT="web_app/webserver.py"
WEB_STATIC_PORT="3000"
WEB_STATIC_HOST="0.0.0.0"

VENV_DIR="$SCRIPT_DIR/venv"
VENV_PYTHON="$VENV_DIR/bin/python"
if [ ! -x "$VENV_PYTHON" ]; then
  echo "ERROR: 未找到虚拟环境的 Python 可执行文件: $VENV_PYTHON"
  echo "请在脚本目录下创建 venv，并安装依赖后重试。例如:"
  echo "  python3 -m venv venv && ./venv/bin/pip install -r requirements.txt"
  exit 1
fi
PYTHON_BIN="$VENV_PYTHON"
STARTUP_WAIT_SECONDS="${STARTUP_WAIT_SECONDS:-5}"

# 端口占用检查：优先使用 lsof，其次 nc，最后用 Python socket 探测
is_port_in_use() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    lsof -iTCP:"$port" -sTCP:LISTEN -n -P >/dev/null 2>&1
    return $?
  elif command -v nc >/dev/null 2>&1; then
    nc -z 127.0.0.1 "$port" >/dev/null 2>&1
    return $?
  else
    "$PYTHON_BIN" - "$port" >/dev/null 2>&1 <<'PY'
import socket, sys
port = int(sys.argv[1])
s = socket.socket()
s.settimeout(0.3)
try:
    in_use = (s.connect_ex(('127.0.0.1', port)) == 0)
finally:
    s.close()
sys.exit(0 if in_use else 1)
PY
    return $?
  fi
}

# 等待端口释放，超时返回 1
wait_for_port_free() {
  local port="$1"
  local timeout="${2:-8}"
  local waited=0
  while is_port_in_use "$port" && [ $waited -lt $timeout ]; do
    sleep 1
    waited=$((waited+1))
  done
  if is_port_in_use "$port"; then
    return 1
  else
    return 0
  fi
}

# 根据 pid 文件尝试关闭进程（先 TERM 后 KILL），成功返回 0
kill_from_pidfile() {
  local name="$1"
  local pid_file="$LOG_DIR/${name}.pid"
  if [ ! -f "$pid_file" ]; then
    echo "未找到 PID 文件: $pid_file"
    return 1
  fi
  local pid
  pid=$(cat "$pid_file" 2>/dev/null || true)
  if [ -z "${pid:-}" ]; then
    echo "PID 文件为空: $pid_file"
    return 1
  fi

  if kill -0 "$pid" 2>/dev/null; then
    echo "发送 SIGTERM -> $name ($pid)"
    kill "$pid" 2>/dev/null || true
    local timeout=5
    local waited=0
    while kill -0 "$pid" 2>/dev/null && [ $waited -lt $timeout ]; do
      sleep 1
      waited=$((waited+1))
    done
    if kill -0 "$pid" 2>/dev/null; then
      echo "$name ($pid) 未在 ${timeout}s 内退出，发送 SIGKILL"
      kill -9 "$pid" 2>/dev/null || true
      sleep 1
    fi
  fi

  if kill -0 "$pid" 2>/dev/null; then
    echo "无法杀掉进程 $name ($pid)"
    return 1
  else
    rm -f "$pid_file" || true
    echo "$name 已停止"
    return 0
  fi
}

# 检查 PID 是否存活
is_pid_alive() {
  local pid="$1"
  if [ -z "${pid:-}" ]; then
    return 1
  fi
  kill -0 "$pid" 2>/dev/null
}

# 校验服务：优先以端口监听为准；若端口未监听但 PID 存活则提示可能在启动中
verify_service() {
  local name="$1"
  local port="$2"
  local pid_file="$LOG_DIR/${name}.pid"
  local pid=""
  if [ -f "$pid_file" ]; then
    pid=$(cat "$pid_file" 2>/dev/null || true)
  fi

  if is_port_in_use "$port"; then
    if [ -n "$pid" ] && is_pid_alive "$pid"; then
      echo "[OK] $name: 端口:$port 正在监听, PID:$pid 存活"
    else
      echo "[OK] $name: 端口:$port 正在监听 (PID 未知)"
    fi
    return 0
  else
    if [ -n "$pid" ] && is_pid_alive "$pid"; then
      echo "[WARN] $name: PID:$pid 存活，但端口:$port 未监听（可能仍在启动或异常）"
      return 1
    else
      echo "[FAIL] $name: 未启动（无 PID 且端口未监听）"
      return 1
    fi
  fi
}

launch() {
  local dir_name="$1"    # e.g. beijing_urban
  local port="$2"        # e.g. 8011
  local name="$3"        # e.g. beijing_urban
  local log_file="$LOG_DIR/${name}.log"
  local py_file="${dir_name}/${name}.py"
  
  # 启动前检查端口是否占用；若占用，尝试通过 pid 文件杀掉旧进程
  if is_port_in_use "$port"; then
    echo "端口 $port 已被占用。尝试通过 PID 文件关闭 $name ..."
    if kill_from_pidfile "$name"; then
      if ! wait_for_port_free "$port" 8; then
        echo "端口 $port 仍被占用，跳过启动 $name。"
        return 0
      fi
    else
      echo "无法通过 PID 文件关闭 $name，跳过启动。"
      return 0
    fi
  fi
  
  # 直接运行 Python 文件（使用文件内部的 uvicorn 配置，包括 mTLS 等参数）
  if [ -f "$py_file" ]; then
    (nohup "$PYTHON_BIN" "$py_file" \
        >"$log_file" 2>&1 & echo $! >"$LOG_DIR/${name}.pid")
    pid=$(cat "$LOG_DIR/${name}.pid")
    echo "启动 $name , Port: $port, PID: $pid, 日志: $log_file"
  else
    echo "ERROR: 未找到 Python 文件: $py_file"
    return 1
  fi
}

# 启动所有服务
for entry in "${NAMES[@]}"; do
  # shellcheck disable=SC2086
  launch $entry
  sleep 0.2  # 稍作间隔，避免端口竞争日志混杂
done

# 启动静态 web server
if [ -f "$WEB_STATIC_SCRIPT" ]; then
  if is_port_in_use "$WEB_STATIC_PORT"; then
    echo "static_web 端口 $WEB_STATIC_PORT 已被占用。尝试通过 PID 文件关闭 static_web ..."
    if kill_from_pidfile "static_web"; then
      if ! wait_for_port_free "$WEB_STATIC_PORT" 8; then
        echo "端口 $WEB_STATIC_PORT 仍被占用，跳过 static_web 启动。"
      else
        (
          nohup "$PYTHON_BIN" "$WEB_STATIC_SCRIPT" --host "$WEB_STATIC_HOST" --port "$WEB_STATIC_PORT" \
            >"$LOG_DIR/static_web.log" 2>&1 & echo $! >"$LOG_DIR/static_web.pid"
        )
      fi
    else
      echo "无法通过 PID 文件关闭 static_web，跳过 static_web 启动。"
    fi
  else
    (
      nohup "$PYTHON_BIN" "$WEB_STATIC_SCRIPT" --host "$WEB_STATIC_HOST" --port "$WEB_STATIC_PORT" \
        >"$LOG_DIR/static_web.log" 2>&1 & echo $! >"$LOG_DIR/static_web.pid"
    )
  fi
else
  echo "WARN: 未找到 $WEB_STATIC_SCRIPT，跳过静态前端服务器启动"
fi
if [ -f "$LOG_DIR/static_web.pid" ]; then
  pid=$(cat "$LOG_DIR/static_web.pid")
  echo "启动 static_web, Port: $WEB_STATIC_PORT, PID: $pid, 日志: $LOG_DIR/static_web.log"
fi

# 启动后等待一段时间再进行校验
echo "等待 ${STARTUP_WAIT_SECONDS}s 以便服务完成启动..."
sleep "$STARTUP_WAIT_SECONDS"

printf "\n启动校验:\n"
ok_count=0
fail_count=0
for entry in "${NAMES[@]}"; do
  # 解析: dir_name port name
  set -- $entry
  local_dir_name="$1"; local_port="$2"; local_name="$3"
  if verify_service "$local_name" "$local_port"; then
    ok_count=$((ok_count+1))
  else
    fail_count=$((fail_count+1))
  fi
done

# 校验静态 web（若存在脚本）
if [ -f "$WEB_STATIC_SCRIPT" ]; then
  if verify_service "static_web" "$WEB_STATIC_PORT"; then
    ok_count=$((ok_count+1))
  else
    fail_count=$((fail_count+1))
  fi
fi

printf "\n启动校验完成: OK=%s, FAIL=%s\n" "$ok_count" "$fail_count"
if [ $fail_count -gt 0 ]; then
  echo "有服务未成功启动，详见上述 [WARN]/[FAIL] 日志以及对应日志文件。"
fi

if [ -f "$LOG_DIR/static_web.pid" ]; then
  echo "前端页面可访问: http://$WEB_STATIC_HOST:$WEB_STATIC_PORT/"
fi
echo "服务启动流程结束，若需停止请运行 ./stop.sh"
