#!/usr/bin/env bash
set -euo pipefail

# =============================
# stop.sh
# 根据 logs 目录中的 PID 文件停止各服务并清理 PID
# =============================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="logs"
if [ ! -d "$LOG_DIR" ]; then
  echo "未找到 logs 目录，无需停止。"
  exit 0
fi

pid_files=("$LOG_DIR"/*.pid)
if [ ! -e "${pid_files[0]}" ]; then
  echo "logs 目录中未发现 PID 文件，无需停止。"
  exit 0
fi

ok_count=0
fail_count=0
declare -a active_names=()
declare -a active_pids=()
declare -a active_pid_files=()

for pid_file in "${pid_files[@]}"; do
  name="$(basename "$pid_file" .pid)"
  pid=$(cat "$pid_file" 2>/dev/null || true)

  if [ -z "${pid:-}" ]; then
    echo "[WARN] $name: PID 文件为空，清理该文件"
    rm -f "$pid_file" || true
    ok_count=$((ok_count+1))
    continue
  fi

  if ! kill -0 "$pid" 2>/dev/null; then
    echo "[INFO] $name: PID $pid 未运行，清理 PID 文件"
    rm -f "$pid_file" || true
    ok_count=$((ok_count+1))
    continue
  fi

  echo "发送 SIGTERM -> $name ($pid)"
  kill "$pid" 2>/dev/null || true
  active_names+=("$name")
  active_pids+=("$pid")
  active_pid_files+=("$pid_file")
done

echo

active_count=${#active_names[@]}
wait_seconds=${STOP_WAIT_SECONDS:-2}
if [ "$active_count" -gt 0 ]; then
  echo "已向 $active_count 个进程发送 SIGTERM，等待 ${wait_seconds}s..."
  if [ "$wait_seconds" -gt 0 ]; then
    sleep "$wait_seconds"
  fi
fi

echo

for idx in "${!active_names[@]}"; do
  name="${active_names[$idx]}"
  pid="${active_pids[$idx]}"
  pid_file="${active_pid_files[$idx]}"

  if kill -0 "$pid" 2>/dev/null; then
    echo "$name: 仍在运行，发送 SIGKILL"
    kill -9 "$pid" 2>/dev/null || true
    sleep 1
  fi

  if kill -0 "$pid" 2>/dev/null; then
    echo "[FAIL] $name: 进程仍在运行 (PID:$pid)"
    fail_count=$((fail_count+1))
  else
    rm -f "$pid_file" || true
    echo "[OK] $name 已停止"
    ok_count=$((ok_count+1))
  fi
done

echo

printf "停止流程完成: OK=%s, FAIL=%s\n" "$ok_count" "$fail_count"
if [ $fail_count -gt 0 ]; then
  exit 1
fi
