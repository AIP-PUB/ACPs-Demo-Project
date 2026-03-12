#!/usr/bin/env python3
"""批量更新集成测试文件以适应新 API 结构。"""

import re
import os

# 需要更新的文件列表
test_files = [
    "test_planning_flow.py",
    "test_scenario_switch.py",
    "test_history_compression.py",
    "test_history_compression_flow.py",
    "test_session_ttl.py",
    "test_task_new_flow.py",
    "test_idempotency_flow.py",
    "test_aggregator_flow.py",
    "test_async_execution_flow.py",
    "test_input_router_flow.py",
    "test_clarification_flow.py",
    "test_full_flow_e2e.py",
]


def update_file(filepath):
    """更新单个测试文件。"""
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return False

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    original = content

    # 1. 添加导入
    if "from .conftest import" not in content:
        content = content.replace(
            'pytest_plugins = ("pytest_asyncio",)',
            'from .conftest import build_submit_request, extract_session_id, is_success_response\n\npytest_plugins = ("pytest_asyncio",)',
        )

    # 2. 替换简单的 json={"query": "xxx"} 格式
    content = re.sub(
        r'json=\{"query": "([^"]+)"\}',
        r'json=build_submit_request(query="\1")',
        content,
    )

    # 3. 替换带 sessionId 的格式（多种变体）
    # json={"sessionId": session_id, "query": "xxx"}
    content = re.sub(
        r'json=\{\s*"sessionId":\s*session_id,\s*"query":\s*"([^"]+)",?\s*\}',
        r'json=build_submit_request(query="\1", session_id=session_id)',
        content,
    )

    # json={"query": "xxx", "sessionId": session_id}
    content = re.sub(
        r'json=\{\s*"query":\s*"([^"]+)",\s*"sessionId":\s*session_id,?\s*\}',
        r'json=build_submit_request(query="\1", session_id=session_id)',
        content,
    )

    # 4. 替换响应提取 r1.json()["sessionId"]
    content = re.sub(
        r'(\w+)\.json\(\)\["sessionId"\]', r"extract_session_id(\1.json())", content
    )

    # 5. 替换 response.json()["sessionId"]
    content = re.sub(
        r'response\.json\(\)\["sessionId"\]',
        r"extract_session_id(response.json())",
        content,
    )

    # 6. 替换 data["sessionId"]
    content = re.sub(
        r'data\["sessionId"\]',
        r'extract_session_id(data) if "result" in data else data.get("sessionId")',
        content,
    )

    if content != original:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Updated: {filepath}")
        return True
    else:
        print(f"No changes: {filepath}")
        return False


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    updated = 0
    for filename in test_files:
        filepath = os.path.join(base_dir, filename)
        if update_file(filepath):
            updated += 1

    print(f"\nTotal updated: {updated}/{len(test_files)}")


if __name__ == "__main__":
    main()
