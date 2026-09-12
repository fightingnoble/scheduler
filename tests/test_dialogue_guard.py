"""tests/test_dialogue_guard.py — dialogue_guard v1.3 专属测试（REQ-026/E0120-E0121 P6）

全部用 tmp 副本 + 环境变量覆盖路径，不触碰真实协议/HISTORY/state 文件，
不运行、不移动任何既有业务测试。
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GUARD = os.path.join(REPO, "cleanup", "tools", "dialogue_guard.py")

PROTOCOL = (
    "# 双 Agent 协作对话\n\n协议正文（测试样例）。\n\n## 事件记录\n\n"
)


def ev(n, actor, typ, state, nxt):
    return (
        "### E%04d | REQ-099 | %s | %s\n\n"
        "- state: `%s`\n"
        "- base_head: %s\n"
        "- paths: `AGENT_DIALOGUE.md`\n"
        "- summary: test event\n"
        "- evidence: N/A\n"
        "- next_writer: `%s`\n\n" % (n, actor, typ, state, "0" * 40, nxt)
    )


def run_guard(tmp, *args, **extra_env):
    env = dict(os.environ)
    env["AGENT_DIALOGUE_FILE"] = os.path.join(tmp, "AGENT_DIALOGUE.md")
    env["AGENT_DIALOGUE_HISTORY"] = os.path.join(tmp, "history", "AGENT_DIALOGUE_HISTORY.md")
    env["AGENT_DIALOGUE_ARCHIVE"] = os.path.join(tmp, "AGENT_DIALOGUE_archive.md")
    env.update(extra_env)
    return subprocess.run([sys.executable, GUARD, *args], capture_output=True, text=True, env=env)


@pytest.fixture()
def tmprepo():
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "history"), exist_ok=True)
        # legacy seed（不透明段：乱序 + 双 E0010 + 大写 Codex 头原样）
        seed = "# 归档\n\n" + ev(1, "Codex", "RESULT", "WAITING_REVIEW", "reviewer") + \
               ev(4, "reviewer", "REVIEW", "CHANGES_REQUESTED", "codex") + \
               ev(2, "codex", "PROPOSAL", "WAITING_REVIEW", "reviewer") + \
               ev(10, "reviewer", "RECOVERY", "CHANGES_REQUESTED", "codex") + \
               ev(10, "codex", "RESULT", "WAITING_REVIEW", "reviewer") + \
               ev(14, "reviewer", "REVIEW", "ACCEPTED", "codex")
        open(os.path.join(tmp, "AGENT_DIALOGUE_archive.md"), "w").write(seed)
        # 活动区：E0015..E0020 六条 → 归档 --through 19 后留 E0020
        blocks = [ev(15 + i, "codex" if i % 2 == 0 else "reviewer",
                     "PROPOSAL" if i % 2 == 0 else "REVIEW",
                     "WAITING_REVIEW" if i % 2 == 0 else "APPROVED",
                     "reviewer" if i % 2 == 0 else "codex") for i in range(6)]
        open(os.path.join(tmp, "AGENT_DIALOGUE.md"), "w").write(PROTOCOL + "".join(blocks))
        yield tmp


def test_archive_preserves_block_bytes(tmprepo):
    text = open(os.path.join(tmprepo, "AGENT_DIALOGUE.md")).read()
    pos = [m.start() for m in re.finditer(r"^### E\d{4} ", text, re.M)]
    pre = {}
    for i, p in enumerate(pos):
        m = re.match(r"^### (E\d{4}) ", text[p:p + 20])
        num = int(m.group(1)[1:])
        if 15 <= num <= 19:
            end = pos[i + 1] if i + 1 < len(pos) else len(text)
            pre[m.group(1)] = hashlib.sha256(text[p:end].encode()).hexdigest()
    r = run_guard(tmprepo, "archive", "--through", "E0019")
    assert r.returncode == 0, r.stdout + r.stderr
    h = open(os.path.join(tmprepo, "history", "AGENT_DIALOGUE_HISTORY.md")).read()
    hpos = [m.start() for m in re.finditer(r"^### E\d{4} ", h, re.M)]
    for i, p in enumerate(hpos):
        m = re.match(r"^### (E\d{4}) ", h[p:p + 20])
        if m.group(1) in pre:
            end = hpos[i + 1] if i + 1 < len(hpos) else len(h)
            assert hashlib.sha256(h[p:end].encode()).hexdigest() == pre[m.group(1)]
    # 活动区只剩 E0020；旧 archive 文件被移动移除
    root = open(os.path.join(tmprepo, "AGENT_DIALOGUE.md")).read()
    assert "### E0020" in root and "### E0015" not in root
    assert not os.path.exists(os.path.join(tmprepo, "AGENT_DIALOGUE_archive.md"))
    # post 建 state → check 通过
    assert run_guard(tmprepo, "post").returncode == 0
    assert run_guard(tmprepo, "check").returncode == 0


def test_merged_chain_and_legacy_opaque(tmprepo):
    assert run_guard(tmprepo, "archive", "--through", "E0019").returncode == 0
    r = run_guard(tmprepo, "check")  # legacy 乱序/双 E0010 不报错；边界 19→20 续接
    assert r.returncode == 0, r.stdout
    # 边界断裂：活动区首事件编号跳档 → check 报错
    root_p = os.path.join(tmprepo, "AGENT_DIALOGUE.md")
    text = open(root_p).read().replace("### E0020", "### E0022")
    open(root_p, "w").write(text)
    r = run_guard(tmprepo, "check")
    assert r.returncode != 0 and ("边界断裂" in r.stdout or "跳档" in r.stdout)


def test_active_cap_rejected(tmprepo):
    assert run_guard(tmprepo, "archive", "--through", "E0019").returncode == 0
    root_p = os.path.join(tmprepo, "AGENT_DIALOGUE.md")
    text = open(root_p).read()
    extra = "".join(ev(21 + i, "codex", "RESULT", "WAITING_REVIEW", "codex") for i in range(5))
    open(root_p, "w").write(text + extra)  # 活动区 6 条
    r = run_guard(tmprepo, "check")
    assert r.returncode != 0 and "超过上限" in r.stdout
    assert run_guard(tmprepo, "post").returncode != 0


def test_state_drift_detection(tmprepo):
    assert run_guard(tmprepo, "archive", "--through", "E0019").returncode == 0
    assert run_guard(tmprepo, "post").returncode == 0
    hist_p = os.path.join(tmprepo, "history", "AGENT_DIALOGUE_HISTORY.md")
    open(hist_p, "a").write("\n# tampered\n")
    r = run_guard(tmprepo, "pre")
    assert r.returncode == 2 and "HISTORY 哈希" in r.stdout


def journals(tmp):
    root = os.path.join(tmp, "history", ".guard_journal")
    out = []
    if os.path.isdir(root):
        for name in sorted(os.listdir(root)):
            jp = os.path.join(root, name, "journal.json")
            if os.path.exists(jp):
                out.append(json.load(open(jp)))
    return out


def test_txn_rollback_on_second_replace_failure(tmprepo):
    """E0123 故障注入：第一个替换（HISTORY）成功后第二个（DIALOGUE）失败——
    无事件丢失、自动回滚恢复原状、staging 清理、journal=rolled_back。"""
    before_root = open(os.path.join(tmprepo, "AGENT_DIALOGUE.md"), "rb").read()
    before_arch = open(os.path.join(tmprepo, "AGENT_DIALOGUE_archive.md"), "rb").read()
    r = run_guard(tmprepo, "archive", "--through", "E0019", GUARD_FAIL_AT="step2")
    assert r.returncode != 0 and "自动回滚" in r.stdout
    # 无事件丢失：根文件逐字节恢复原状（E0015-E0020 全在）
    assert open(os.path.join(tmprepo, "AGENT_DIALOGUE.md"), "rb").read() == before_root
    # HISTORY 被前像恢复：归档前不存在 → 应回滚为不存在（或不存在前像场景下保持原状）
    hist_p = os.path.join(tmprepo, "history", "AGENT_DIALOGUE_HISTORY.md")
    assert not os.path.exists(hist_p)
    # 旧 archive 原样保留
    assert open(os.path.join(tmprepo, "AGENT_DIALOGUE_archive.md"), "rb").read() == before_arch
    # staging 清理
    assert not os.path.exists(os.path.join(tmprepo, ".dialogue.stage"))
    assert not os.path.exists(os.path.join(tmprepo, "history", ".hist.stage"))
    # journal 终态 rolled_back（非阻断态）
    recs = journals(tmprepo)
    assert any(x.get("state") == "rolled_back" for x in recs)
    # 回滚后 guard 不被 journal 阻断：唯一预期错误是活动区 6>5 上限（本用例未成功归档）
    rr = run_guard(tmprepo, "check")
    assert rr.returncode != 0 and "超过上限" in rr.stdout and "journal" not in rr.stdout


def test_txn_rollback_failure_blocks_writes(tmprepo):
    """E0123：回滚本身失败——journal/前像保留、pre/check 非零阻断。"""
    before_root = open(os.path.join(tmprepo, "AGENT_DIALOGUE.md"), "rb").read()
    r = run_guard(tmprepo, "archive", "--through", "E0019",
                  GUARD_FAIL_AT="step2,rollback")  # step2 失败 + 回滚注入失败
    assert r.returncode != 0 and "回滚失败" in r.stdout
    # journal 处于 rollback_failed 阻断态
    recs = journals(tmprepo)
    assert any(x.get("state") == "rollback_failed" for x in recs)
    # pre / check 被阻断（非零退出且指向 journal）
    for cmd in ("pre", "check"):
        rr = run_guard(tmprepo, cmd)
        assert rr.returncode != 0 and "journal" in rr.stdout
    # 前像仍完整保留于 journal 目录（可人工恢复）
    root_recovered = False
    jroot = os.path.join(tmprepo, "history", ".guard_journal")
    for name in os.listdir(jroot):
        pre = os.path.join(jroot, name, "pre_dialogue")
        if os.path.exists(pre):
            root_recovered = open(pre, "rb").read() == before_root
    assert root_recovered, "前像 pre_dialogue 应保留完整原根文件"
