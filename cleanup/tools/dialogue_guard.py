#!/usr/bin/env python3
"""dialogue_guard.py — AGENT_DIALOGUE.md 写入守卫（协议 v1.1，用户指令 2026-09-03）

用法:
  python3 cleanup/tools/dialogue_guard.py check   # 全文件校验（编号/字段/接力链）
  python3 cleanup/tools/dialogue_guard.py pre     # 写前：验证尾部哈希，报告合法写入者与下一事件号
  python3 cleanup/tools/dialogue_guard.py post    # 写后：校验 + 更新状态快照

角色槽位（R11）：codex 与 reviewer 事件只能由对应会话撰写。
接力链（R13）：事件 actor 应等于前一事件 next_writer；RECOVERY 型越权写入必须自带说明，
              守卫将其降级为 WARN，由 reviewer/用户裁决。
守卫是机械辅助，不提供密码学认证——伪造无法被绝对阻止，只能被快速检测与归因。
"""
import hashlib, json, os, re, sys

HERE = os.path.abspath(__file__)
DIALOGUE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), "AGENT_DIALOGUE.md")
STATE = DIALOGUE + ".state.json"
ADJUDICATED = {
    (20, "reviewer"): "supersedes forged E0019; user confirmed reviewer role 2026-09-03",
}

HDR = re.compile(r"^### (E\d{4}) \| (REQ-\d{3}) \| (codex|reviewer) \| ([A-Z_]+)\s*$")

def tail_sha(text, n=400):
    return hashlib.sha256(text[-n:].encode("utf-8")).hexdigest()

def parse(text):
    lines = text.splitlines()
    events, cur = [], None
    for i, ln in enumerate(lines, 1):
        m = HDR.match(ln)
        if m:
            if cur: events.append(cur)
            cur = {"line": i, "num": int(m.group(1)[1:]), "req": m.group(2),
                   "actor": m.group(3), "type": m.group(4), "next_writer": None}
        elif cur and ln.startswith("- next_writer:"):
            cur["next_writer"] = ln.split("`")[1] if "`" in ln else ln.split(":", 1)[1].strip()
    if cur: events.append(cur)
    return events

def check(events):
    errs, warns = [], []
    nums = [e["num"] for e in events]
    if nums != sorted(nums):
        errs.append("事件编号非单调递增: %s" % nums)
    for a, b in zip(nums, nums[1:]):
        if b != a + 1:
            errs.append("编号跳档/重复: E%04d -> E%04d" % (a, b))
    prev = None
    for e in events:
        if not e["next_writer"]:
            errs.append("E%04d 缺 next_writer 字段" % e["num"])
        if prev and e["actor"] != prev["next_writer"] and ADJUDICATED.get((e["num"], e["actor"])) is None:
            tag = "WARN" if e["type"] == "RECOVERY" else "ERR "
            msg = "%s 接力断裂: E%04d(%s) 声明 next_writer=%s，但 E%04d 由 %s 撰写" % (
                tag, prev["num"], prev["type"], prev["next_writer"], e["num"], e["actor"])
            (warns if e["type"] == "RECOVERY" else errs).append(msg)
        prev = e
    return errs, warns

def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    text = open(DIALOGUE, encoding="utf-8").read()
    events = parse(text)
    errs, warns = check(events)
    last = events[-1] if events else None
    if cmd == "check":
        for w in warns: print(w)
        for e in errs: print("ERR", e)
        print("events=%d last=E%04d next_writer=%s tail_sha=%s" % (
            len(events), last["num"], last["next_writer"], tail_sha(text)[:16]))
        sys.exit(1 if errs else 0)
    if cmd == "pre":
        st = json.load(open(STATE)) if os.path.exists(STATE) else None
        if st and st["tail_sha256"] != tail_sha(text):
            print("ERR 尾部哈希与快照不符——存在并发写入或中途插入；重读全文、按 RECOVERY 处理")
            sys.exit(2)
        print(json.dumps({"expected_next_event": "E%04d" % (last["num"] + 1),
                          "legitimate_writer": last["next_writer"],
                          "last": "E%04d" % last["num"], "tail_sha256": tail_sha(text)}))
        return
    if cmd == "post":
        if errs:
            for e in errs: print("ERR", e)
            sys.exit(1)
        json.dump({"last_event": "E%04d" % last["num"], "next_writer": last["next_writer"],
                   "tail_sha256": tail_sha(text), "lines": len(text.splitlines())},
                  open(STATE, "w"), indent=1)
        print("state updated: last=E%04d next_writer=%s" % (last["num"], last["next_writer"]))
        return

if __name__ == "__main__":
    main()
