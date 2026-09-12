#!/usr/bin/env python3
"""dialogue_guard.py — AGENT_DIALOGUE.md 写入守卫（协议 v1.3，REQ-026/E0120-E0121）

用法:
  python3 cleanup/tools/dialogue_guard.py check              # 联合校验（HISTORY+活动区）
  python3 cleanup/tools/dialogue_guard.py pre                # 写前：双 SHA 对账（活动尾 + HISTORY）
  python3 cleanup/tools/dialogue_guard.py post               # 写后：校验 + 重建 state（含 history_sha256）
  python3 cleanup/tools/dialogue_guard.py archive --through ENNNN
                                                             # 原子归档：把活动区编号<=ENNNN 的可守卫
                                                             # 事件移入 HISTORY（seed=现有 HISTORY 或旧
                                                             # AGENT_DIALOGUE_archive.md），逐块 SHA 验证，
                                                             # 全部通过后才落盘（temp+rename）

v1.3 规则（P1-P6，E0121）:
  P1 字节保真——archive 对每个移动事件块做 SHA 清单，落盘后逐项复验。
  P2 legacy 不透明段——编号<=14（E0001-E0014，含乱序与双 E0010）不做编号/接力校验、不重写；
     仅整文件 SHA 记入 state。严格 +1 与接力链自编号 15（E0015）起。
  P4 五条上限——活动区事件 >5 时 check/post 返回错误。
  P5 state 双摘要——history_sha256（legacy+已归档）+ tail_sha256（活动尾），pre 同时校验。

环境覆盖（测试用）: AGENT_DIALOGUE_FILE / AGENT_DIALOGUE_HISTORY / AGENT_DIALOGUE_ARCHIVE
"""
import hashlib, json, os, re, shutil, sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIALOGUE = os.environ.get("AGENT_DIALOGUE_FILE") or os.path.join(REPO, "AGENT_DIALOGUE.md")
HISTORY = os.environ.get("AGENT_DIALOGUE_HISTORY") or os.path.join(REPO, "cleanup/history/AGENT_DIALOGUE_HISTORY.md")
OLD_ARCHIVE = os.environ.get("AGENT_DIALOGUE_ARCHIVE") or os.path.join(REPO, "AGENT_DIALOGUE_archive.md")
STATE = DIALOGUE + ".state.json"

LEGACY_MAX = 14      # P2: E0001-E0014 为 legacy 不透明段（乱序/双 E0010 原样保留）
GUARDED_START = 15   # 严格校验自 E0015 起
ACTIVE_CAP = 5       # P4: 活动区上限

# P3 事务恢复（E0123）：journal 根目录（持久，与 HISTORY 同级）
JOURNAL_ROOT = os.path.join(os.path.dirname(HISTORY), ".guard_journal")
BLOCKING_STATES = ("in_flight", "rollback_failed")   # 存在任一即阻断 pre/check


def find_blocking_journals():
    """扫描 journal 根目录，返回处于阻断态的事务记录列表。"""
    out = []
    if not os.path.isdir(JOURNAL_ROOT):
        return out
    for name in sorted(os.listdir(JOURNAL_ROOT)):
        jp = os.path.join(JOURNAL_ROOT, name, "journal.json")
        if os.path.exists(jp):
            try:
                rec = json.load(open(jp))
                if rec.get("state") in BLOCKING_STATES:
                    out.append(jp)
            except Exception:
                out.append(jp)  # 损坏的 journal 也视为阻断
    return out

ADJUDICATED = {
    (20, "reviewer"): "supersedes forged E0019; user confirmed reviewer role 2026-09-03",
    (34, "reviewer"): "E0034 typed REVIEW but is an in-request authorization amendment (C4/C8) delivered on user-relayed codex evidence; typing note in E0035",
    (67, "reviewer"): "E0067 typed REVIEW but is an in-request pickle-boundary erratum following E0066 approval (precedent: E0034/E0035 pattern); content valid",
    (108, "reviewer"): "User explicitly authorized preserving E0108 on 2026-09-12; one historical synchronization exception only, not a source approval or future precedent",
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

def header_positions(text):
    return [m.start() for m in re.finditer(r"^### E\d{4} ", text, re.M)]

def block_text(text, pos_list, i):
    end = pos_list[i+1] if i+1 < len(pos_list) else len(text)
    return text[pos_list[i]:end]

def check_chain(events, errs, warns, origin):
    """严格 +1 编号 + 接力链（ADJUDICATED 白名单键为全局编号）。"""
    nums = [e["num"] for e in events]
    for a, b in zip(nums, nums[1:]):
        if b != a + 1:
            errs.append("%s 编号跳档/重复: E%04d -> E%04d" % (origin, a, b))
    prev = None
    for e in events:
        if not e["next_writer"]:
            errs.append("%s E%04d 缺 next_writer 字段" % (origin, e["num"]))
        if prev and e["actor"] != prev["next_writer"] and ADJUDICATED.get((e["num"], e["actor"])) is None:
            tag = "WARN" if e["type"] == "RECOVERY" else "ERR "
            msg = "%s %s 接力断裂: E%04d(%s) 声明 next_writer=%s，但 E%04d 由 %s 撰写" % (
                tag, origin, prev["num"], prev["type"], prev["next_writer"], e["num"], e["actor"])
            (warns if e["type"] == "RECOVERY" else errs).append(msg)
        prev = e

def load_state():
    return json.load(open(STATE)) if os.path.exists(STATE) else None

def history_sha():
    return hashlib.sha256(open(HISTORY, "rb").read()).hexdigest() if os.path.exists(HISTORY) else None

def run_check():
    """联合校验：HISTORY（legacy 不透明 + 可守卫段）+ 活动区（编号续接 + 接力 + 上限）。"""
    errs, warns = [], []
    root_text = open(DIALOGUE, encoding="utf-8").read()
    root_events = parse(root_text)
    hist_guarded = []
    if os.path.exists(HISTORY):
        hist_text = open(HISTORY, encoding="utf-8").read()
        hist_events = parse(hist_text)
        legacy = [e for e in hist_events if e["num"] <= LEGACY_MAX]
        hist_guarded = [e for e in hist_events if e["num"] >= GUARDED_START]
        # P2: legacy 段不校验编号/接力（乱序与双 E0010 原样保留）；仅确认存在
        if not legacy:
            errs.append("HISTORY 缺少 legacy 段（E0001-E0014）")
        check_chain(hist_guarded, errs, warns, "HISTORY")
    # 活动区必须是可守卫编号（>=GUARDED_START）
    for e in root_events:
        if e["num"] < GUARDED_START:
            errs.append("活动区出现 legacy 编号事件 E%04d（应已归档）" % e["num"])
    # 合并链：hist_guarded + root（跨界接力与续接编号）
    merged = hist_guarded + root_events
    check_chain(merged, errs, warns, "合并链")
    if hist_guarded and root_events and root_events[0]["num"] != hist_guarded[-1]["num"] + 1:
        errs.append("边界断裂: HISTORY 尾 E%04d -> 活动区首 E%04d 未续接" % (hist_guarded[-1]["num"], root_events[0]["num"]))
    # P4 上限
    if len(root_events) > ACTIVE_CAP:
        errs.append("活动区事件 %d 条，超过上限 %d" % (len(root_events), ACTIVE_CAP))
    # P3 阻断：未完成 / 回滚失败的事务
    for jp in find_blocking_journals():
        errs.append("存在阻断态事务 journal（%s）——先恢复或裁决后再写入" % jp)
    return errs, warns, root_events

def cmd_check():
    errs, warns, root_events = run_check()
    for w in warns: print(w)
    for e in errs: print("ERR", e)
    last = root_events[-1] if root_events else None
    hs = history_sha()
    print("active=%d history=%s last=%s next_writer=%s tail_sha=%s hist_sha=%s" % (
        len(root_events), "yes" if hs else "no",
        ("E%04d" % last["num"]) if last else "-",
        last["next_writer"] if last else "-",
        tail_sha(open(DIALOGUE, encoding='utf-8').read())[:16], (hs or "-")[:16]))
    sys.exit(1 if errs else 0)

def cmd_pre():
    st = load_state()
    text = open(DIALOGUE, encoding="utf-8").read()
    root_events = parse(text)
    last = root_events[-1] if root_events else None
    blocking = find_blocking_journals()
    if blocking:
        print("ERR 存在阻断态事务 journal（%s）——先恢复或裁决" % blocking[0]); sys.exit(2)
    if st:
        if st.get("tail_sha256") != tail_sha(text):
            print("ERR 活动尾哈希与快照不符——并发写入或中途插入；重读全文按 RECOVERY 处理"); sys.exit(2)
        hs = history_sha()
        if st.get("history_sha256") is not None:
            if hs is None:
                print("ERR state 记录了 history_sha256 但 HISTORY 文件缺失"); sys.exit(2)
            if hs != st["history_sha256"]:
                print("ERR HISTORY 哈希与快照不符——历史被改动；重读并按 RECOVERY 处理"); sys.exit(2)
    print(json.dumps({"expected_next_event": "E%04d" % (last["num"] + 1) if last else "E0001",
                      "legitimate_writer": last["next_writer"] if last else "-",
                      "last": ("E%04d" % last["num"]) if last else "-",
                      "tail_sha256": tail_sha(text), "history_sha256": history_sha()}))

def cmd_post():
    errs, warns, root_events = run_check()
    if errs:
        for e in errs: print("ERR", e)
        sys.exit(1)
    text = open(DIALOGUE, encoding="utf-8").read()
    last = root_events[-1]
    payload = json.dumps({"last_event": "E%04d" % last["num"], "next_writer": last["next_writer"],
                          "tail_sha256": tail_sha(text), "lines": len(text.splitlines()),
                          "history_sha256": history_sha(), "active_count": len(root_events)}, indent=1)
    # E0123: state 原子写（同目录 temp + os.replace）
    st_tmp = STATE + ".tmp"
    open(st_tmp, "w", encoding="utf-8").write(payload)
    os.replace(st_tmp, STATE)
    print("state updated: last=E%04d next_writer=%s active=%d" % (last["num"], last["next_writer"], len(root_events)))

def cmd_archive(through):
    through = int(through[1:])
    text = open(DIALOGUE, encoding="utf-8").read()
    pos = header_positions(text)
    events = parse(text)
    to_move_idx = [i for i, e in enumerate(events) if GUARDED_START <= e["num"] <= through]
    keep_idx = [i for i, e in enumerate(events) if e["num"] > through]
    if not to_move_idx:
        print("ERR 无可归档事件（--through E%04d）" % through); sys.exit(1)
    if any(events[m]["num"] < events[k]["num"] for m in to_move_idx for k in keep_idx) is False and keep_idx == []:
        pass  # 允许全部移空（但当前流程要求至少保留在途事件）
    if keep_idx and min(events[i]["num"] for i in to_move_idx) >= min(events[i]["num"] for i in keep_idx):
        print("ERR 归档边界与保留区交错"); sys.exit(1)
    moved_blocks = [block_text(text, pos, i) for i in to_move_idx]
    keep_blocks = [block_text(text, pos, i) for i in keep_idx]
    manifest = {("E%04d" % events[i]["num"]): hashlib.sha256(b.encode("utf-8")).hexdigest()
                for i, b in zip(to_move_idx, moved_blocks)}
    prefix = text[:pos[to_move_idx[0]]] if to_move_idx[0] == 0 else None
    # 前缀 = 首个被移事件头之前的全部内容（协议正文 + 事件记录节头）
    prefix = text[:pos[to_move_idx[0]]]
    # seed：现有 HISTORY 优先，其次旧 archive
    if os.path.exists(HISTORY):
        seed = open(HISTORY, encoding="utf-8").read().rstrip() + "\n\n"
        seed_from = "history"
    elif os.path.exists(OLD_ARCHIVE):
        seed = open(OLD_ARCHIVE, encoding="utf-8").read().rstrip() + "\n\n"
        seed_from = "archive"
    else:
        seed = ("# AGENT_DIALOGUE 历史归档\n\n"
                "> E0001-E0014 为 legacy 不透明段（早期乱序与重复编号原样保留，不做链校验）；\n"
                "> E0015 起为可守卫段，与根 AGENT_DIALOGUE.md 活动区构成联合编号/接力链。\n\n")
        seed_from = "none"
    new_history = seed + "".join(moved_blocks)  # 块按原字节拼接（末块尾随空行原样保留，不做 rstrip 规范化）
    new_root = prefix + "".join(keep_blocks)

    # ===== P3 事务化提交（E0123）：目标目录 staging + 持久 journal/前像 + 自动回滚 =====
    fail_at = os.environ.get("GUARD_FAIL_AT", "")  # 故障注入：逗号组合，如 "step2,rollback"
    def _inj(tag): return tag in fail_at.split(",")
    os.makedirs(JOURNAL_ROOT, exist_ok=True)
    txn = "txn_%d_%04d" % (int(__import__("time").time()), through)
    jdir = os.path.join(JOURNAL_ROOT, txn)
    os.makedirs(jdir)
    def journal(**kw):
        rec = json.load(open(os.path.join(jdir, "journal.json"))) if os.path.exists(os.path.join(jdir, "journal.json")) else {"txn": txn, "through": through}
        rec.update(kw)
        open(os.path.join(jdir, "journal.json"), "w").write(json.dumps(rec, indent=1))
        return rec

    hist_dir = os.path.dirname(HISTORY); root_dir = os.path.dirname(DIALOGUE)
    th = os.path.join(hist_dir, ".hist.stage"); tr = os.path.join(root_dir, ".dialogue.stage")
    # 前像（before-images）持久保存到 journal
    pre_hist = os.path.join(jdir, "pre_history"); pre_root = os.path.join(jdir, "pre_dialogue"); pre_arch = os.path.join(jdir, "pre_archive")
    try:
        journal(state="in_flight", phase="staging", seed_from=seed_from,
                history_existed=os.path.exists(HISTORY), archive_existed=os.path.exists(OLD_ARCHIVE))
        if os.path.exists(HISTORY): shutil.copy2(HISTORY, pre_hist)
        if os.path.exists(OLD_ARCHIVE): shutil.copy2(OLD_ARCHIVE, pre_arch)
        shutil.copy2(DIALOGUE, pre_root)
        # staging 写入目标目录（同文件系统 → os.replace 原子）
        open(th, "w", encoding="utf-8").write(new_history)
        open(tr, "w", encoding="utf-8").write(new_root)
        # 落盘前复验（staging 内容）
        htext = open(th, encoding="utf-8").read()
        hpos = header_positions(htext)
        got = {}
        for i, p in enumerate(hpos):
            m = re.match(r"^### (E\d{4}) ", htext[p:p+20])
            num = int(m.group(1)[1:])
            if GUARDED_START <= num <= through:
                got[m.group(1)] = hashlib.sha256(block_text(htext, hpos, i).encode("utf-8")).hexdigest()
        bad = {k: (v, got.get(k)) for k, v in manifest.items() if got.get(k) != v}
        if bad:
            journal(state="aborted", phase="pre_write_verify", bad_blocks=list(bad)[:3])
            print("ERR 归档复验失败（块 SHA 不符）:", list(bad)[:3]); sys.exit(1)
        rtext = open(tr, encoding="utf-8").read()
        rpos = header_positions(rtext)
        rnums = [int(re.match(r"^### (E\d{4}) ", rtext[p:p+20]).group(1)[1:]) for p in rpos]
        if rnums != [events[i]["num"] for i in keep_idx]:
            journal(state="aborted", phase="pre_write_verify", reason="keep_sequence_mismatch")
            print("ERR 保留区复验失败（编号序列不符）"); sys.exit(1)

        def rollback(reason):
            """按前像恢复；任一恢复失败 → journal=rollback_failed 并抛出。"""
            try:
                if _inj("rollback"): raise OSError("injected rollback failure")
                if os.path.exists(pre_root): shutil.copy2(pre_root, DIALOGUE)
                if os.path.exists(pre_hist): shutil.copy2(pre_hist, HISTORY)
                elif not json.load(open(os.path.join(jdir, "journal.json")))["history_existed"] and os.path.exists(HISTORY):
                    os.remove(HISTORY)  # 归档前 HISTORY 不存在 → 回滚时移除新建文件
                if os.path.exists(pre_arch) and os.path.exists(OLD_ARCHIVE) is False and \
                   json.load(open(os.path.join(jdir, "journal.json")))["archive_existed"]:
                    shutil.copy2(pre_arch, OLD_ARCHIVE)
                for s in (th, tr):
                    if os.path.exists(s): os.remove(s)
                journal(state="rolled_back", reason=reason)
                print("ERR 提交失败（%s）已自动回滚，现场恢复原状；journal=%s" % (reason, jdir))
            except Exception as re_:
                journal(state="rollback_failed", reason=reason, rollback_error=str(re_))
                print("ERR 提交失败（%s）且回滚失败——journal 与前像保留于 %s，pre/check 已阻断" % (reason, jdir))
            sys.exit(1)

        # commit step1: HISTORY
        journal(phase="commit_step1")
        if _inj("step1"): raise OSError("injected step1 failure")
        os.replace(th, HISTORY)
        # commit step2: DIALOGUE
        journal(phase="commit_step2")
        if _inj("step2"): rollback("injected step2 failure")
        os.replace(tr, DIALOGUE)
        # commit step3: 移除旧 archive（seed 已并入）
        journal(phase="commit_step3")
        if _inj("step3"): rollback("injected step3 failure")
        if seed_from == "archive" and os.path.exists(OLD_ARCHIVE):
            os.remove(OLD_ARCHIVE)
        for s in (th, tr):
            if os.path.exists(s): os.remove(s)
        journal(state="committed",
                history_sha=hashlib.sha256(open(HISTORY,'rb').read()).hexdigest(),
                root_sha=hashlib.sha256(open(DIALOGUE,'rb').read()).hexdigest())
        print(json.dumps({"moved": len(manifest), "kept": len(keep_idx), "seed_from": seed_from,
                          "history_bytes": os.path.getsize(HISTORY), "root_bytes": os.path.getsize(DIALOGUE),
                          "journal": jdir,
                          "manifest_sha_all": hashlib.sha256(
                              "".join(sorted(manifest.values())).encode()).hexdigest()}))
        print("归档成功——请立即运行 post 重建 state")
    except SystemExit:
        raise
    except Exception as e:
        rollback("unexpected: %s" % e)

def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "check": cmd_check()
    if cmd == "pre": cmd_pre(); return
    if cmd == "post": cmd_post(); return
    if cmd == "archive":
        if len(sys.argv) < 4 or sys.argv[2] != "--through":
            print("用法: archive --through ENNNN"); sys.exit(1)
        cmd_archive(sys.argv[3]); return
    print("未知子命令:", cmd); sys.exit(1)

if __name__ == "__main__":
    main()
