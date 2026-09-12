# Cleanup status

Last updated: 2026-09-12 (REQ-025 已由 E0119 接受：两份 legacy-prune 规则副本已同步；没有在途请求，准备根目录审计归档与测试迁移提案)

This is the canonical global status file for the scheduler cleanup work. It supersedes `PHASE1_STATUS_FOR_NEXT_AGENT.md` as the main entry point for future agents.

Use this file to understand the current cleanup state, approved decisions, executed batches, protected areas, and next actions. Use `FILE_ADJUSTMENT_RECORD.md` for the global action/change history.

## Current Review Gate (authoritative, 2026-09-12)

- REQ-024已由用户授权恢复并提交为e758540058faa4f30773b43e6f1fe244543bfeab；E0108保留为一次性历史同步例外，不授权业务改动。
- REQ-025的四份外部skill/policy文本已由reviewer实施，并由Codex在E0119接受；规则只覆盖暂时不用的通用工具、全局参数和数学助手，不扩大为所有未用符号的默认处置。
- 验证已闭环：两份skill通过`quick_validate.py`；SKILL块按“从`### 临时辅助代码`起至下一H2前空行起点，含正文末尾一个LF”计为340字符/828字节/SHA256 3332c623…305d3；policy段按“从`Narrow rule`起至段末、不含其后LF”计为904字符/906字节/SHA256 c1dc3a53…d10a；每对副本逐字节一致。
- 当前没有在途请求，下一合法写入者是codex。根目录审计归档和根级测试迁移尚未提案或执行；业务源码、测试、依赖、账本、audit的.claude/、受保护文档和原始scheduler worktree保持不变。
- 下方较早的“当前”段落保留作历史证据；与本节冲突时，以本节和协议末尾事件为准。

## Goal Gate (2026-09-12)

当前有效状态（E0111）：用户已明确授权保留E0108，并只为这一个历史事件登记同步例外。REQ-024恢复并关闭；正常审查流程重新启用。该例外不批准源码、不豁免未来事件，也不改变“Codex提案和最终复核、reviewer审查并实施”的分工。恢复验证已完成：`dialogue_guard.py check` rc0（仅五条既有历史WARN）、`post` rc0并把state更新为E0111/codex、`pre` rc0并返回E0112/codex；`git diff --check` rc0，index为空，保护路径无diff。B24的177文件基线中176项未变；唯一差异是本次用户授权的`cleanup/tools/dialogue_guard.py`一行历史裁决，SHA从`15879c4…`变为`67773b52…`。五个明确路径已提交为恢复快照`e758540058faa4f30773b43e6f1fe244543bfeab`，不含未跟踪内容。REQ-025四份外部skill/policy文本已由reviewer实施，E0114 RESULT已返回Codex；E0115仅为本文件状态表述与计数口径勘误，reviewer执行后E0116 RESULT待Codex最终复核。本段以下较早文字保留为历史背景。

恢复后第2轮复核（2026-09-12）：上一轮只有状态复核，没有清理进展。本轮仍无E0108用户裁决；check=1/pre=2，末尾E0110/state E0109、HEAD968b12e及四metadata差异未变。当前目标保持active，重新计数尚未达到三轮阻塞阈值。源码/skill/协议/测试/提交均不操作；仅更新双全局记录，不能把重复核验当作实质推进。

执行状态：BLOCKED_WAITING_USER_DECISION，不是目标完成。相同的E0108历史越权/协作门禁问题已连续三轮出现：用户偏好更新轮、五个表格脚本只读补查轮、本轮原计划完成度核对。前两轮确有纠正与新增证据，本轮复查仍为check=1/pre=2；未收到用户对一次性历史裁决的明确同意。真实clean会话30074经实际轮询仍存活但闲置，不是运行中任务等待。不得自动重开会话、恢复例外、手改state或自行批准提案。

原计划核验结果：

| 项目 | 当前结论 | 核验依据 |
| --- | --- | --- |
| 行为基线 | 已验收并提交 | B10/E0025；test_binpack_pipeline_contract.py及Split/fixcore-Repack基线在库 |
| 初始化上下文封装 | 已实施 | sim_main.py:577的init_sched_components；approach/approach_setup.py:36调用；B10转发契约保护 |
| 收窄perform_bin_packing | 未实施，REQ-003取消 | 函数仍在sim_main.py:381；不能把取消写成完成 |
| Repack拆分 | 延期保留 | E0029 DEFER_KEEP_AS_IS；scratch问题不在本轮修复 |
| 资源分配计算/提交拆分 | 并入未来Repack设计 | E0031 DEFER_WITH_REPACK；不擅自重开 |
| 调度表附属功能拆分 | 裁定KEEP_AS_IS | E0027；类内与仍在用的辅助功能保留 |
| approach包迁移 | 已验收并提交 | E0037/B11；七个实现和根兼容入口、32项测试；包含在ea46783 |
| 两份历史approach文件归档 | 已验收并提交 | E0041/B12；old/appoach_plot6.py和old/approach_util33.py、7项测试；包含在ea46783 |

全仓后续探索及独立审查尚未全部完成；最近五表格脚本检查只补证据，不是全仓通过或新清理授权。两个legacy-prune仍保留旧符号后缀默认，新偏好只已落在双全局记录并通知对方，skill同步尚待实施。REVIEW项继续原位保留。B11-B25快照已存在，当前四份元数据修改仍待审未提交。

解除条件：用户明确裁决是否为E0108登记一次性历史例外并恢复同步；不得扩大到未来事件豁免、源码批准或抹去历史。恢复后先独立审核协议修复、同步新偏好并建立精确提交快照，再继续未完成的清理；不把自动续跑当作这项批准。

## Current Roles (2026-09-12)

REQ-025：E0113 APPROVED → reviewer 已实施四 skill/policy 文件同步（窄规则=暂时用不到的通用工具/全局参数/数学助手；原位保留→依赖核查后原文件尾注释分隔→整文件就近迁移保留文件名；禁默认 *_old/_unused 改名；既有提交不回滚；优先于通用 symbol-slice 命名表且不泛化）。quick_validate 两份 PASS；两份新增文本分别跨副本一致。**E0119 已接受：当前写入者与两种不同计数边界均已独立复核。**audit .claude/ 与业务路径零触碰。其下较早的“当前”段落为历史背景，非当前状态。

当前有效状态（E0111）：Codex已按用户授权完成一次性历史裁决，并独立接受REQ-024的协议恢复；E0112/REQ-025现由Codex提案，`next_writer=reviewer`。E0108保留作错误证据，但只通过`cleanup/tools/dialogue_guard.py`中精确的`(108, reviewer)`条目被识别，不是通用例外。REQ-025只同步用户的新“原位保留通用工具/全局参数/数学助手”偏好到两份legacy-prune skill及其policy；reviewer审查并实施后，由Codex独立复核。任何源码批次仍须独立提案、reviewer批准和实施、Codex验收。

当前核验（Codex，E0110之后）：REQ-024尚未验收，等待用户确认是否为E0108登记一次性历史裁决并恢复同步，不推进源码或skill实施。对方新增守卫例外已撤回，守卫与HEAD逐字节一致，177代码SHA全部匹配。仅四个tracked元数据文件修改，index为空，git diff --check通过。本轮没有新commit/push；原快照968b12e与源码快照ea46783仍在。实际协议末尾E0110/next_writer=codex；state停在E0109/next_writer=reviewer（不是E0108），因为post拒绝更新。guard check=1（E0107→E0108历史越权）、pre=2（快照落后），不能当作通过，不能手改state。所有历史事件保留，撤销只追加裁决，不执行后文旧条目的删事件建议。

用户新偏好（立即有效）：暂时用不到的通用工具、全局参数和数学助手，优先先不动；需要归类时，先检查执行/初始化依赖，再放到原文件结尾并加注释分割线；其次才考虑就近文件夹迁移并保留原文件名，不默认改成*_unused.py。旧提交不自动回滚，fit.py归档暂停。两份legacy-prune与引用政策尚未同步，不能声称已完成。新增gurobi回归330=325通过+原5collector失败，0error/skip。

用户已要求继续清理并交换实施职责：Codex负责提案和独立结果审查，原clean会话负责审查提案、实施和修正。会话身份仍为codex/reviewer，禁止代写对方事件。协议 v1.2 正文已实施（E0103→E0104→E0105→E0106→E0107）；E0108 系 reviewer 在 next_writer=codex 时的越权写入（依据的澄清实为 Codex 转述、非用户直接授权），已被 Codex E0109 拒绝为无效交接，事件保留作证据；尚未启动新源码批次。

实际执行基线为968b12e6882619bb67693c5fa5ec1865fcd47ef1，源码快照ea46783已提交，177业务代码SHA与验收状态一致；当前 tracked 改动仅为四个 metadata 路径（AGENT_DIALOGUE.md 协议正文与事件、双全局记录、快照），index 无暂存。旧段落中的暂停、Codex修改和未提交只代表当时状态，本轮以上述用户新角色指令和协议末尾为准。主线master和原worktree不动。

本轮已重读CLAUDE测试环境、doc/spec/readme、全局状态与协议；沿既有DFS只读throughput_cnt和fit，不作迁移。旧11983工具句柄失效，恢复相同clean历史UUID并确认B25/E0102勘误及audit目录；当前交互会话30074保持。未给其他项目的Claude进程输入。

下一步：按 Codex E0109 限定纠正执行完毕（守卫 (108) 行已撤回、与 HEAD 逐字节一致 SHA 15879c4e…；本两处状态行更正；E0110 reviewer RESULT 已追加）。**guard check 预期非零**：E0107→E0108 历史断链 ERR 保留为诚实状态，不绕过、不手动改 state；等待用户裁决恢复同步点。此后 Codex 复核 E0110；177 业务代码零改动。

REQ-024 当前审查：E0109要求的守卫撤回已由对方完成，Codex已独立核实E0110结果。当前不是ACCEPTED；协议同步仍须用户裁决。新偏好已记入双全局文件并通知原clean会话，两份skill尚未同步。

## Read-only Followup (2026-09-12)

协作门禁未恢复期间，继续已有DFS的实验后处理链，只读补完analyze/xlsl_e2e_lat_abla.py、xlsl_e2e_latency.py、xlsl_max_tp.py、xlsl_min_core.py、xlsl_safe_scalable.py五份源码（246/123/59/77/55逻辑行，共560；wc仅557换行，因为3文件末尾无LF）。run/1/all.sh:22-24明确直接调用后三个统计入口中的max_tp/min_core/e2e_latency，这三份KEEP；abla/safe_scalable原位REVIEW保留，均无迁移/删除批准。模块顶层会解析参数并读CSV、写CSV/Excel，不能仅凭无import判死，也不能把被顶层调用的定义移到调用之后。

验证使用CLAUDE.md指定gurobi绝对Python：五份AST有效；python -B -m analyze.<模块> --help全部rc0；直接python -B analyze/<文件>.py --help则前四份因No module named 'analyze'返回1，safe_scalable返回0。openpyxl和xlsxwriter在该环境find_spec均为空，pandas可找到；未安装依赖，没有读实验CSV或输出工作簿，也没运行run/1/all.sh主体。帮助检查不等于实验通过。日志/tmp/scheduler-xlsl-readonly-4f8cp25j；177代码SHA全同，仍仅四metadata未提交。细节与后续运行问题见FILE_ADJUSTMENT_RECORD末条，不是新清理批次或reviewer验收。

## Latest Git Snapshot

- 源码快照：`ea46783b5892b80c135e1e59180971a1475797c8`，父提交 `a1d933b1f26efd4d569eb3d8ffb313447294443b`。
- 分支：`audit/minimal-from-test_pipeline-20260612`；主线 `master` 和原始 `test_pipeline` worktree未操作。
- 范围：13个已独立验收源码批次B11-B21、B23、B24；B22/B25分析、获批测试、归档及恢复材料。精确97路径，见 `cleanup/reports/snapshot-b11-b25-20260906.json`。
- 验证：本次15模块330项，325通过，原5个collector失败不变，0error/skip。177代码哈希与验收状态一致。不是全仓测试全绿。
- 当前动作：源码快照已完成。本节与历史条目随后单独提交，只含两份全局记录；该记录提交可用 `git log -1 --format='%H %s' -- CLEANUP_STATUS.md FILE_ADJUSTMENT_RECORD.md` 定位，无需在文件内自引用其hash。
- 历史口径：下方旧批次的“未提交”“无commit授权”是当时状态；本次用户明确授权并完成上述快照，旧审查证据不回写。
- 下一步：当前持续目标暂停，本轮不继续源码清理；下一批开始前重新读取实际HEAD、协议末尾和本节，不沿用a1d933b作为新的执行基线。

## Mandatory update rule

After every cleanup action, preflight, environment verification, status-maintenance step, or batch execution, update both global records before handing off:

- `CLEANUP_STATUS.md`: update the current global action state, decision status, tracked diff, validation state, and recommended next action.
- `FILE_ADJUSTMENT_RECORD.md`: append a chronological action-history entry. If the action did not change source/tracked files, explicitly record that it was a no-source-change or preflight/status-only action.

Per-batch packets and ledgers are still required when applicable, but they do not replace these two global records.

用户清理规则（2026-09-06 重申）：主线基准 master，每批另记实际执行前 HEAD；类内先不管，旧 scheduler 类整体分离；默认移动而非删除，不改逻辑或接口，不用换行伪装改写；路径变化可配套改 import。先理解模块，再沿依赖深度优先审查。根目录整文件放 root/old 或 unused，包内整文件放原 package/old 或 unused，符号片段放源文件同目录的 basename_old.py 或 basename_unused.py。old 表示被新版本替代，unused 表示独立但当前未用。

这些规则已同步到 `C:/Users/diyuf/.agents/skills/legacy-prune/SKILL.md` 和 `D:/document/Research/coding/legacy-cleanup-slim/.claude/skills/legacy-prune/SKILL.md`，并在两处 references/cleanup-policy.md 声明优先级。没有整目录覆盖，也未修改原 worktree 或 audit 内受保护的 .claude。备份和恢复命令见 FILE_ADJUSTMENT_RECORD.md。

规则验证：两份legacy-prune用户规则块一致，真实clean reviewer已进行只读场景核对。账本追加先核列结构，被活路径借用的历史组件需单独提案；规则同步动作本身没有改源码。E0037是此前B11收尾记录，当前批次状态以下方协调区为准。没有自动提交授权。

偏好汇总复核（2026-09-06，B15收尾时）：两份legacy-prune的1696字符用户规则块完全一致，两处cleanup-policy均声明这些规则优先。master基准、双全局记录、类内暂留、禁止逻辑/接口改写、移动优先、深度优先和三类归档位置已落实，无需重复写skill。此次只补记核对与已有B15验证，不新增源码批次、不更改已有裁决。当前批次状态以本文件协调区为准。

偏好规则再次核对（2026-09-06，RULE-PREF-CHECK-20260906）：已直接读取上述两份 legacy-prune skill，统一 CRLF/LF 后，1696 字符的用户规则块完全一致，覆盖用户本轮全部要求，无需重复改写 skill。当前状态查本文件，实际动作与恢复说明查 FILE_ADJUSTMENT_RECORD.md；旧 PHASE1 文件不再承载全局状态。本次只核对规则并更新双记录，没有推进或验收在途代码批次，不更改 decision、reviewer 裁决或提交状态。源码修改前仍须取得独立 reviewer 批准，Codex 不自行批准；完全未用的判断本身也不是删除授权。

本轮偏好总结核验（2026-09-06，PREFERENCE-RECAP）：重新读取两份 skill 与两份 cleanup-policy，分别定位规则块后确认 1696 字符完全一致，引用政策均声明用户规则优先。没有重写 skill，也没有启动新批次、运行测试、修改协议或操作 Git；仅更新本段、页首维护说明和 FILE_ADJUSTMENT_RECORD.md。以下批次状态沿用已有记录，不作为本轮重新验收的结论。

## Peer-review gate

`AGENT_DIALOGUE.md` is the coordination channel between Codex and the peer reviewer. Only one request may be active. Codex must receive proposal approval before changing the requested code and result acceptance before starting the next request.

Current coordination state:

- Request: `REQ-023` / `B25-LEGACY-OPTIMIZER-RUN-AUDIT`：`CLOSED / ACCEPTED`，真实reviewer E0102确认28文件KEEP_AS_IS，补核已落盘。本批没有源码迁移、删除或接口退役授权。
- Scope: 7Python全文、15shell全文、1shell全文差异，4既有old归档只审边界，NoC沿用B24全文再核。保护文档API、交互测试和类内保留；独立optimizer/旧脚本的不确定项原位REVIEW。
- Evidence: 已直接读取/tmp/b25_review_final/review.json（SHAa805668c…61cfcf），6条help rc0/0/0/0/0/1，16份bash -n全0，5导入探针原torch/graphviz缺失与NoC TypeError吻合，177代码及双账本/6文档SHA一致。证据持久存入B25报告，不把语法或帮助成功说成实验通过。
- Erratum: NUL分隔git ls-files恰16份shell，全部存在；此前两个假失败来自reviewer把含空格的run/aba_scalability_scan copy.sh拆开，不是索引缺失。reviewer已补正历史说明，E0102原文不回写；不改index。
- Safety: 本轮未运行训练、实验、交互绘图、清理主体或pytest，未安装依赖。run/clean.sh及共享/tmp/fd1旧脚本保持未执行。六个旧目标仅记录不在tracked/approved清单，不检查untracked。
- Followup: 等待期间完整只读fit.py、throughput_cnt.py、scripts/test_alloc_lat.py。前两者待引用/行为审查；后者是保护测试，L2把原始worktree加入sys.path，因此不执行、不改写。此三项不扩展E0102源码授权。
- Snapshot: 已提交 `ea46783b5892b80c135e1e59180971a1475797c8`（`audit: checkpoint accepted B11-B25 cleanup work`），父提交a1d933b1f26efd4d569eb3d8ffb313447294443b。13个源码批次B11-B21/B23/B24、B22/B25分析及相关审计/测试/归档均已纳入，97精确路径；不再有这些批次的未提交源码。177代码与验收SHA一致，15模块回归325通过+同5个collector失败。两old归档已纳入，未收其他untracked、未push、未操作原worktree。当前只补双全局记录并形成单独记录提交。
- Next/protocol: 末尾E0102、next_writer=codex，11983会话保持。全仓语义审查未完成，持续目标当前暂停；本轮只按用户要求建立快照，不开新源码批次、不重开REQ003/005/006/007。

B24 已关闭记录（以下不是当前在途请求）：

- Request: `REQ-022` / `B24-MAPPER-LEGACY-HELPERS`：`CLOSED / ACCEPTED`，真实reviewer E0100及同会话补核已完成，没有在途请求。REQ-003取消，REQ-005/006/007原裁决不变。
- Scope: mapper/mem_planner.py三个完整顶层函数共120行，原样迁入同目录mem_planner_old.py及mem_planner_unused.py；新增40项测试。7类内部、其余13顶层函数和原imports/EOF不改。旧导出和函数pickle路径退休属已批准MEDIUM兼容变化，仓外消费者未知。
- Tests: Codex直接解析reviewer的/tmp/b24_review_final/b24only.xml（40通过）和fifteen.xml（330=325通过+同样5项collector失败，0error/skip），全部15模块数量和异常消息与本批回归一致。三help rc0；保护test_mem_planner实际import rc1、原NameError，不修、不删。不是全仓测试全绿。
- Supplement: reviewer在/tmp/b24_review_final/supplement.json补核迁移前173份SHA、两表原前缀和历史坏行、4条实际CLI命令及三个新位置pickle身份，均符合批准条件。Codex已读取并将结果及文件SHAac7f5d2b…be92da持久化到B24报告。177份迁移后快照核对和173份迁移前后比对是两个独立检查，不互相替代。
- Integrity: 源纯删120行，29334字节，剩余SHA482af528…e39d10，与原HEAD精确删除批准块的字节重建一致；三个归档原块各一次。O5工具少1LF纠正及首轮39/1新测试错误留档；E0098仅两行Optional测试修正，未改行为golden。当前177代码SHA全部不变，161Python AST有效，保护区/现有测试/依赖/HEAD/index不变。
- Ledgers/recovery: move50→53追加3×14，reachable271→274追加3×10，原25708/74515字节前缀及8历史坏行保留。B24 JSON validation.recovery保存四代码路径限定patch（1旧源+2归档+1新测试），SHA577a2834…d9ebe；持久解码后reverse--check=0，未执行恢复。diff --check仍只有已知sim_main.py:677尾空行警告。
- Read-only frontier: mapper路径本轮读完；model/noc.py的3类和两个空包初始化保留。NoC实际导入原TypeError: abstract class列REVIEW，不改注解，不将类内修复混入清理。
- Protocol/commit: E0100与快照一致，next_writer=codex，11983 reviewer会话保持打开。B11-B21加B23、B24共13个源码批次已验收未提交；B22只分析，不计源码批数。无commit/push授权。
- Next action: 沿已记录mapper/model依赖顺序继续只读DFS，不确定项保留；任何下一源码动作仍需先提案批准、执行后验收。全仓语义审查尚未完成，持续目标保留。
- Closeout check: 记录落盘后再次执行新40测试，全过，日志/tmp/scheduler-b24-close-9ie44_bp/new.xml；报告/DFS JSON有效，原行为golden不变，177代码与两账本SHA一致，reviewer两XML与本批回归逐字段一致。协议仍E0100、pre=0，无源码新增或提交。

B23 已关闭记录（以下不是当前在途请求）：

- Request: `REQ-021` / `B23-UNUSED-EXPERIMENT-HELPERS`：`CLOSED / ACCEPTED`，真实 reviewer E0092。没有在途请求；REQ-003 取消，REQ-005/006/007 裁决不变。
- Scope: round_to_step 完整3行、extract_num_cores 完整7行，原字节迁入各自同目录 _unused；新增31项测试。其余源码、原imports、类内及旧分析链保持不变。不是逻辑重写。
- Tests: 已直接解析 reviewer 的 /tmp/b23_review_final/b23only.xml（31 passed）与 fourteen.xml（290=285 passed+相同5个collector失败，0 error/skip）；十四模块计数和异常消息逐项匹配 Codex 的实际回归。没有跳过或修复原失败。
- N3 supplement: reviewer 在同一会话补做169个非本批文件逐键SHA比对，并按 validation.cli 的11条原命令执行：前10条help全0，最后analyze_tp仍以1退出、报原sim_seq AttributeError。CLI结果见 /tmp/b23_review_final/n3-supplement.json；repack_sweep不属于本批范围。此处补足E0092宽泛表述，不改历史事件。
- Integrity: Codex另独立核对169非本批SHA及全部174当前代码，均无差异；158份Python可解析。两源精确0增/3删、0增/7删，等于HEAD删批准块的全字节重建；两归档原块各一次。原e2e有尾LF/stat无尾LF保持，N5授权的单字节纠正已完成，不改预期。
- Boundary: 两旧导出及函数pickle路径退休是E0090明确批准的MEDIUM兼容变化，仓外使用未知。两候选不在保护指南API清单；E0088五个exp_common文档API继续保留。保护区、依赖、HEAD与index未变。
- Ledgers/recovery: move48→50新增2×14，reachable268→271新增3×10；原24806/73533字节前缀与8条历史坏行原样保留。B23 JSON validation.recovery存五代码路径限定patch，SHA3360d780…fa7f7f6，持久解码后reverse --check为0，未执行恢复。git diff --check仍只有E0082接受的sim_main.py:677尾空行警告。
- Read-only DFS: 五XLSL模块help通过、三个旧直接入口缺analyze包，不修；手工scan选项保留。另完整读取debug_sink_constraint和example/bm1、bm2、bm3：诊断脚本保留，三个样例尚无活调用证据，列REVIEW并保留。mapper/mem_planner只读了1–37行及AST元数据，三项候选仅REVIEW，未改源码。
- Protocol/commit: E0092与快照一致，next_writer=codex，11983 reviewer会话保持打开。B11–B21加B23共十二个源码批次已验收但未提交；B22是分析批次，不计源码批数。验收不是commit/push授权。
- Closeout check: B23/DFS JSON完整解析通过；当前协调区、审核包、全局历史与E0092一致；174代码及两账本SHA未变，frontier所记SHA和N3补核文件SHA匹配，守卫pre为0。没有改协议或新增源码批次。
- Next action: 沿mapper/mem_planner路径继续只读审查完整顶层函数、保护测试和实际调用方，再判断是否提案。任何下一源码动作须先获真实reviewer批准、实施后验收。全仓语义审查未完成，持续目标保留。

B22 已关闭记录（以下不是当前在途请求）：

- Request: `REQ-020` / `B22-RESOURCE-EXP-BOUNDARY-AUDIT`：`CLOSED / ACCEPTED`，真实reviewer E0088裁定`KEEP_AS_IS`。这是分析性请求，不构成源码迁移、删除、导出退休或文档修改授权；当前无在途请求。
- Request: `REQ-023` / `B25-LEGACY-OPTIMIZER-RUN-AUDIT` — **CLOSED, ACCEPTED (E0102) — 分析性请求，28 文件 KEEP_AS_IS，不构成源码授权**。optimizer/scheduler_base.py（draw_computational_graph 被delta_ver.py:305/340 实际调用+保护指南列出）、ops_test.py（既有交互测试）、optimizer 其余三模块 REVIEW、model/noc.py REVIEW（原 TypeError）、run 脚本 REVIEW/KEEP 原位、old/ 四档 REUSE_CANDIDATE 整体保留。独立复核：6 cli 重放 rc 与声明逐项一致（含 motiv shell -h 原 exit 1）、16 shell bash -n 全 0（【E0102 收尾勘误】首轮 2 FAIL 系 reviewer 检查命令对含空格文件名 `run/aba_scalability_scan copy.sh` 的分词错误——NUL 清单实得恰 16 份全部实存且 bash -n 全 0，Git 索引无缺失，非'index 幽灵条目'）、5 导入探针 rc=1×5 异常类型吻合（缺包如实保留未安装）、双账本+6 保护文档 SHA 全一致、**B24 177 基线独立重hash OK=177 变化 0**。13 个已验收源码批次数不增加；语法/help ≠ 实验通过（clean.sh 递归 rm 未执行）。全仓审查仍未完成，后续沿既有 DFS 检查剩余独立模块。
- KEEP decision: scripts/exp_common的group_by_key、compute_group_means、setup_dual_axis_plot、save_and_close_figure、add_value_labels虽无仓内可执行引用，仍是保护指南列明的公共API。原五导出及pickle路径保留，不加兼容层，不改指南；未来若要处置须单独提案，并先由用户裁定两份指南API条目的修订。
- Resource/message review: 本轮完整读取11模块，另核TaskInt/旧Scheduler局部上下文。DDL/RT、Monitor、performance、bin_list_utils、trace_analyser均有实际借用证据；dummy仅有导入不声称实例化。类、自带示例/测试和旧runtime整体保留。
- Evidence: 171代码SHA在分析后、E0088后均全量核对不变。155Python AST/16shell及保守静态67入口可达只表示当前库存和静态边（含main/TYPE_CHECKING/惰性导入），不是全仓语义审完；初始137Python快照保留。B22 JSON/packet和DFS latest_semantic_review已同步验收状态。
- Read-only continuation: 等待期间读取两个独立E2E脚本的顶层函数和入口，两个--help实际rc0，日志/tmp/scheduler-e2e-entry-audit-g_vk883k。它们是独立CLI，不因缺少前三入口导入而归档；未运行完整实验或新增pytest。round_to_step只发现定义与历史候选行，暂列REVIEW，未经提案/批准不能移动。
- Scope: 本轮没有源码/测试/依赖/保护文档/ledger改动；HEAD/index和171代码不变。B21的259=254+原5仅为历史回归背景，未冒充本轮测试。B11-B21共十一源码批次已验收未提交，B22只读分析，不计入源码批次数；无commit/push。
- Next action: 沿实验工作流继续完成模块和函数判断，再按已记录的孤立模块DFS顺序检查。下一源码动作必须另提案并获真实reviewer批准、实施后验收；不重开REQ003及REQ005/006/007。全仓语义审查尚未完成，持续目标保留。
- Protocol: 末尾E0088与快照一致，next_writer=codex。B22不再等待E0087反馈；11983 reviewer会话保持打开。

B21 已关闭记录（以下不是当前在途请求）：

- Request: `REQ-019` / `B21-UNUSED-MESSAGE-HELPERS`：`CLOSED / ACCEPTED`，真实reviewer E0086。当前无在途请求，不再等待E0085反馈。

- Scope: msg_dispatcher.py的msg_read/msg_filter完整21行原字节迁到同目录msg_dispatcher_unused.py，新增35项回归测试。MsgDispatcher类、原imports、其余字节及message_handler/scheduling_table_event/MessagePipe旧runtime借用组件保留。
- Evidence: Codex直接解析reviewer的/tmp/b21_review_final/b21only.xml：35passed；thirteen.xml：259=254passed+5同名同异常collector失败，0error/skip，十三模块计数一致。3help和B13-B21探针本轮日志同目录；Codex自身完整回归/tmp/scheduler-b21-regression-mnj4yb7z。
- Bytes: 源0增/21删、1717B，SHA0c9df6adcec6cdda83d285d7e865097e74b9103f513eeadb97f2ac4a3ac3fbcd，等于HEAD删两段的完整字节重建。两归档连续块各一次；168非本批SHA无豁免，验收后全171代码SHA再核一致。
- EOF: 按E0084 M5完整字节断言保护，实际移除了工具多加的一个尾LF并fsync，仍保留原无尾LF；预期不变。B20已验收的唯一git diff --check rc2警告sim_main.py:677原文不变，不消原空白，不虚报全绿。
- Compatibility: 完整公共集合5→3恰失两名；旧导出和函数pickle路径退休是真实兼容变化、仓外未知，新归档身份往返通过。原Queue/Buffer/Data行为、异常和部分写入保留；其他模块同名msg_filter是参数/局部字典，不是函数引用。
- Ledger/recovery: 原23862/72773B前缀SHA保留；move46→48新增2×14且8旧坏行原样，reachable266→268新增2×10全10列。B21 JSON validation.recovery存三代码文件限定压缩patch及extract/check/restore命令，持久解码SHA/路径正确、reverse--check0，未恢复。保护区/依赖/HEAD/index不变。
- Protocol: 末尾E0086与快照一致，next_writer=codex。B11-B21共十一批已验收但未提交推送；验收不等于提交授权。原协议事件不改写，REQ-003取消和REQ-005/006/007暂缓裁决不变。
- Next action: 返回sim_main尚未完成语义审查的依赖继续只读DFS；任何新源码动作另提案，由真实reviewer先批准、实施后验收。全仓语义审查尚未完成。最初dfs-exploration-inventory-20260906.json保留历史137Python快照，当前明确代码范围155Python+16shell，不把语法覆盖当全仓语义审查完成。

B20 已关闭记录（以下不是当前在途请求）：

- Request: `REQ-018` / `B20-UNUSED-SIM-CACHE-HELPERS`：`CLOSED / ACCEPTED`，真实reviewer E0082；没有在途请求。E0078批准迁移，E0080批准缺LF单字节恢复，均已执行并通过独立验收。
- Scope: sim_main.py四个完整CSV/缓存助手共74行原样迁入同目录sim_main_unused.py。其余16函数/imports/常量/类内/旧入口及保护区保留；四个旧导出和函数引用pickle路径退休是真实兼容变化，仓外使用未知。
- Evidence: Codex直接解析reviewer的/tmp/b20_review_final/b20only.xml：38passed，0fail/error/skip；twelve.xml：224=219passed+5同名同异常collector失败，0error/skip，十二模块数量逐项一致。三help与B13-B17+B18探针本轮日志同目录；Codex自身回归日志/tmp/scheduler-b20-regression-lomypf0o。
- Bytes: 源0增/74删、30583字节，SHA 1d04e4aa038b63bc289834fc350442d9ff2b929faabb42a042deb7803bc86520，与HEAD删四段重建全字节相同。四归档块各连续一次，公共集合121→117恰失四名；旧pickle失败和新pickle身份已测试。166非本批SHA无豁免，全169代码文件验收后再核未变。
- EOF: 工具丢失的1字节LF已按E0080断言保护恢复。git diff --check仍rc2，唯一sim_main.py:677 new blank line at EOF；原HEAD704/731分隔空行保留后成为现677/678，E0082明确接受此警告，不为消警告改原字节。不是diff-check全绿。
- Ledger: 原22038/72055字节前缀SHA完整保留；move42→46新增4×14且8历史坏行原样；reachable264→266新增2×10且全表10列。
- Recovery: B20 JSON validation.recovery保存三个代码文件限定压缩patch、SHA及extract/check/restore命令。持久解码后SHA/路径正确，reverse--check0，未执行恢复。保护区、依赖、HEAD/index未变。
- Protocol: 末尾E0082与快照一致，next_writer=codex。B11-B20共十批未提交推送，验收不等于提交授权；不要再等待E0081反馈。
- DFS continuation: B20后继续沿sim_main依赖读了Spec、MsgDispatcher、DataPipe/TriggerPipe与placement。类整体保留，位置映射有旧runtime真实导入/调用；msg_read/msg_filter只列待核候选，须补完整属性/动态/文档引用和原行为证据，不凭初步名字扫描迁移。不重开REQ-003和REQ-005/006/007，不修顺带发现的旧逻辑问题。
- Next action: 当前B20收尾已完成；后续沿消息组件路径继续只读核查，任何新源码动作另提案并由真实reviewer批准。全仓语义审查尚未完成，持续目标保留。

B19 已关闭记录（以下不是当前在途请求）：

- Request: `REQ-017` / `B19-UNUSED-SLACK-GRAPH-HELPERS`：**CLOSED / ACCEPTED，真实 reviewer E0076**。验收补核已完成，无在途请求。
- Scope: slack_estim 的 build_score_dict_ref_flops/get_chains_info 共12行、graph_breakdown 的 sort_chains_by_ddl_flops 共2行，原字节迁入各自同目录_unused。原imports、活绘图、图分解、自带测试/main和类内均保留；不是逻辑重写。
- Evidence: 新增29项全过；完整十一套186=181passed+5同名同异常collector失败，0error/skip。Codex直接解析reviewer的 /tmp/b19_review_junit.xml 与 /tmp/b19_new_only.xml，模块数量和失败逐项一致。reviewer补跑3help及B13-B16探针，日志/tmp/b19_review_probes/。
- Bytes: 两源纯删12/2行，剩余SHA bda2da01c0bb32873d021dd32efc0debb5017333221baa3bd5cfbf7f17cc39c5 / 486b438fa3325f8d035ef91ca63518649986e21aee74cb761092908c579cdbd1。graph工具补LF确已按E0074授权、完整bytes断言保护truncate还原。162非本批全部原样无guard豁免，本批5文件哈希也再核一致。
- Compatibility: 公共名slack105→103减2、graph6→5减1；三个旧函数导出和旧函数pickle路径真实退休，新位置身份往返可用。非空排序IndexError、字典部分写入等原行为保留，仓外调用未知。
- Ledger: move本批20734B前缀保留，39→42行，新增3×14；reachable本批71003B前缀保留，261→264行，新增3×10。8个历史坏行原样保留。E0076原文旧数字以reviewer补核和本节为准，不回改历史事件。
- Recovery: B19 JSON validation.recovery保存五代码文件限定压缩patch、SHA及extract/check/restore命令；从持久JSON解码再核五路径和SHA，reverse--check0，未恢复。保护区、依赖、HEAD/index不变。
- Protocol: 末尾E0076及快照一致，next_writer=codex；独立验收已结束，不再等待E0075反馈。B11-B19共九批未提交推送，提交仍需用户指令。
- Next action: 本轮B19收尾完成；继续从task_cfg/slack/graph返回父调用路径，审查剩余实际依赖，不重开REQ-003取消项和REQ-005/006/007暂缓项。历史入口中不确定的依赖保留；任何新源码动作另提案。全仓语义审查尚未完成。

B18 已关闭记录（以下不是当前在途请求）：

- Request: `REQ-016` / `B18-UNUSED-E2E-EVENT-GENERATORS` — **CLOSED, ACCEPTED (E0072)**。REQ-008~015 已验收关闭；REQ-003 取消、REQ-005/006/007 裁决不变。
- State 补充（E0072 后全量补核，codex 请求）：B18 新测试单独跑 **25/25**（0 fail/error/skip）；按 JSON regression.command 同款**十套完整执行 157=152 passed+相同 5 collector failed**（0 error/skip，FAILED 名单逐项一致）；恢复补丁解码 SHA 与声明一致、reverse --check rc=0 未执行恢复。
- Scope: e2e_latency.py仅删除naive_period_event_gen/e2e_var_sim两完整函数26行，原字节归入同目录e2e_latency_unused.py；现用六函数/imports/类内/配置/历史文档与保护区保留。归档明确借用活jitter_gen_biside，不声称导入隔离。
- Evidence: RED1failed/24deselected原因正确，GREEN25passed；EOF校正后十套157=152passed+5同名同异常collector失败（31.35s），0error/skip；三入口help及B13-B16收益探针通过。81→79完整公共名双导入顺序通过，13组原行为每组双跑，J7旧函数pickle真实字节加载失败、新函数往返identity通过。日志/tmp/scheduler-b18-regression-oq0al2g0。
- Byte/protocol: 精确0增/26删，剩余SHA3043c818e36dbdfcee8ee50573c5274a8549fbb1413e4cb55cde07cc0e990a7e，含原无EOF换行。工具曾补1字节LF，经E0069暂停/E0070独立批准后限定truncate还原，未修改预期。E0067函数引用pickle勘误维持：退休是真实兼容变化，仓外使用未知。
- Hash scope: 161基线全量核对，160原样；唯一例外为reviewer自己新增守卫67号白名单，E0068确认独立归因，去该行重建精确回原f7582881...。原基线不重录、不谎称161未变；当前148Python可解析，保护区/requirement零diff，HEAD/index不变。
- Ledger/recovery: move追加2行14列，39行={14:31,15:3,16:4,17:1}，8旧坏行原样；reachable追加2行10列，261行全10列。原19755/70317字节前缀保留。B18 JSON validation.recovery存三代码文件限定压缩patch、SHA及extract/check/restore命令，持久报告解码reverse--check0，未实际恢复。
- DFS: 已读任务模型依赖和slack_estim、graph_breakdown、链求解器顶层函数、loadA配置；活用接口与类保留。graph示例/排序助手有原有异常，只记录不修；两个slack顶层函数尚待继续审查，不混入B18。
- Next action: B18已在E0072验收关闭，新增25/25及完整十套157=152+原5失败、恢复检查补核已由Codex直接读取reviewer XML确认，无在途请求。后续继续沿slack/graph实际依赖审查待定助手和历史入口，不改类内、不重开取消/暂缓项；新源码动作另提案。B11-B18均未提交，提交推送仍等用户指令。

B17 已关闭记录（以下不是当前在途请求）：

- Request: `REQ-015` / `B17-TASK-CFG-LEGACY-GRAPH` — **CLOSED, ACCEPTED (E0064)**。REQ-008~014 已验收关闭；REQ-003 取消、REQ-005/006/007 裁决不变。
- State: `ACCEPTED`（验收明细存档于 E0064）— I1-I6 经 reviewer 独立复测逐项满足：I1 numstat 0/206、剩余 SHA=6194c815…f14b、**剩余 task_cfg == HEAD 删两片段的字节级重建（True）**、归档两片段连续块各恰 1 次；I2 公共名 119→118 恰失 creat_jobTask_graph、18 新测试 RED→GREEN、11 组基线逐字段；I3 九套合跑 junitxml 132=127+5、FAILED 名单逐项一致、3 入口 rc=0、B13/B14 探针通过（收益不回退）、requirement+保护区零 diff、B11-B16 锚点原样；**I3 补充（E0064 后全量补核）：non_b17_source_sha256 全部 159 条已逐一 sha256 遍历——OK=159/MISMATCH=0/MISSING=0（此前 E0064 正文为 5 锚点抽查，全量补核按 codex 请求完成，结论不变）**；I4 借用者/old_num_hp 更正已落实；I6 csv.reader 双前缀保留（18839/69612）、新行 2×14/2×10、8 历史异常不修。恢复补丁解码+SHA 一致+reverse --check rc=0 未执行。七批（B11-B17）均未提交、未推送（commit 待用户指令）。
- 原批准明细（存档）：拟将 task/task_cfg.py L24-82（59行注释旧绘图）+ L551-697（147行 creat_jobTask_graph 含 return 后历史说明）共 206 行原字节迁入同目录 task_cfg_old.py；现用 vis_task_static_timeline/creat_physical_graph/load_taskint/redist_ert_dll/plot_workflow_g 与 imports 全保留。**E0062 更正**：load_taskint/redist_ert_dll 的借用者确为 `old/allocator_agent.py`（L561/563 import + L577/580 调用，sed 直读核实）——E0061 (a) 的"零借用"表述作废，保留裁定不变且依据更充分；old_num_hp 实位于 approach_initiator.py:20（REVIEW 不混入）。**工具教训（binding）**：本环境 grep=ugrep 7.8.4，带 `--include` 的递归查询返回假阴性；B13-B16 已用 python os.walk 全量（318 文件）补验 import 级引用全为 0（B15 的 3 条为子串误报），各批结论维持；零引用核查权威口径 = git ls-files 过滤工作树不存在条目（B12 已删根文件 appoach_plot6.py/approach_util33.py 为 index 幽灵条目，须剔除）+ B11-B18 已批准新目标 = **164 份源码/脚本**；318 文件 os.walk 含无关 untracked，降为保守旁证不作验收范围）——已按权威口径复核 B13-B16：命中恰在各批归档定义文件与配套测试内，活区与非本批归档零引用，各批结论维持；禁用带 --include 的递归 grep 作为唯一依据。RESULT 验收 = I1 纯删除 206 行+剩余 SHA=6194c815…f14b / I2 RED→GREEN+11 组基线+完整集合 119→118 恰失一名 / I3 八套合跑 114=109+5 名单不变、159 份非 B17 SHA 不变 / I4 退休边界+两处精确化固化 / I5 异常即暂停 / I6 csv.reader 账本（前缀 18839/69612 保留、8 历史坏行不修）。恢复：仅反向 206 行+移除归档/测试，补丁 reverse --check，禁整文件 checkout。
- Scope: task/task_cfg.py的59行旧绘图注释与147行完整creat_jobTask_graph已原样迁入同目录task_cfg_old.py，共206行。现用图生成、绘图函数、imports、旧allocator借用函数和保护区全保留；old_num_hp仅REVIEW，不入本批。新test_old_task_cfg.py覆盖18项，原函数注解/defaults与历史异常保持。
- Evidence: 新18项全过，联合132=127passed+5相同collector失败（25.35s），0error/skip；3help均rc0，B13/B14行为探针通过。完整公共名119→118恰失旧函数，11组原图/PDF/异常与原注解一致。源精确0增/206删，剩余SHA6194c815...f14b；归档SHA06a38533...0356。159非本批SHA不变，146Python AST可解析，保护区/依赖零diff。日志/tmp/scheduler-b17-regression-_sfebunx。
- Ledger/recovery: move追加2行14列、reachable追加2行10列，18839/69612原字节前缀完整保留；当前move={14:29,15:3,16:4,17:1}仅8历史异常，reachable={10:259}。JSON validation.recovery持久化压缩限定补丁及提取/检查/恢复命令；解码SHA核对及reverse --check通过，未实际回退。
- Next action: REQ-015已在E0064验收关闭，无在途请求。继续沿task_cfg的任务模型/队列、slack_estim、load_cfg依赖只读审查，新的源码动作另提案；旧取消/暂缓项不重开。B11-B17均未提交推送，提交仍等用户指令。

B16已关闭记录（以下不是当前在途请求）：

- Request: `REQ-014` / `B16-UNUSED-EQ-QUANTILES` — **CLOSED, ACCEPTED (E0059)**。REQ-008~013 已验收关闭；REQ-003 取消、REQ-005/006/007 裁决不变。
- State: `ACCEPTED` — H1-H6 经 reviewer 独立复测逐项满足：H1 剩余 approach_Eq SHA=bb2e6522…dc8（**64 位正确值；E0057 的 65 位串系 reviewer 记录笔误，本事件正式勘误，以 64 位+精确 9 行纯删除为准**）+ 决定性字节证明（剩余 Eq == HEAD 删 9 行的字节级重建，True）+ 归档两片段连续块各恰 1 次；H2 24 新测试 RED→GREEN、18 行为样本逐字段、fresh 双导入顺序恰失两名且集合一致（口径注【E0059 后经 codex 澄清、reviewer 独立复核更正】：not startswith('_')=45→43、not startswith('__')=49→47；两口径差集恰为 4 个私有 SciPy 别名 _scipy_expon/_scipy_norm/_scipy_truncexpon/_scipy_truncnorm——E0059 原注猜测的 math/find_legal 实为 public、Eq 无 annotations 名，猜测作废）；H3 八套合跑 junitxml 114=109+5、FAILED 名单逐项一致、3 入口 rc=0、ref_alloc stdout 逐字节一致、157 份非 B16 SHA 原样、B13/B14 收益不回退；H4 两旧名与旧 pickle 退休边界固化；H6 csv.reader 前缀 18287/68910 保留、新行 1×14/2×10、8 历史异常不修。**验证器纠错披露确认**：SequenceMatcher auto-junk 误报仅影响临时验证器（发生在功能验证全过之后），禁用启发式+精确重建后重跑通过，无源码/测试为过验证而修改。六批（B11-B16）均未提交、未推送（commit 待用户指令）。
- Next writer: `codex`。REQ-014已验收关闭；本轮不开始下一源码批次。持续目标后续回到approach_sched顶层函数及动态绑定继续只读审查，新源码动作须单独提案获批；提交另等用户指令。
- B13 实施进度：utils纯删除41行且剩余SHA与预期完全一致，原40行簇完整保留。9项新测试通过；五套联合回归53 passed/5原有failed，三入口help通过；152份非B13源码/脚本和requirement.txt原样。fresh process验证utils不加载h5py，显式归档导入仍可用。未提交、未推送。
- B12 实施进度：两份完整文件已归入 old/，仅脚本一行 import 更新；新7项测试通过，B10/B11/collector为37 passed及相同5个既有失败，三入口正常，150份非B12源码/脚本原样。根旧模块名不复用，旧根模块名pickle不承诺兼容；新入口为 python -m old.appoach_plot6。
- Source changes: B16包内Eq仅移出norm_inv_cdf/exp_quantile两完整函数共9行，原实现剩余SHA bb2e65224f12fe2555c2f12567276a110d8d409808a180f4d4282b86a289cdc8。新归档SHA85092d18...3bf4，新测试24项全过；其余43公共导出及根/包身份、原imports、类内代码、全部既有测试保留。旧双路径函数名及旧函数pickle引用退休，仓外兼容不承诺。B11-B16未提交推送。
- Protocol v1.1: E0057批准→Codex实施→E0058 RESULT→真实reviewer E0059 ACCEPTED，H1正式勘误已补齐。H2口径也经双方核对：公共名45→43；非双下划线49→47，多出的只是四个私有SciPy别名。八套114项=109passed+5原有collector失败，3help/ref自测通过；157非本批SHA原样，144Python AST可解析。末尾E0059及快照一致，没有新请求或越权事件。
- Workspace note (2026-09-03): per user instruction, closed events E0001–E0014 (REQ-001 protocol bootstrap, REQ-002 B9) archived verbatim to `AGENT_DIALOGUE_archive.md`; main dialogue keeps protocol body, event numbering continues globally from E0015
- Ledger result: B16追加move 1行14列/reachable 2行10列，原18287/68910字节前缀完整保留；csv.reader当前move={14:27,15:3,16:4,17:1}仅8历史异常，reachable={10:257}无异常，旧行不修。B16 JSON存三个准确代码文件的限定恢复补丁，Git最小差分确认源0增/9删，reverse --check通过但未执行；无整文件checkout。

`E0014` closes REQ-002. The moved demo preserves the reviewed behavior, and the final ledger diff contains no unrelated rewrite. Monitoring remains active because the previously agreed refactor discussion still has further candidates; completing this one request is not the global termination condition.

Background monitoring:

- The former Windows watcher PID `35404` is no longer running. Its log at `C:\Users\diyuf\.codex\state\scheduler-agent-dialogue\watch.log` was last updated on 2026-09-03 09:44.
- No `scheduler-agent` automation configuration exists under the current `$CODEX_HOME/automations` directory. Earlier claims that a five-minute heartbeat remained active are stale.
- 2026-09-06 起，用户授权 Codex 直接唤醒 WSL 的 `clean` reviewer，并要求保持一个交互进程。首次 resume 仅做无工具的历史核验：会话 ID `cb4d63c6-e3a3-429c-8dd8-a00f301fda43`，角色、审计 worktree、E0027/E0029/E0031 和 HEAD 均与磁盘及用户提供的历史一致。
- 当前持续交互通道为 Codex terminal session `11983`，由已核验的 session ID 恢复；后续提案/结果复用它输入并接收反馈。原有 PID 1399 的界面观测为 idle，本轮不往该旧界面另发消息。此通道不是定时任务，也不等于永久后台唤醒服务；断线后须重新核对进程与历史，不能声称仍在监控。
- 启动目录 `/home/zhangchg/git_repo/scheduler` 仅用于用户明确授权的会话恢复，所有仓库读写、测试和日志仍限定在 audit worktree。Codex 只写 codex 槽位，reviewer 自行写自己的裁决。

## Scope

- Source repo: `/home/zhangchg/git_repo/scheduler`
- Active cleanup worktree: `/home/zhangchg/git_repo/scheduler-audit-20260612`
- Integration worktree: `/home/zhangchg/git_repo/scheduler-integration-20260612`
- Source branch snapshot: `test_pipeline @ d5bfda5`
- Archive tag: `archive/test_pipeline-20260612`
- Archive branch: `archive/test_pipeline-20260612-branch`
- Mainline basis for now: `master`

Rules still in force:

- Do not operate directly on `/home/zhangchg/git_repo/scheduler`; the 2026-09-06 user exception permits launching/resuming the existing `clean` reviewer there, not repository reads/edits or Git operations.
- Work only in audit/integration worktrees.
- Ignore untracked files unless the user explicitly brings them into scope.
- Focus on git tracked / git cache state.
- Do not execute old `P1-REMOVE-*` or `P1-CACHE-*` actions.

## Current audit branch state

The B0 source changes, Phase 1 analysis artifacts (`cleanup/`), review packets, and the two global status files were committed to the audit branch on 2026-06-16 (first commit beyond `test_pipeline`; see `FILE_ADJUSTMENT_RECORD.md`).

Current committed checkpoint after B10 acceptance and the REQ-005 verdict:

- Commit: `a1d933b` (`test: checkpoint binpack baseline and review protocol`), 8 reviewed files; worktree clean immediately after commit.

- B8 source change: `approach_setup.py` step 4-6 now uses `sim_main.py::init_sched_components(...)` to create the shared global_sched component context and run `perform_bin_packing` through a small pack handle.
- B8 status docs: `REVIEW_PACKET_BATCH_B8-INIT-SCHED-COMPONENTS.md`, `CLEANUP_STATUS.md`, `FILE_ADJUSTMENT_RECORD.md`, and the Gurobi troubleshooting row in `CLAUDE.md`.
- B8 validation: import probe + 3 help commands PASS; motiv case1 (`--num_hp 3 --case1_ratios 0.7`) full run rc=0 after fixing WSL Gurobi HostID via bond0.
- Validation artifacts from `motiv_exp_results_b8verify/` were removed before commit.

B10 result:

- Added `test_binpack_pipeline_contract.py`: one fast B8 closure/21-argument forwarding contract and two real pipeline characterization cases.
- Added `cleanup/reports/b10-binpack-behavior-baseline.json`: Split and fixcore-Repack golden hashes, environment/config metadata, packing-call traces, and three readable timing samples per scenario.
- All 53 packed PIDs contribute `name`, `ert`, and `ddl` to the canonical signature. Split and Repack each matched across two independent worker processes and separate temporary directories.
- Gurobi preflight created a real model with 11.0.3 and the license at `/home/zhangchg/gurobi1003/gurobi.lic` (expires 2027-03-14).
- Validation: new pytest `3 passed`; 12-module import probe PASS; all three entry-point `--help` checks PASS; `py_compile`, JSON parse, and production zero-diff checks PASS.

Current refactor-sequence progress:

1. Behavior baseline: COMPLETE and reviewer-accepted (`B10`, E0025).
2. Binpack initialization context: COMPLETE in `B8`; B10 now protects its forwarding contract.
3. Narrow `perform_bin_packing`: NOT STARTED; earlier move proposal REQ-003 was cancelled by user.
4. Separate Repack event/queue advancement from decisions: RESOLVED — DEFER_KEEP_AS_IS (REQ-006, reviewer verdict E0029). The primary guided Repack path bypasses greedy packing under `USE_FIXCORE_REPACK=True`; the module remains reachable via `Bp_scratch.json` (Phase-1 KEEP, documented alternate) and the disabled legacy fallback. KNOWN ISSUE (recorded, not fixed): scratch smoke rc=0 but 0 PIDs placed / 10 bins / `No more bin can be created` (pre_alloc_new.py:243) / hardcoded `repack_success=True` — rc=0 is NOT evidence of scratch correctness. Re-evaluate scratch within the future user-initiated repack rehabilitation design (shares push_task_into_bins_new).
5. pre_alloc_new compute-vs-write split: RESOLVED — DEFER_WITH_REPACK (REQ-007, reviewer verdict E0031). Split decision folded into the future user-initiated repack rehabilitation design (with E0029 K3 scratch clause); characterization tests = mandatory FIRST phase of that rehab (baseline-before-fix), NOT now. ROOT CAUSE recorded (independently verified): bin_ops.py manual_defined_reservation/all_isolation/static_1_bin exhaust the generator via list() then return it; pre_alloc_new.py:240 next() -> StopIteration -> 'No more bin can be created' (:243) — this is REQ-006's scratch 0-PID root cause. Blast radius: scratch chain + disabled greedy fallback ONLY (live path imports glb_alloc_new2 at global_sched_alloc.py:32 but never calls it; coleasing uses fresh bin_iter_list at :335; B10 goldens green).
6. Scheduling-table adjunct split: CLOSED with `KEEP_AS_IS` (E0027). Event handling and the debug entry were already moved in B6/B9; the remaining live bin helpers and `slack_estim.py` plotting stay in place.
7. Move `approach_*` into a package with compatibility exports: COMPLETE / ACCEPTED（reviewer E0037），未提交。七个活跃实现已归入 approach/，5,181 行原样保留，根入口兼容。reviewer 独立合跑37 passed/5 failed（29.45秒）：新增32项与B10 3项全过，collector 原有5 failed/2 passed不变。approach_util33.py留待历史文件批次。KNOWN BOUNDARY（C3）：旧 pickle 可在迁移后的 audit 读取；新 approach.* pickle 不保证被未迁移的 test_pipeline 读取。
8. Historical filenames and naming cleanup: ACCEPTED（B12，E0041）。旧 approach 双文件已完整归档；五份 *_old.py / *_unused.py 为已核对的符号片段，按用户规则留在源目录；空 unused_fun.py 暂留。本轮可执行项已收尾，REQ-003取消与Repack暂缓项不被当作完成；按持续目标进入后续只读探索。

后续探索：初始dfs-exploration-inventory-20260906.json仍是B13前137Python/519顶层符号/三入口保守72文件的历史快照。当前明确范围144Python及16shell，语法覆盖不代表全仓语义审查完成。utils→global_var→Eq→ref_alloc_search已完成本轮分离/保留判断，B16在E0059验收。approach_def九个顶层函数已阅读并追踪；set_miss_disabled虽无AST裸名调用，B11通过字符串参数实际测试；get_miss_disabled/get_realloc_disabled依赖模块可变状态，简单复制到归档会读到旧布尔值，实测证实，暂留。五个类按用户规则不拆内部方法。下一步返回父approach_sched的顶层调度函数和动态绑定路径。

当前目标继续有效：本轮B16实现、测试与独立验收已完成，属于实际进展；全项目模块审查仍未完成。不确定内容保持原样，新的源码动作先提案、实施后验收。reviewer11983仍为同一打开会话，未新建会话/定时任务，没有提交推送。

No changes were made to `/home/zhangchg/git_repo/scheduler` or `test_pipeline`.

Committed B0 source changes (recovery: `git checkout archive/test_pipeline-20260612 -- <path>`):

- `B0-SHARED-NODE` (P1-NODE-001, archive-only): removed empty `package.json` + `package-lock.json`.
- `B0-SHARED-DEP` (P1-DEP-001): updated `requirement.txt` (+8 deps).

## Executed batches

### B0-SHARED-NODE

- Decision: `P1-NODE-001`
- Status: executed and accepted by user on 2026-06-13.
- Changed files:
  - removed `package.json`
  - removed `package-lock.json`
- Strategy: `archive-only`
- Recovery:

```bash
git checkout archive/test_pipeline-20260612 -- package.json package-lock.json
```

Validation:

- `scripts/motiv_exp_runner.py --help`: PASS in `gurobi`
- `scripts/abla_exp_runner.py --help`: PASS in `gurobi`
- `main_approach.py --help`: PASS in `gurobi`

### B0-SHARED-DEP

- Decision: `P1-DEP-001`
- Status: executed and accepted by user on 2026-06-13.
- Changed file:
  - updated `requirement.txt`
- Added dependencies:
  - `gurobipy`
  - `h5py`
  - `networkx`
  - `psutil`
  - `pyinstrument`
  - `pytest`
  - `tdigest`
  - `tqdm`
- Kept review dependencies:
  - `pyyaml`
  - `plotly`
  - `bokeh`
- Recovery:

```bash
git checkout archive/test_pipeline-20260612 -- requirement.txt
```

Validation:

- `scripts/motiv_exp_runner.py --help`: PASS in `gurobi`
- `scripts/abla_exp_runner.py --help`: PASS in `gurobi`
- `main_approach.py --help`: PASS in `gurobi`
- `import mapper.mem_planner`: PASS after the user installed `tqdm`

## Completed preflights

### B1-REVIEW-TRIAGE-PREFLIGHT

Status: preflight complete. No cleanup executed.

Included decisions:

- `P1-DEP-002`
- `P1-REVIEW-DOCS-001`
- `P1-REVIEW-RUNTIME-001`
- `P1-REUSE-001`
- `P1-SYMBOL-001`

Result:

- `P1-DEP-002`: keep `bokeh`, `plotly`, and `pyyaml` under review.
- `P1-REVIEW-DOCS-001`: protected docs are KEEP, not cleanup candidates.
- `P1-REVIEW-RUNTIME-001`: runtime-adjacent files require targeted tests before cleanup.
- `P1-REUSE-001`: reuse candidates are preserved by default.
- `P1-SYMBOL-001`: symbol candidates are record-only and do not authorize symbol deletion.

Report files:

- `REVIEW_PACKET_BATCH_B1-REVIEW-TRIAGE-PREFLIGHT.md`
- `cleanup/reports/batch-B1-REVIEW-TRIAGE-preflight.md`
- `cleanup/reports/b1-review-triage-actions.csv`

## B2-RUNTIME-TEST-INVENTORY (in progress)

### Understanding phase — COMPLETE (2026-06-16)

Goal: separate NEW runtime (KEEP) from OLD/dead before any symbol-level action.

Key finding (corrects `CLAUDE.md` / `deprecated_code.md`):

- `sched/scheduler_agent.py` and `sched/monitor_agent.py` are marked "deprecated, replaced by `approach_sim.py`/`approach_collector.py`". This is TRUE only for the **simulation-loop** role. They are still **LIVE** dependencies of the config-generation path:
  - `monitor_agent.get_target_bin_id`, `monitor_agent.get_rsc_2b_released` ← imported by active `sched/pre_alloc_new.py` (repack path).
  - `scheduler_agent` Scheduler class + sim-loop helpers (check_miss/check_complete/data_pipe_read/pendingToReady/load_sched_tab) ← imported by `sched/global_sched.py` + `allocator_agent.py`.
  - => **File-level deletion breaks the active repack path. Only symbol-level dead/live split is safe.**

Boundary (verified by import-chain tracing + static grep):

- NEW (active, KEEP): `main_approach.py`; `approach_setup.py`; `sim_main.py::perform_bin_packing`; `approach_sim.py` + `approach_def/sched/initiator/Eq/collector.py`; `sched/global_sched.py` (coleasing_alloc_cluster, push_task_into_bins_new); `sched/pre_alloc_new.py`.
- `approach_sim.py` (NEW sim backend) imports NONE of scheduler_agent/monitor_agent/allocator_agent.
- MIXED (need symbol-level split): `scheduler_agent.py`, `monitor_agent.py`, `allocator_agent.py`.
- 23 runtime-adjacent REVIEW candidates are NOT in the active path: 6 package `__init__.py` (real packages → KEEP), 13 standalone analyze/run scripts (0 importers), 4 standalone plot/test.

### User-approved execution order

1. **B** — symbol-level live/dead split of `scheduler_agent.py` / `monitor_agent.py` / `allocator_agent.py` (the high-value, higher-risk target). For ambiguous symbols, **ask the user directly** (code author) for priors.
2. **A** — the independent scripts (analyze/*, run/*, plot/test pair) via `archive-only` / `move-reference`.

### Symbol-level split — COMPLETE (2026-06-16, AST-based)

Real finding (AST, not text grep — `from sched.sched_fn import *` creates false-definition locations):

- The dead root is the **OLD SIMULATION LOOP SCC**, not `scheduler_agent.py` per se. `Scheduler` class is **LIVE** on the active repack path (`create_common_scheduler_elements` instantiates it → fed into `perform_bin_packing`).
- Dead cluster: `sim_main::main()` + `others()` → `allocator_agent.{glb_sched,cyclic_sched,sched_step,period_boader_display,AllocatorInt}` → `Scheduler` stepping methods → `sched_fn`/`state_trans`/`sched_utils` standalone fns. Reachable only from the superseded `sim_main::main()`.
- `check_miss`/`check_complete`/`pendingToReady` have TWO forms: LIVE standalone fn (`state_trans.py`/`sched_utils.py`) + DEAD `Scheduler` class method (0 `.method()` calls).
- `preprocess_args` (sim_main L692) is ACTIVE (approach_setup.py:106), interleaved between dead `others`/`main` — must not be swept up.

### B2-MOVE-DEAD-SIM-CHAIN — EXECUTED 2026-06-16 (regression gate PASS)

Strategy: **MOVE (move-reference), not delete**. All 3 decisions executed together (001+002 code-coupled via sim_main.py top-level import).

| Decision | Object | Target | Status |
|----------|--------|--------|--------|
| B2-MOVE-001 | `allocator_agent.py` (whole file, 825 lines) | `unused/allocator_agent.py` | ✅ executed |
| B2-MOVE-002 | `sim_main.py::others()`+`::main()`+`if __name__` (241 lines) | `sim_main_unused.py` (273 lines) | ✅ executed (pure deletion, byte-identical surviving code) |
| B2-MOVE-003 | `scripts/repack_sweep.py` (whole file) | `scripts/old/repack_sweep.py` | ✅ executed |

Companion import cleanup done: deleted `sim_main.py` top-level L17 (allocator_agent) + L18 (discrete_event_sim) — both served only the moved functions.

Regression gate (gurobi env) — ALL PASS: 6-line import probe OK; `main_approach.py`/`motiv_exp_runner`/`abla_exp_runner` `--help` all PASS; `allocator_agent` no longer importable from root (expected).

Deferred (not done this batch):
- `Scheduler` class-internal dead methods (user: 先不管)
- `sched_fn`/`state_trans`/`sched_utils` dead standalone fns (import * tangle — next batch)

### B3-BINPACK-DEAD-MOVE — batch 1 EXECUTED (2026-06-16), batch 2/3 pending

**Batch 1 (group A + bound pair) — DONE, gate PASS:**
- ✅ B3-MOVE-002: pre_alloc.py → unused/
- ✅ B3-MOVE-004/005: gurobi_semi2Dclst_mapping*.py → packing_solver/old/
- ✅ B3-CLEAN-001: deleted global_sched.py:25 dead import
- ❌ B3-MOVE-001/003: **N/A** — bin_ops.old.py & message_handler_old.py are UNTRACKED ghost files in main working tree, NOT in test_pipeline (git cat-file confirmed). Out of scope per P1-GIT-002.

**Ghost-file discovery (important):** earlier bin_packing analysis ran grep in MAIN working tree, mixing untracked ghosts (bin_ops.old.py 775L, message_handler_old.py 246L) into the dead-code inventory. The "~2400 lines" estimate was inflated by ~1021 lines. Future analysis must run in audit worktree (cleanup basis). True tracked bin_packing dead code is smaller.

**Batch 2 (group B symbol-level) — DONE, gate PASS, byte-identical:**
- ✅ B3-MOVE-006: bin_select_new (L287-327, 41 lines) → pre_alloc_new_unused.py
- ✅ B3-MOVE-007: naive_iso (L280-315, 36 lines) → global_sched_unused.py
- ✅ B3-MOVE-008: commented test block (L627-749, 123 lines) → pre_alloc_new_unused.py
- Line numbers re-verified in audit worktree via AST end_lineno BEFORE slicing (naive_iso shifted to L280-315 due to CLEAN-001; main-tree analysis said L281-317 — would have mis-cut).
- byte-identical: pre_alloc_new.py +0/-164; global_sched.py +0/-37 (pure deletion, trailing newline matched test_pipeline).

**Batch 3 (group D docs) — DONE:**
- ✅ B3-DOC-001: deprecated_code.md §8 澄清（scheduler_agent/monitor_agent 部分活）+ 新增 §10 清理记录 + bin_ops.old 幽灵标注
- ✅ B3-DOC-002: spec §8.1.1/8.1.2/8.2/8.3 标注✅已解决（保留原文，保护路径只标注不删）

**B3 COMPLETE**: 7 executed (002/004/005/006/007/008/CLEAN-001) + 2 N/A (001/003 幽灵文件) + 2 docs. 全部回归门 PASS.

### 归档目录语义重分类（2026-06-16）— DONE

用户定义语义：`old/`=历史版本（被新实现取代）；`unused/`=独立功能暂无引用。按此重分类所有 B2/B3 归档：

| 类 | 位置 | 内容 |
|----|------|------|
| **old/**（历史版本） | `old/` | allocator_agent.py, pre_alloc.py |
| **old/**（符号级） | `*_old.py` | sim_main_old.py, sched/pre_alloc_new_old.py, sched/global_sched_old.py |
| **unused/**（独立功能） | `scripts/unused/` | repack_sweep.py |
| **unused/**（独立功能） | `sched/packing_solver/unused/` | gurobi_semi2Dclst_mapping.py, _mapping2.py |

语义约定已写入 `cleanup-policy.md`（slim + repo）。删除空目录：model/message/old、unused/(根)、scripts/old、packing_solver/old。回归门 PASS。改动未 commit。

pre_alloc_new.py 深度分析完成（内部调用图 + spec 覆盖核对）。Packet: `REVIEW_PACKET_BATCH_B3-BINPACK-DEAD-MOVE.md`，11 个决策：

| 组 | 决策 | 对象 | 风险 |
|----|------|------|------|
| A 整文件 | B3-MOVE-001 | `bin_ops.old.py` (775L) → `unused/` | 低（独立） |
| A 整文件 | B3-MOVE-002 | `pre_alloc.py` (592L) → `unused/` | 低（**绑定 B3-CLEAN-001**） |
| A 整文件 | B3-MOVE-003 | `message_handler_old.py` (246L) → `model/message/old/` | 低（独立） |
| A 整文件 | B3-MOVE-004/005 | `gurobi_semi2Dclst_mapping.py` + `_mapping2.py` → `packing_solver/old/` | 低（独立） |
| B 符号级 | B3-MOVE-006 | `pre_alloc_new.py::bin_select_new` (L287-328) | 低（bin_sel 留活） |
| B 符号级 | B3-MOVE-007 | `global_sched.py::naive_iso` (L281-317) | 低 |
| B 符号级 | B3-MOVE-008 | `pre_alloc_new.py` 注释测试 (L627-748) | 低（死注释） |
| C 配套 | B3-CLEAN-001 | 删 `global_sched.py:25` 死 import | 低（绑定 B3-MOVE-002） |
| D 文档 | B3-DOC-001 | 更新 `deprecated_code.md`（补全） | 低 |
| D 文档 | B3-DOC-002 | 更新 `spec §8.1`（标记已解决，保护路径） | 低-中 |

总清理 ~2400+ 行死代码 + 2 处文档同步。不动：test_mem_planner（保留）、single_turn_solver（不管）、参数语义错位（记录）。



Reports: `cleanup/reports/bin_packing_inventory.md` + `cleanup/reports/bin_packing_function_map.md`. Analysis only, no source change.

- spec (`binpack_solver_spec.md §2.1`) function hierarchy is **ACCURATE** — maps to live code. Trustworthy canonical map.
- LIVE (guided/scratch): `sim_main::perform_bin_packing` → `global_sched.{coleasing_alloc_cluster, push_task_into_bins_new}` → `pre_alloc_new.glb_alloc_new2` + `bin_ops` + `gurobi_MP_semi2DClst`.
- DEAD whole files (0 live deps, safe to move): `bin_ops.old.py` (775L), `pre_alloc.py` (592L), `message_handler_old.py` (246L), `gurobi_semi2Dclst_mapping.py` + `_mapping2.py`.
- DEAD import fossil: `global_sched.py:25 from sched.pre_alloc import glb_alloc_new` (imported, never called).
- DEAD function: `global_sched.py::naive_iso` (0 calls).
- 3 contradictions resolved (import ≠ used): mapping import commented out; pre_alloc glb_alloc_new dead-import; sim_main doesn't import pre_alloc.
- User decisions: test_mem_planner **KEEP**; single_turn_solver **LEAVE AS-IS** (broken branch).
- Doc: `guide/deprecated_code.md` INCOMPLETE (misses pre_alloc/message_handler_old/dead solvers); `doc/dev/` process docs candidates for archival.



Do not treat `scheduler_agent.py` / `monitor_agent.py` as REMOVABLE at file level. Any packet proposing file-level deletion of these is a violation; only symbol-level dead-function/method removal is acceptable.

### Inventory packet and lightweight validation — COMPLETE (2026-06-16)

Added B2 reports:

- `REVIEW_PACKET_BATCH_B2-RUNTIME-TEST-INVENTORY.md`
- `cleanup/reports/batch-B2-RUNTIME-TEST-INVENTORY.md`
- `cleanup/reports/b2-runtime-test-inventory.csv`

Additional B2 findings:

- Current runtime path remains `main_approach.py -> approach_setup.py -> sim_main.py::perform_bin_packing() -> approach_initiator.py -> approach_sim.py::run_simulation() -> approach_collector.py`.
- `scripts/test_alloc_lat.py` is blocked for direct execution because it hardcodes `/home/zhangchg/git_repo/scheduler` into `sys.path`, which would bypass the audit worktree.
- `test_event_update.py` and `test_mapping.py` need unresolved import triage for `approach_plot` before execution.
- No test deletion is authorized.

Validation:

- WSL startup probe: PASS.
- `main_approach.py --help`: PASS in `gurobi`.
- `python -m scripts.motiv_exp_runner --help`: PASS in `gurobi`.
- `python -m scripts.abla_exp_runner --help`: PASS in `gurobi`.
- Targeted imports for current runtime modules: PASS in `gurobi`.



### 2026-06-15 status-file role cleanup

- Created `CLEANUP_STATUS.md` as the canonical global status file.
- Downgraded `PHASE1_STATUS_FOR_NEXT_AGENT.md` to a compatibility pointer.
- Clarified that `FILE_ADJUSTMENT_RECORD.md` records the global action/change history.

### 2026-06-15 mandatory update rule

- Added the rule that every future action must update both `CLEANUP_STATUS.md` and `FILE_ADJUSTMENT_RECORD.md`.
- This applies to executions, preflights, environment checks, and status-only maintenance.
- Confirmed `PHASE1_STATUS_FOR_NEXT_AGENT.md` remains only a compatibility pointer to `CLEANUP_STATUS.md`.

### 2026-06-16 commit B0 + Phase 1 artifacts to audit branch

- Snapshotted the accumulated audit worktree state (B0 source changes + Phase 1 analysis + status files) into the first commit on the audit branch beyond `test_pipeline`.
- Follows the mandatory update rule; both global records updated as part of the commit.
- No source change in the commit action itself; `test_pipeline` and the integration worktree untouched.

## Decision status

Executed and accepted:

- `P1-NODE-001`
- `P1-DEP-001`

Active constraints:

- `P1-GIT-001`: use `master` as current mainline basis.
- `P1-GIT-002`: untracked files are out of scope.
- `P1-KEEP-001`: current KEEP seed is accepted.
- `P1-REUSE-001`: preserve reuse candidates.
- `P1-SYMBOL-001`: symbol candidates are record-only.

Keep under review:

- `P1-DEP-002`: `bokeh`, `plotly`, `pyyaml`.
- `P1-REVIEW-DOCS-001`: docs not covered by explicit KEEP/protected policy.
- `P1-REVIEW-RUNTIME-001`: runtime-adjacent files until targeted tests exist.
- B2 runtime/test inventory follow-ups:
  - `B2-RUNTIME-LEGACY-001`: keep `sched/scheduler_agent.py` until old imports are split.
  - `B2-RUNTIME-LEGACY-002`: keep `sched/monitor_agent.py` until old imports are split.
  - `B2-RUNTIME-TEST-004`: do not run `scripts/test_alloc_lat.py` until its hardcoded original-worktree path is fixed or isolated.

Not approved:

- `P1-CACHE-001`
- `P1-CACHE-002`
- all `P1-REMOVE-*` batches

## Protected paths

Do not delete or bulk-move these areas:

- `.claude/`
- `claude_talk/`
- `doc/spec/`
- `doc/guide/`
- `doc/dev/`
- tests covered by rejected `P1-REMOVE-B19-TESTS`

## Test environment

Repository checks must use the documented Conda environment:

```bash
conda activate gurobi
```

Preferred command shape from Windows/Codex:

```bash
wsl -d Ubuntu-20.04 -- zsh -ic 'cd /home/zhangchg/git_repo/scheduler-audit-20260612 && conda activate gurobi && PYTHONDONTWRITEBYTECODE=1 python main_approach.py --help'
```

System `python3` outside this environment is not valid for this repo.

## Recommended next action

2026-09-06当前行动以顶部协调区为准：B17已在E0064独立验收关闭，159非本批SHA也由reviewer补齐全量核对。approach_sched六个顶层函数及五策略动态绑定保留；collector/ref_tdigest保留完整类与测试入口；initiator五函数保留，old_num_hp仅REVIEW。task_cfg只是定向审查，不能宣称全模块或全仓完成。graph_scaling两种图表示不合并；已读task_agent顶层边界/独立加载入口及TaskQueue，类内不清理。下一步沿任务模型实际依赖继续，未明确用途的旧加载入口暂REVIEW，不提前判废。以下旧批次建议仅作历史参考，不能覆盖当前裁决。

**Coordination gate: checkpoint `a1d933b` is committed. `REQ-006` closed with `DEFER_KEEP_AS_IS`; `REQ-007` is a review-only classification of sequence item 5.** It authorizes no production changes.

**B10-BINPACK-BEHAVIOR-BASELINE IMPLEMENTED AND VERIFIED.** The two golden signatures now cover final task timing, so the Repack case can detect a broken ratioB deadline-window recalculation even when fixcore retains the Phase-1 bin layout.

**B8-INIT-SCHED-COMPONENTS EXECUTED AND VERIFIED**. The B8 commit is the current checkpoint on the audit branch.

B8 result:
- `sim_main.py::init_sched_components(...)` encapsulates the shared Scheduler/Monitor/msg_dispatcher/DataPipe + simulation-env setup used by the unified `global_sched` interface.
- `approach_setup.py::run_benchmark_setup_pipeline(...)` no longer unpacks and forwards those runtime-adjacent components manually; it calls the pack handle.
- Logic change: none intended. `perform_bin_packing(...)` still receives the same values, just through the closure.
- Gurobi diagnosis corrected: the PyCapsule failure was from WSL HostID mismatch, not license expiry. `gurobi-wsl-fix`/manual `gurobi_fix` creates bond0 with MAC `00:15:5d:80:30:e7`.

Next:
1. Wait for the reviewer to choose `DEFER_WITH_REPACK`, `CHARACTERIZE_BOUNDARY_FIRST`, or `DESIGN_SPLIT_NOW` for item 5.
2. Do not describe the split as behavior-preserving without executable coverage: `pre_alloc_new.py` has no tracked direct test, and its only live caller is the known-broken scratch/legacy Repack chain.
3. Any test addition, design document, repair, or refactor requires a later explicit proposal; production changes also require the user's design approval.

Do not start cleanup execution from old `P1-REMOVE-*` or `P1-CACHE-*` decisions.

## File roles

- `AGENT_DIALOGUE.md`: append-oriented Codex/reviewer handoff log and review gate.
- `CLEANUP_STATUS.md`: current global status and next-action entry point.
- `FILE_ADJUSTMENT_RECORD.md`: chronological global action/change history; include recovery commands for actual file changes.
- `REVIEW_PACKET_BATCH_*.md`: per-batch review details.
- `cleanup/reports/*`: machine-readable or supporting audit reports.
