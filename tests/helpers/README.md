# tests/helpers/ — 非自动收集的保留测试材料

- `test_mem_planner.py`：签名依赖手动喂参（pytest fixture 不可解析），内存规划手动场景，非自动收集。
- `test_event_update.py` / `test_mapping.py`：依赖已消亡模块 `approach_plot`
  （`update_processors_for_hyperperiod` 全仓无定义），属死模块遗留。
  按 E0126 "保留测试"边界移入此处（不删除、不自动收集）。
- 内存规划交互场景同此模式。
- `test_alloc_lat.py`：导入时直接读取缓存并运行仿真，不是可独立收集的单元测试；
  已移除原始 worktree 的硬编码路径，手工运行时始终使用当前 checkout。
- `optimizer/ops_test.py`：依赖可选 Torch 的交互式数学函数绘图器，模块导入即
  打开绘图界面，不作为 pytest 测试收集。
