import sys
import argparse
import copy


def _parse_extra_args(argv):
    """
    仅解析本脚本自带的参数，并返回 (extra_args, remaining_argv)
    这样可以把剩余参数交给原项目的 input_parser 解析，避免冲突。
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--repack_B_list", type=str, default=None,
                        help="逗号分隔的 B 列表，如: 0.55,0.60,0.70")
    parser.add_argument("--split_first", action="store_true",
                        help="先运行一次 split（only_split），再按多组 B 做 repack")
    extra_args, remaining = parser.parse_known_args(argv)
    return extra_args, remaining


def _parse_B_list(b_list_str):
    if not b_list_str:
        return []
    return [float(x) for x in b_list_str.split(",") if x.strip()]


def main():
    # 先取出本脚本的参数，再把其余参数交给项目原有解析器
    extra_args, remaining = _parse_extra_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + remaining

    from utils import input_parser
    from sim_main import main as sim_main

    base_args = input_parser()

    # 1) 可选先做一次 split（天然 only_split）：不进入 repack 分支
    if extra_args.split_first:
        split_args = copy.deepcopy(base_args)
        # 显式关闭 repack
        setattr(split_args, "repack_mode", False)
        if hasattr(split_args, "binpack_cfg") and isinstance(split_args.binpack_cfg, dict):
            split_args.binpack_cfg["only_split"] = True
        print("==================== Split phase (only_split) ====================")
        sim_main(split_args)

    # 2) 多组 B 的 repack 循环
    B_list = _parse_B_list(extra_args.repack_B_list)
    if not B_list:
        print("[repack_sweep] 未提供 --repack_B_list，流程结束。")
        return

    for B in B_list:
        repack_args = copy.deepcopy(base_args)
        # 显式进入 repack 模式
        setattr(repack_args, "repack_mode", True)
        setattr(repack_args, "exec_t_comp_ratioB", B)
        # 确保不会被 only_split 覆盖
        if hasattr(repack_args, "binpack_cfg") and isinstance(repack_args.binpack_cfg, dict):
            repack_args.binpack_cfg["only_split"] = False
        print("==================== Repack phase (B={:.2f}) ====================".format(B))
        sim_main(repack_args)


if __name__ == "__main__":
    main()


