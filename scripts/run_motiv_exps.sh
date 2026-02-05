#!/bin/bash
# Motivation Experiments Runner - Shell Wrapper
# 方便快速运行三个motivation实验

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 默认参数
OUTPUT_DIR="./motiv_exp_results"
NUM_HP=100
VERBOSE=""
DRY_RUN=""

# 使用说明
usage() {
    cat << EOF
Usage: $0 [OPTIONS] CASE

运行Motivation实验的便捷脚本

Arguments:
    CASE        实验编号: 1, 2, 3, 或 'all'（运行所有）

Options:
    -o DIR      输出目录（默认: ./motiv_exp_results）
    -n NUM      超周期数（默认: 100）
    -v          详细输出
    -d          干运行模式（只打印命令）
    -h          显示此帮助信息

Examples:
    # 运行Case 1
    $0 1

    # 运行所有实验，详细输出
    $0 -v all

    # 运行Case 3，使用自定义输出目录和更多周期
    $0 -o ./my_results -n 500 3

EOF
    exit 1
}

# 解析参数
while getopts "o:n:vdh" opt; do
    case $opt in
        o) OUTPUT_DIR="$OPTARG" ;;
        n) NUM_HP="$OPTARG" ;;
        v) VERBOSE="--verbose" ;;
        d) DRY_RUN="--dry_run" ;;
        h) usage ;;
        \?) echo "Invalid option -$OPTARG" >&2; usage ;;
    esac
done

shift $((OPTIND-1))

# 检查CASE参数
if [ $# -eq 0 ]; then
    echo -e "${RED}Error: CASE argument required${NC}"
    usage
fi

CASE=$1

# 验证CASE有效性
if [[ ! "$CASE" =~ ^(1|2|3|all)$ ]]; then
    echo -e "${RED}Error: Invalid CASE: $CASE${NC}"
    echo "CASE must be 1, 2, 3, or 'all'"
    exit 1
fi

# 项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 进入项目根目录
cd "$PROJECT_ROOT"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Motivation Experiments Runner${NC}"
echo -e "${GREEN}========================================${NC}"
echo "Output Directory: $OUTPUT_DIR"
echo "Num Hyperperiods: $NUM_HP"
if [ -n "$VERBOSE" ]; then echo "Verbose: ON"; fi
if [ -n "$DRY_RUN" ]; then echo -e "${YELLOW}DRY RUN MODE${NC}"; fi
echo -e "${GREEN}========================================${NC}\n"

# 运行Case 1
run_case1() {
    echo -e "\n${GREEN}>>> Running Case 1: 纯静态调度 - 利用率问题${NC}"
    python scripts/motiv_exp_runner.py \
        --case 1 \
        --output_dir "$OUTPUT_DIR" \
        --num_hp "$NUM_HP" \
        $VERBOSE \
        $DRY_RUN
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Case 1 完成${NC}"
    else
        echo -e "${RED}✗ Case 1 失败${NC}"
        return 1
    fi
}

# 运行Case 2
run_case2() {
    echo -e "\n${GREEN}>>> Running Case 2: 纯动态调度 - 可扩展性问题${NC}"
    python scripts/motiv_exp_runner.py \
        --case 2 \
        --output_dir "$OUTPUT_DIR" \
        --num_hp "$NUM_HP" \
        $VERBOSE \
        $DRY_RUN
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Case 2 完成${NC}"
    else
        echo -e "${RED}✗ Case 2 失败${NC}"
        return 1
    fi
}

# 运行Case 3
run_case3() {
    echo -e "\n${GREEN}>>> Running Case 3: 切换行为的不确定性${NC}"
    
    # Case 3使用更多周期（如果未自定义，使用1000）
    CASE3_PERIODS=$NUM_HP
    if [ "$NUM_HP" -eq 100 ]; then
        CASE3_PERIODS=1000
        echo "  (Using $CASE3_PERIODS periods for Case 3)"
    fi
    
    python scripts/motiv_exp_runner.py \
        --case 3 \
        --output_dir "$OUTPUT_DIR" \
        --case3_num_periods "$CASE3_PERIODS" \
        --case3_mode binned \
        $VERBOSE \
        $DRY_RUN
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Case 3 完成${NC}"
    else
        echo -e "${RED}✗ Case 3 失败${NC}"
        return 1
    fi
}

# 执行实验
if [ "$CASE" == "all" ]; then
    echo "Running all cases..."
    run_case1 || exit 1
    run_case2 || exit 1
    run_case3 || exit 1
    
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}所有实验完成！${NC}"
    echo -e "${GREEN}结果保存在: $OUTPUT_DIR${NC}"
    echo -e "${GREEN}========================================${NC}"
elif [ "$CASE" == "1" ]; then
    run_case1
elif [ "$CASE" == "2" ]; then
    run_case2
elif [ "$CASE" == "3" ]; then
    run_case3
fi

echo -e "\n${GREEN}Done!${NC}"

