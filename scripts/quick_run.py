"""
曜珑GleamLM 快速训练 + 验证一体化脚本

三级规模，适合不同验证场景：

  Level 1 冒烟测试  (~30s):  验证代码能跑通，loss 是否下降
  Level 2 小规模跑  (~5min):  看模型是否在学，生成是否开始有语义
  Level 3 全量训练  (~小时): 正式训练获取可用模型

用法:
    python scripts/quick_run.py --level 1         # 冒烟测试
    python scripts/quick_run.py --level 2         # 小规模训练+验证
    python scripts/quick_run.py --level 3         # 全量训练
"""

import argparse
import os
import shutil
import subprocess

# 测试数据独立目录，不污染生产数据
TEST_DATA_DIR = "data/nano_test_splits"
TEST_CKPT_DIR = "checkpoints_smoke"


def run(cmd, desc="", conda_env="dl2llm"):
    """执行命令并打印，支持自定义 conda 环境"""
    if conda_env:
        cmd = f"conda run -n {conda_env} {cmd}"
    if desc:
        print(f"\n{'=' * 60}")
        print(f"  {desc}")
        print(f"{'=' * 60}")
    print(f"  $ {cmd}")
    result = subprocess.run(
        cmd, shell=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    if result.returncode != 0:
        print(f"  [!] Command failed (exit code {result.returncode})")
        return False
    return True


def prepare_small_data(n_train=2000, n_valid=500):
    """准备小数据集 — 写入独立测试目录，不碰 production"""
    print("\n>>> 准备小数据集...")
    os.makedirs(TEST_DATA_DIR, exist_ok=True)

    for split, n in [("train", n_train), ("valid", n_valid)]:
        src = f"data/nano_data/{split}.txt"
        dst = f"{TEST_DATA_DIR}/{split}.txt"

        if not os.path.exists(src):
            print(f"  跳过: {src} 不存在")
            continue

        lines = []
        with open(src, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= n:
                    break
                lines.append(line)

        with open(dst, "w", encoding="utf-8") as f:
            f.writelines(lines)

        # 删除旧预分词缓存
        npy = f"{TEST_DATA_DIR}/{split}_ids.npy"
        if os.path.exists(npy):
            os.remove(npy)

        print(f"  {split}.txt: {len(lines)} 行, {os.path.getsize(dst) / 1024:.0f} KB")

    print("  小数据集准备完成!")


def main():
    parser = argparse.ArgumentParser(description="曜珑GleamLM 快速训练+验证")
    parser.add_argument(
        "--level",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="1=冒烟测试 2=小规模训练+验证 3=全量训练",
    )
    parser.add_argument("--verify_only", action="store_true", help="只验证已有模型，不训练")
    parser.add_argument(
        "--conda_env",
        type=str,
        default="dl2llm",
        help="conda 环境名 (默认: dl2llm，传空字符串则不使用 conda)",
    )
    args = parser.parse_args()

    # 验证已有模型
    if args.verify_only:
        print("\n>>> 验证已有模型: gleamlm-nano/checkpoints/best_model.pt")
        run(
            "python scripts/eval_ppl.py --max_batches 50 --batch_size 4",
            "PPL 评估 (50 batches)",
            conda_env=args.conda_env,
        )
        run(
            "python gleamlm-nano/evaluation/generate_samples.py "
            "--model gleamlm-nano/checkpoints/best_model.pt --temperature 0.8 --top_k 50 --top_p 0.9 --max_new_tokens 64",
            "生成样例",
            conda_env=args.conda_env,
        )
        return

    # Level 1: 冒烟测试
    if args.level == 1:
        print("\n" + "=" * 60)
        print("  Level 1: 冒烟测试 (验证代码能跑通)")
        print("=" * 60)

        # 备份正式 checkpoint
        ckpt_path = "gleamlm-nano/checkpoints/best_model.pt"
        ckpt_backup = "gleamlm-nano/checkpoints/best_model.pt.backup"
        if os.path.exists(ckpt_path):
            shutil.copy(ckpt_path, ckpt_backup)
            print("  已备份: best_model.pt -> best_model.pt.backup")

        prepare_small_data(n_train=2000, n_valid=500)

        ok = run(
            "python gleamlm-nano/train.py "
            f"--data_dir {TEST_DATA_DIR} "
            "--epochs 2 --batch_size 8 --accumulate_grad 4 "
            f"--checkpoint_dir ./{TEST_CKPT_DIR}",
            "训练 2 epochs (预计 ~30s)",
            conda_env=args.conda_env,
        )
        if not ok:
            print("\n[!] 训练失败")
            if os.path.exists(ckpt_backup):
                shutil.copy(ckpt_backup, ckpt_path)
            return

        run(
            f"python scripts/eval_ppl.py "
            f"--data_dir {TEST_DATA_DIR} "
            f"--model {TEST_CKPT_DIR}/best_model.pt --max_batches 30 --batch_size 4",
            "PPL 评估",
            conda_env=args.conda_env,
        )
        run(
            "python gleamlm-nano/evaluation/generate_samples.py "
            f"--model {TEST_CKPT_DIR}/best_model.pt --temperature 0.8 --max_new_tokens 32",
            "生成样例",
            conda_env=args.conda_env,
        )

        # 恢复 checkpoint
        if os.path.exists(ckpt_backup):
            shutil.copy(ckpt_backup, ckpt_path)
            os.remove(ckpt_backup)
            print("  已恢复: best_model.pt")

        # 清理
        if os.path.exists(TEST_CKPT_DIR):
            shutil.rmtree(TEST_CKPT_DIR)
        if os.path.exists(TEST_DATA_DIR):
            shutil.rmtree(TEST_DATA_DIR)

        print("\n>>> Level 1 完成!")

    # Level 2: 小规模训练 + 验证
    elif args.level == 2:
        print("\n" + "=" * 60)
        print("  Level 2: 小规模训练 + 验证")
        print("=" * 60)

        ckpt_path = "gleamlm-nano/checkpoints/best_model.pt"
        ckpt_backup = "gleamlm-nano/checkpoints/best_model.pt.backup"
        if os.path.exists(ckpt_path):
            shutil.copy(ckpt_path, ckpt_backup)
            print("  已备份: best_model.pt -> best_model.pt.backup")

        prepare_small_data(n_train=10000, n_valid=2000)

        ok = run(
            "python gleamlm-nano/train.py "
            f"--data_dir {TEST_DATA_DIR} "
            "--epochs 5 --batch_size 8 --accumulate_grad 8 "
            f"--checkpoint_dir ./{TEST_CKPT_DIR}",
            "训练 5 epochs (预计 ~5min)",
            conda_env=args.conda_env,
        )
        if not ok:
            print("\n[!] 训练失败")
            if os.path.exists(ckpt_backup):
                shutil.copy(ckpt_backup, ckpt_path)
            return

        print("\n>>> 开始完整验证...")
        run(
            f"python scripts/eval_ppl.py "
            f"--data_dir {TEST_DATA_DIR} "
            f"--model {TEST_CKPT_DIR}/best_model.pt --max_batches 100 --batch_size 4",
            "PPL 评估 (100 batches)",
            conda_env=args.conda_env,
        )
        run(
            "python gleamlm-nano/evaluation/generate_samples.py "
            f"--model {TEST_CKPT_DIR}/best_model.pt --temperature 0.8 --top_k 50 --top_p 0.9 --max_new_tokens 64",
            "生成样例 (temperature=0.8)",
            conda_env=args.conda_env,
        )

        if os.path.exists(ckpt_backup):
            shutil.copy(ckpt_backup, ckpt_path)
            os.remove(ckpt_backup)
            print("  已恢复: best_model.pt")

        if os.path.exists(TEST_CKPT_DIR):
            shutil.rmtree(TEST_CKPT_DIR)
        if os.path.exists(TEST_DATA_DIR):
            shutil.rmtree(TEST_DATA_DIR)

        print("\n>>> Level 2 完成!")

    # Level 3: 全量正式训练
    elif args.level == 3:
        print("\n" + "=" * 60)
        print("  Level 3: 全量正式训练")
        print("=" * 60)

        cmd = "python gleamlm-nano/train.py --epochs 8 --batch_size 8 --accumulate_grad 8"
        ok = run(cmd, "全量训练 8 epochs", conda_env=args.conda_env)
        if not ok:
            print("\n[!] 训练异常退出")
            return

        print("\n>>> 训练完成，开始验证...")
        run("python scripts/eval_ppl.py --batch_size 4", "完整 PPL 评估", conda_env=args.conda_env)
        run(
            "python gleamlm-nano/evaluation/generate_samples.py "
            "--model gleamlm-nano/checkpoints/best_model.pt --temperature 0.8 --top_k 50 --top_p 0.9 --max_new_tokens 128",
            "生成样例",
            conda_env=args.conda_env,
        )
        print("\n>>> Level 3 完成!")


if __name__ == "__main__":
    main()
