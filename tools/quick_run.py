"""
GleamLM 快速训练 + 验证一体化脚本

三级规模，适合不同验证场景：

  Level 1 冒烟测试  (~30s):  验证代码能跑通，loss 是否下降
  Level 2 小规模跑  (~5min):  看模型是否在学，生成是否开始有语义
  Level 3 全量训练  (~小时): 正式训练获取可用模型

用法:
    python tools/quick_run.py --level 1 --variant nano
    python tools/quick_run.py --level 2 --variant lite
    python tools/quick_run.py --level 3 --variant pro
"""

import argparse
import os
import shutil
import subprocess

TEST_DATA_DIR = "data/smoke_splits"
TEST_CKPT_DIR = "checkpoints_smoke"


def run(cmd, desc="", conda_env="dl2llm"):
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
    print("\n>>> 准备小数据集...")
    os.makedirs(TEST_DATA_DIR, exist_ok=True)

    for split, n in [("train", n_train), ("valid", n_valid)]:
        src = f"data/nano/pretrain/{split}.txt"
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
    parser.add_argument(
        "--variant", type=str, choices=["nano", "lite", "pro"], default="nano", help="模型变体"
    )
    parser.add_argument("--verify_only", action="store_true", help="只验证已有模型，不训练")
    parser.add_argument(
        "--conda_env",
        type=str,
        default="dl2llm",
        help="conda 环境名 (默认: dl2llm，传空字符串则不使用 conda)",
    )
    args = parser.parse_args()

    v = args.variant
    ckpt_dir = f"checkpoints/{v}"

    # 验证已有模型
    if args.verify_only:
        print(f"\n>>> 验证已有模型: {ckpt_dir}/best_model.pt")
        run(
            f"python -m tools.eval_runner --model {ckpt_dir}/best_model.pt --data_dir data/{v}/pretrain --benchmarks ppl --max_batches 50 --batch_size 4",
            "PPL 评估 (50 batches)",
            conda_env=args.conda_env,
        )
        run(
            f"python -m gleamlm.inference.cli --model {ckpt_dir}/best_model.pt --prompt \"介绍一下你自己\"",
            "生成样例",
            conda_env=args.conda_env,
        )
        return

    # Level 1: 冒烟测试
    if args.level == 1:
        print("\n" + "=" * 60)
        print("  Level 1: 冒烟测试 (验证代码能跑通)")
        print("=" * 60)

        ckpt_path = f"{ckpt_dir}/best_model.pt"
        ckpt_backup = f"{ckpt_dir}/best_model.pt.backup"
        if os.path.exists(ckpt_path):
            shutil.copy(ckpt_path, ckpt_backup)
            print("  已备份: best_model.pt -> best_model.pt.backup")

        prepare_small_data(n_train=2000, n_valid=500)

        ok = run(
            f"python scripts/train.py --variant {v} "
            f"--data_dir {TEST_DATA_DIR} --epochs 2 --batch_size 8 --accumulate_grad 4 "
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
            f"python -m tools.eval_runner --model {TEST_CKPT_DIR}/best_model.pt "
            f"--data_dir {TEST_DATA_DIR} "
            f"--benchmarks ppl --max_batches 30 --batch_size 4",
            "PPL 评估",
            conda_env=args.conda_env,
        )
        run(
            f"python -m gleamlm.inference.cli --model {TEST_CKPT_DIR}/best_model.pt --prompt \"介绍一下你自己\"",
            "生成样例",
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

        print("\n>>> Level 1 完成!")

    elif args.level == 2:
        print("\n" + "=" * 60)
        print("  Level 2: 小规模训练 + 验证")
        print("=" * 60)

        ckpt_path = f"{ckpt_dir}/best_model.pt"
        ckpt_backup = f"{ckpt_dir}/best_model.pt.backup"
        if os.path.exists(ckpt_path):
            shutil.copy(ckpt_path, ckpt_backup)
            print("  已备份: best_model.pt -> best_model.pt.backup")

        prepare_small_data(n_train=10000, n_valid=2000)

        ok = run(
            f"python scripts/train.py --variant {v} "
            f"--data_dir {TEST_DATA_DIR} --epochs 5 --batch_size 8 --accumulate_grad 8 "
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
            f"python -m tools.eval_runner --model {TEST_CKPT_DIR}/best_model.pt "
            f"--data_dir {TEST_DATA_DIR} "
            f"--benchmarks ppl --max_batches 100 --batch_size 4",
            "PPL 评估 (100 batches)",
            conda_env=args.conda_env,
        )
        run(
            f"python -m gleamlm.inference.cli --model {TEST_CKPT_DIR}/best_model.pt --prompt \"介绍一下你自己\"",
            "生成样例",
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

    elif args.level == 3:
        print("\n" + "=" * 60)
        print("  Level 3: 全量正式训练")
        print("=" * 60)

        cmd = f"python scripts/train.py --variant {v}"
        ok = run(cmd, f"全量训练 ({v})", conda_env=args.conda_env)
        if not ok:
            print("\n[!] 训练异常退出")
            return

        print("\n>>> 训练完成，开始验证...")
        run(
            f"python -m tools.eval_runner --model {ckpt_dir}/best_model.pt --data_dir data/{v}/pretrain --benchmarks ppl --batch_size 4",
            "完整 PPL 评估",
            conda_env=args.conda_env,
        )
        run(
            f"python -m gleamlm.inference.cli --model {ckpt_dir}/best_model.pt --prompt \"介绍一下你自己\"",
            "生成样例",
            conda_env=args.conda_env,
        )
        print("\n>>> Level 3 完成!")


if __name__ == "__main__":
    main()
