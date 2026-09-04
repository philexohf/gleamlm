"""
实验对比 — 从 SQLite 读取多个 run 的指标并输出对比表格。

用法:
  runs = compare_runs(["nano_lr_1e4", "nano_lr_3e4", "nano_lr_1e3"], db_path="experiments.db")
  # 返回 {run_id: {"config": {...}, "metrics": {"loss": [(step, val), ...], ...}}}
  # 配合 print_table(runs) 输出终端表格
"""

from tools.tracker import ExperimentTracker


def compare_runs(
    run_ids: list[str],
    db_path: str = "experiments.db",
    metric_keys: list[str] | None = None,
) -> dict[str, dict]:
    tracker = ExperimentTracker("_compare", db_path=db_path)
    results = {}
    for rid in run_ids:
        config_data = tracker._conn.execute(
            "SELECT config FROM runs WHERE id=?", (rid,)
        ).fetchone()
        config = __import__("json").loads(config_data[0]) if config_data else {}
        metrics = tracker.get_metrics(rid, keys=metric_keys)
        results[rid] = {"config": config, "metrics": metrics}
    tracker.close()
    return results


def print_table(results: dict[str, dict], key: str = "loss", top_n: int = 5):
    """打印指定指标的最后 N 步对比表格。"""
    rows = []
    for run_id, data in results.items():
        vals = data["metrics"].get(key, [])
        if not vals:
            continue
        last_val = vals[-1][1]
        best_val = min(v[1] for v in vals)
        cfg = data["config"]
        rows.append((last_val, best_val, run_id, cfg))

    rows.sort(key=lambda r: r[0])
    print(f"{'Rank':<6} {'Run ID':<30} {'Last':<10} {'Best':<10}  Config")
    print("-" * 90)
    for rank, (last, best, rid, cfg) in enumerate(rows[:top_n], 1):
        cfg_str = ", ".join(f"{k}={v}" for k, v in sorted(cfg.items()) if k != "run_name")
        print(f"{rank:<6} {rid:<30} {last:<10.4f} {best:<10.4f}  {cfg_str[:80]}")
