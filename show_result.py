import os
import pandas as pd
import numpy as np


# ===== 你指定的 task 顺序 =====
TASK_ORDER = "dtlz1 dtlz2 dtlz3 dtlz4 zdt1 zdt2 zdt3 zdt4 zdt6 dtlz5 dtlz6 dtlz7 re21 re22 re23 re24 re25 re31 re32 re33 re34 re35 re36 re37 re41 re42 re61".split()


def parse_metadata_from_path(root):
    """
    解析路径中的 meta 信息
    results/model-guidance-method-task-setting-version/run_name/
    """
    parts = root.split(os.sep)

    meta = {}

    # ===== 解析 task 所在目录 =====
    try:
        task_dir = parts[-2]  # Diffusion-Guidance-Crowding-DTLZ1-Exact-v0
        tokens = task_dir.split("-")

        meta["model"] = tokens[0]
        meta["guidance"] = tokens[1]
        meta["method"] = tokens[2]
        meta["task"] = tokens[3].lower()   # 统一小写
        meta["setting"] = tokens[4]
        meta["version"] = tokens[5]
    except:
        meta["task"] = "unknown"

    # ===== 解析 seed =====
    run_name = parts[-1]
    meta["seed"] = "unknown"

    for t in run_name.split("-"):
        if "seed" in t:
            meta["seed"] = t.replace("seed", "")

    return meta


def extract_results(results_root):
    """
    遍历所有 hv_results.csv
    """
    results = []

    for root, dirs, files in os.walk(results_root):
        if "hv_results.csv" in files:
            file_path = os.path.join(root, "hv_results.csv")

            try:
                df = pd.read_csv(file_path)
                record = df.iloc[0].to_dict()

                meta = parse_metadata_from_path(root)
                record.update(meta)

                results.append(record)

            except Exception as e:
                print(f"读取失败: {file_path}, error: {e}")

    return results


def summarize_by_task(results):
    """
    按 task 聚合统计
    """
    grouped = {}

    for r in results:
        task = r["task"]
        grouped.setdefault(task, []).append(r)

    summary = {}

    for task, items in grouped.items():
        hv10 = [r["hypervolume_10/100th"] for r in items if "hypervolume_10/100th" in r]
        hv20 = [r["hypervolume_20/100th"] for r in items if "hypervolume_20/100th" in r]

        summary[task] = {
            "runs": len(items),
            "HV@10_mean": np.mean(hv10) if hv10 else None,
            "HV@10_std": np.std(hv10) if hv10 else None,
            "HV@20_mean": np.mean(hv20) if hv20 else None,
            "HV@20_std": np.std(hv20) if hv20 else None,
        }

    return summary


if __name__ == "__main__":
    results_dir = "./results"

    results = extract_results(results_dir)

    print(f"\n共读取 {len(results)} 个 runs\n")

    # ===== 打印每个 run（调试用）=====
    for r in results:
        print({
            "task": r["task"],
            "seed": r["seed"],
            "HV@10": r.get("hypervolume_10/100th"),
            "HV@20": r.get("hypervolume_20/100th"),
        })

    # ===== 汇总 =====
    summary = summarize_by_task(results)

    print("\n===== SUMMARY (Ordered) =====")

    for task in TASK_ORDER:
        if task in summary:
            stats = summary[task]

            print(f"\nTASK: {task}")
            print(f"runs: {stats['runs']}")

            if stats["HV@10_mean"] is not None:
                print(f"HV@10: {stats['HV@10_mean']:.4f} ± {stats['HV@10_std']:.4f}")
            else:
                print("HV@10: None")

            if stats["HV@20_mean"] is not None:
                print(f"HV@20: {stats['HV@20_mean']:.4f} ± {stats['HV@20_std']:.4f}")
            else:
                print("HV@20: None")

        else:
            print(f"\nTASK: {task} (no data)")