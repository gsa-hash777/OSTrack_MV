import csv
import json
import os

# OUT_DIR = "results3_wild_cat_T"
OUT_DIR = "results_md_B12_0.25"

def save_md_results(md, results, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)

    # ---------- JSON ----------
    with open(os.path.join(out_dir, f"{md}.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ---------- CSV ----------
    with open(os.path.join(out_dir, f"{md}.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["md", "view", "SR", "PR20", "NP", "IoU"])
        for view, m in results.items():
            writer.writerow([md, view, m["SR"], m["PR20"], m["NP"], m["IoU"]])

def aggregate_dataset_by_md(md_list, all_results, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)

    dataset_by_md = {}

    for md, md_res in zip(md_list, all_results):
        SRs, PR20s, NPs, IoUs = [], [], [], []

        for m in md_res.values():
            SRs.append(m["SR"])
            PR20s.append(m["PR20"])
            NPs.append(m["NP"])
            IoUs.append(m["IoU"])

        dataset_by_md[md] = {
            "SR": sum(SRs) / len(SRs),
            "PR20": sum(PR20s) / len(PR20s),
            "NP": sum(NPs) / len(NPs),
            "IoU": sum(IoUs) / len(IoUs)
        }

    # JSON
    with open(os.path.join(out_dir, "dataset_by_md.json"), "w") as f:
        json.dump(dataset_by_md, f, indent=2)

    # CSV
    with open(os.path.join(out_dir, "dataset_by_md.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["md", "SR", "PR20", "NP", "IoU"])
        for md, m in dataset_by_md.items():
            writer.writerow([md, m["SR"], m["PR20"], m["NP"], m["IoU"]])

    return dataset_by_md

def aggregate_whole_dataset(dataset_by_md, out_dir=OUT_DIR):
    """
    dataset_by_md:
    {
        "md2001": {"SR": ..., "PR20": ..., "NP": ..., "IoU": ...},
        "md2002": {...},
        ...
    }
    """
    SRs, PR20s, NPs, IoUs = [], [], [], []

    for m in dataset_by_md.values():
        SRs.append(m["SR"])
        PR20s.append(m["PR20"])
        NPs.append(m["NP"])
        IoUs.append(m["IoU"])

    dataset_avg = {
        "SR": sum(SRs) / len(SRs),
        "PR20": sum(PR20s) / len(PR20s),
        "NP": sum(NPs) / len(NPs),
        "IoU": sum(IoUs) / len(IoUs)
    }

    # ---------- JSON ----------
    import json, os
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "dataset_overall.json"), "w") as f:
        json.dump(dataset_avg, f, indent=2)

    # ---------- CSV ----------
    import csv
    with open(os.path.join(out_dir, "dataset_overall.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["SR", "PR20", "NP", "IoU"])
        writer.writerow([
            dataset_avg["SR"],
            dataset_avg["PR20"],
            dataset_avg["NP"],
            dataset_avg["IoU"]
        ])

    return dataset_avg

def aggregate_dataset_by_view(md_list, all_results, out_dir=OUT_DIR):
    """
    all_results:
    [
      {1: {"SR":..., "PR20":..., "NP":..., "IoU":...},
       2: {...}},          # md2001
      {1: {...}, 2: {...}}, # md2002
      ...
    ]
    """

    from collections import defaultdict
    import json, csv, os

    os.makedirs(out_dir, exist_ok=True)

    view_metrics = defaultdict(lambda: {
        "SR": [], "PR20": [], "NP": [], "IoU": []
    })

    # -------- 汇总所有 md 的同一视角 --------
    for md, md_res in zip(md_list, all_results):
        for view_id, m in md_res.items():
            view_metrics[view_id]["SR"].append(m["SR"])
            view_metrics[view_id]["PR20"].append(m["PR20"])
            view_metrics[view_id]["NP"].append(m["NP"])
            view_metrics[view_id]["IoU"].append(m["IoU"])

    # -------- 计算平均 --------
    dataset_by_view = {}
    for view_id, metrics in view_metrics.items():
        dataset_by_view[view_id] = {
            "SR": sum(metrics["SR"]) / len(metrics["SR"]),
            "PR20": sum(metrics["PR20"]) / len(metrics["PR20"]),
            "NP": sum(metrics["NP"]) / len(metrics["NP"]),
            "IoU": sum(metrics["IoU"]) / len(metrics["IoU"]),
        }

    # -------- JSON --------
    with open(os.path.join(out_dir, "dataset_by_view.json"), "w") as f:
        json.dump(dataset_by_view, f, indent=2)

    # -------- CSV --------
    with open(os.path.join(out_dir, "dataset_by_view.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["view", "SR", "PR20", "NP", "IoU"])
        for view_id, m in dataset_by_view.items():
            writer.writerow([view_id, m["SR"], m["PR20"], m["NP"], m["IoU"]])

    return dataset_by_view

