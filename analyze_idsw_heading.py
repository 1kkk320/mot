import argparse
import csv
import math
import os
from collections import defaultdict

import numpy as np


def wrap_to_pi(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def symmetric_heading_diff_deg(a, b):
    delta = wrap_to_pi(float(a) - float(b))
    symmetric_delta = math.acos(np.clip(abs(math.cos(delta)), 0.0, 1.0))
    return math.degrees(symmetric_delta)


def bbox_iou_xyxy(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def bbox_ioa_xyxy(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    if area_a <= 0:
        return 0.0
    return inter / area_a


def linear_assignment_max(score_matrix):
    if score_matrix.size == 0:
        return np.empty((0, 2), dtype=int)
    try:
        import lap

        _, x, y = lap.lapjv(-score_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0], dtype=int)
    except Exception:
        from scipy.optimize import linear_sum_assignment

        row_ind, col_ind = linear_sum_assignment(-score_matrix)
        return np.stack([row_ind, col_ind], axis=1).astype(int)


def read_kitti_objects(path):
    frame_dict = defaultdict(list)
    if not os.path.isfile(path):
        return frame_dict
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 17:
                continue
            frame_id = int(parts[0])
            obj = {
                "frame": frame_id,
                "id": int(parts[1]),
                "cls": parts[2].lower(),
                "truncation": float(parts[3]),
                "occlusion": float(parts[4]),
                "alpha": float(parts[5]),
                "bbox": np.array(list(map(float, parts[6:10])), dtype=np.float64),
                "height_2d": float(parts[9]) - float(parts[7]),
                "h": float(parts[10]),
                "w": float(parts[11]),
                "l": float(parts[12]),
                "x": float(parts[13]),
                "y": float(parts[14]),
                "z": float(parts[15]),
                "rot_y": float(parts[16]),
                "score": float(parts[17]) if len(parts) > 17 else 1.0,
            }
            frame_dict[frame_id].append(obj)
    return frame_dict


def match_detections(gt_objs, trk_objs, iou_thr):
    if len(gt_objs) == 0 or len(trk_objs) == 0:
        return np.empty((0, 2), dtype=int)
    sim = np.zeros((len(gt_objs), len(trk_objs)), dtype=np.float64)
    for i, gt in enumerate(gt_objs):
        for j, trk in enumerate(trk_objs):
            sim[i, j] = bbox_iou_xyxy(gt["bbox"], trk["bbox"])
    sim[sim < iou_thr] = 0.0
    matched = linear_assignment_max(sim)
    if matched.size == 0:
        return matched
    keep = sim[matched[:, 0], matched[:, 1]] > 0
    return matched[keep]


def preprocess_kitti_frame(gt_all, trk_all, target_cls="car", iou_thr=0.5):
    distractor_classes = {"car": {"van"}, "pedestrian": {"person"}}.get(target_cls, set())
    gt_relevant = [g for g in gt_all if g["cls"] == target_cls or g["cls"] in distractor_classes]
    gt_ignore = [g for g in gt_all if g["cls"] == "dontcare"]
    trk_relevant = [t for t in trk_all if t["cls"] == target_cls]

    matched_initial = match_detections(gt_relevant, trk_relevant, iou_thr)
    remove_tracker_indices = set()
    matched_tracker_indices = set()
    for gt_idx, trk_idx in matched_initial:
        matched_tracker_indices.add(int(trk_idx))
        gt = gt_relevant[int(gt_idx)]
        is_distractor = gt["cls"] in distractor_classes
        is_occluded_or_truncated = gt["occlusion"] > 2 or gt["truncation"] > 0
        if is_distractor or is_occluded_or_truncated:
            remove_tracker_indices.add(int(trk_idx))

    unmatched_tracker_indices = [i for i in range(len(trk_relevant)) if i not in matched_tracker_indices]
    for trk_idx in unmatched_tracker_indices:
        trk = trk_relevant[trk_idx]
        too_small = trk["height_2d"] <= 25.0
        in_ignore = any(bbox_ioa_xyxy(trk["bbox"], ig["bbox"]) > 0.5 for ig in gt_ignore)
        if too_small or in_ignore:
            remove_tracker_indices.add(int(trk_idx))

    trk_final = [t for i, t in enumerate(trk_relevant) if i not in remove_tracker_indices]
    gt_final = [
        g for g in gt_relevant
        if g["cls"] == target_cls and not (g["occlusion"] > 2 or g["truncation"] > 0)
    ]
    return gt_final, trk_final


def center_distance(box_a, box_b):
    ca = np.array([(box_a[0] + box_a[2]) * 0.5, (box_a[1] + box_a[3]) * 0.5], dtype=np.float64)
    cb = np.array([(box_b[0] + box_b[2]) * 0.5, (box_b[1] + box_b[3]) * 0.5], dtype=np.float64)
    return float(np.linalg.norm(ca - cb))


def summarize_events(events):
    total = len(events)
    analyzable = [e for e in events if e["confuser_heading_diff_deg"] is not None]
    print("=" * 80)
    print("IDSW Heading Analysis")
    print("=" * 80)
    print(f"Total IDSW events: {total}")
    print(f"Events with current-frame confuser GT: {len(analyzable)}")
    if len(analyzable) == 0:
        print("No analyzable IDSW events with paired confuser GT were found.")
        return

    diffs = np.array([e["confuser_heading_diff_deg"] for e in analyzable], dtype=np.float64)
    print(f"Confuser heading diff mean/median/min/max: {diffs.mean():.2f} / {np.median(diffs):.2f} / {diffs.min():.2f} / {diffs.max():.2f} deg")
    for thr in (5, 10, 15, 20, 30, 45):
        cnt = int(np.sum(diffs <= thr))
        print(f"Confuser heading diff <= {thr:2d} deg: {cnt:3d} ({cnt / len(diffs) * 100:.1f}%)")

    match_errs = [e["matched_heading_error_deg"] for e in events if e["matched_heading_error_deg"] is not None]
    if match_errs:
        match_errs = np.array(match_errs, dtype=np.float64)
        print(f"Matched heading error mean/median: {match_errs.mean():.2f} / {np.median(match_errs):.2f} deg")

    seq_counter = defaultdict(int)
    small_counter = defaultdict(int)
    for event in analyzable:
        seq_counter[event["seq"]] += 1
        if event["confuser_heading_diff_deg"] <= 15.0:
            small_counter[event["seq"]] += 1
    print("-" * 80)
    print("Per-sequence analyzable IDSW counts:")
    for seq in sorted(seq_counter):
        print(f"  {seq}: total={seq_counter[seq]}, <=15deg={small_counter[seq]}")


def analyze_sequence(seq, gt_path, tracker_path, iou_thr):
    gt_frames = read_kitti_objects(gt_path)
    trk_frames = read_kitti_objects(tracker_path)
    frame_ids = sorted(set(gt_frames.keys()) | set(trk_frames.keys()))

    prev_match_by_gt = {}
    events = []

    for frame_id in frame_ids:
        gt_final, trk_final = preprocess_kitti_frame(
            gt_frames.get(frame_id, []),
            trk_frames.get(frame_id, []),
            target_cls="car",
            iou_thr=iou_thr,
        )
        matches = match_detections(gt_final, trk_final, iou_thr)

        gt_by_id = {g["id"]: g for g in gt_final}
        trk_by_id = {t["id"]: t for t in trk_final}
        current_gt_to_trk = {}
        current_trk_to_gt = {}
        for gt_idx, trk_idx in matches:
            gt_id = int(gt_final[int(gt_idx)]["id"])
            trk_id = int(trk_final[int(trk_idx)]["id"])
            current_gt_to_trk[gt_id] = trk_id
            current_trk_to_gt[trk_id] = gt_id

        for gt_id, trk_id in current_gt_to_trk.items():
            prev_trk_id = prev_match_by_gt.get(gt_id)
            if prev_trk_id is not None and prev_trk_id != trk_id:
                gt_obj = gt_by_id[gt_id]
                trk_obj = trk_by_id[trk_id]
                confuser_gt_id = current_trk_to_gt.get(prev_trk_id)
                confuser_heading_diff = None
                confuser_center_dist = None
                confuser_iou = None
                if confuser_gt_id is not None and confuser_gt_id != gt_id and confuser_gt_id in gt_by_id:
                    confuser_obj = gt_by_id[confuser_gt_id]
                    confuser_heading_diff = symmetric_heading_diff_deg(
                        gt_obj["rot_y"], confuser_obj["rot_y"]
                    )
                    confuser_center_dist = center_distance(gt_obj["bbox"], confuser_obj["bbox"])
                    confuser_iou = bbox_iou_xyxy(gt_obj["bbox"], confuser_obj["bbox"])
                matched_heading_error = symmetric_heading_diff_deg(
                    gt_obj["rot_y"], trk_obj["rot_y"]
                )
                events.append({
                    "seq": seq,
                    "frame": int(frame_id),
                    "gt_id": int(gt_id),
                    "prev_trk_id": int(prev_trk_id),
                    "curr_trk_id": int(trk_id),
                    "confuser_gt_id": int(confuser_gt_id) if confuser_gt_id is not None else "",
                    "gt_heading_deg": float(math.degrees(gt_obj["rot_y"])),
                    "matched_heading_deg": float(math.degrees(trk_obj["rot_y"])),
                    "matched_heading_error_deg": float(matched_heading_error),
                    "confuser_heading_deg": float(math.degrees(gt_by_id[confuser_gt_id]["rot_y"])) if confuser_gt_id in gt_by_id else "",
                    "confuser_heading_diff_deg": float(confuser_heading_diff) if confuser_heading_diff is not None else None,
                    "confuser_center_dist_px": float(confuser_center_dist) if confuser_center_dist is not None else None,
                    "confuser_gt_iou": float(confuser_iou) if confuser_iou is not None else None,
                })

        prev_match_by_gt.update(current_gt_to_trk)

    return events


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "seq",
        "frame",
        "gt_id",
        "prev_trk_id",
        "curr_trk_id",
        "confuser_gt_id",
        "gt_heading_deg",
        "matched_heading_deg",
        "matched_heading_error_deg",
        "confuser_heading_deg",
        "confuser_heading_diff_deg",
        "confuser_center_dist_px",
        "confuser_gt_iou",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze whether residual IDSWs happen when vehicle headings are already similar.")
    parser.add_argument("--gt-folder", default=r"E:\mot\datasets\kitti\train\label_02")
    parser.add_argument("--tracker-folder", default=r"E:\mot\results\virconv_OCM\data")
    parser.add_argument("--seqs", default="", help="Comma separated seq ids, e.g. 0000,0001")
    parser.add_argument("--iou-thr", type=float, default=0.5)
    parser.add_argument("--output-csv", default=r"E:\mot\logs\idsw_heading_events.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seqs.strip():
        seqs = [s.strip() for s in args.seqs.split(",") if s.strip()]
    else:
        seqs = sorted(
            os.path.splitext(x)[0]
            for x in os.listdir(args.tracker_folder)
            if x.endswith(".txt")
        )

    all_events = []
    for seq in seqs:
        gt_path = os.path.join(args.gt_folder, seq + ".txt")
        tracker_path = os.path.join(args.tracker_folder, seq + ".txt")
        if not os.path.isfile(gt_path) or not os.path.isfile(tracker_path):
            continue
        seq_events = analyze_sequence(seq, gt_path, tracker_path, args.iou_thr)
        all_events.extend(seq_events)

    write_csv(args.output_csv, all_events)
    summarize_events(all_events)
    print("-" * 80)
    print(f"Saved per-event CSV to: {args.output_csv}")


if __name__ == "__main__":
    main()
