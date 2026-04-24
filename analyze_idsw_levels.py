import csv
import math
import os
from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment


GT_ROOT = os.path.join("datasets", "kitti", "train")
TRACKER_ROOT = os.path.join("results", "virconv_OCM")
TRACKER_DATA_DIR = os.path.join(TRACKER_ROOT, "data")
ASSOC_DIR = os.path.join(TRACKER_ROOT, "assoc_levels")
SEQMAP_PATH = os.path.join(GT_ROOT, "evaluate_tracking.seqmap.training")
EVENTS_OUT = os.path.join(TRACKER_ROOT, "idsw_level_events.csv")
SUMMARY_OUT = os.path.join(TRACKER_ROOT, "idsw_level_summary.txt")


def load_seqmap(path):
    seqs = []
    lengths = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 4:
                seq = row[0]
                seqs.append(seq)
                lengths[seq] = int(row[3])
    return seqs, lengths


def parse_kitti_file(path, is_gt):
    frames = defaultdict(list)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return frames
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            frame = int(float(parts[0]))
            track_id = int(float(parts[1]))
            cls = parts[2].lower()
            if (not is_gt) and track_id < 0:
                continue
            entry = {
                "frame": frame,
                "id": track_id,
                "class": cls,
                "bbox": np.array([float(parts[6]), float(parts[7]), float(parts[8]), float(parts[9])], dtype=np.float64),
            }
            if is_gt:
                entry["truncation"] = int(float(parts[3]))
                entry["occlusion"] = int(float(parts[4]))
            else:
                entry["score"] = float(parts[17]) if len(parts) > 17 else 1.0
            frames[frame].append(entry)
    return frames


def load_assoc_levels(path):
    assoc = {}
    if not os.path.exists(path):
        return assoc
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (int(row["frame"]), int(row["track_id"]))
            assoc[key] = {
                "assoc_level": row["assoc_level"],
                "assoc_frame": int(row["assoc_frame"]),
                "time_since_update": int(row["time_since_update"]),
                "hits": int(row["hits"]),
            }
    return assoc


def box_iou_matrix(a, b, do_ioa=False):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    area_a = np.maximum(0.0, a[:, 2] - a[:, 0]) * np.maximum(0.0, a[:, 3] - a[:, 1])
    area_b = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.maximum(0.0, rb - lt)
    inter = wh[:, :, 0] * wh[:, :, 1]
    if do_ioa:
        denom = np.maximum(area_a[:, None], 1e-12)
    else:
        denom = np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-12)
    return inter / denom


def preprocess_frame(gt_entries, tracker_entries):
    distractor_classes = {"van"}
    cls_id = "car"

    gt_eval_candidates = [g for g in gt_entries if g["class"] in {cls_id, "van"}]
    gt_ids = [g["id"] for g in gt_eval_candidates]
    gt_boxes = [g["bbox"] for g in gt_eval_candidates]
    gt_classes = [g["class"] for g in gt_eval_candidates]
    gt_occ = [g.get("occlusion", 0) for g in gt_eval_candidates]
    gt_trunc = [g.get("truncation", 0) for g in gt_eval_candidates]

    tracker_eval = [t for t in tracker_entries if t["class"] == cls_id]
    tracker_ids = [t["id"] for t in tracker_eval]
    tracker_boxes = [t["bbox"] for t in tracker_eval]

    similarity = box_iou_matrix(gt_boxes, tracker_boxes)

    to_remove_matched = set()
    unmatched_indices = list(range(len(tracker_ids)))
    if gt_ids and tracker_ids:
        matching_scores = similarity.copy()
        matching_scores[matching_scores < 0.5 - np.finfo(float).eps] = 0
        rows, cols = linear_sum_assignment(-matching_scores)
        keep = matching_scores[rows, cols] > 0 + np.finfo(float).eps
        rows = rows[keep]
        cols = cols[keep]
        matched_cols = set(int(c) for c in cols.tolist())
        for r, c in zip(rows.tolist(), cols.tolist()):
            is_distractor = gt_classes[r] in distractor_classes
            is_bad = gt_occ[r] > 2 or gt_trunc[r] > 0
            if is_distractor or is_bad:
                to_remove_matched.add(int(c))
        unmatched_indices = [i for i in unmatched_indices if i not in matched_cols]

    unmatched_boxes = [tracker_boxes[i] for i in unmatched_indices]
    unmatched_heights = [max(0.0, box[3] - box[1]) for box in unmatched_boxes]
    is_too_small = [h <= 25 + np.finfo(float).eps for h in unmatched_heights]

    ignore_regions = [g["bbox"] for g in gt_entries if g["class"] == "dontcare"]
    ioa = box_iou_matrix(unmatched_boxes, ignore_regions, do_ioa=True)
    within_ignore = np.any(ioa > 0.5 + np.finfo(float).eps, axis=1) if len(unmatched_boxes) > 0 else np.array([], dtype=bool)
    to_remove_unmatched = {
        unmatched_indices[i]
        for i in range(len(unmatched_indices))
        if is_too_small[i] or bool(within_ignore[i])
    }
    to_remove_tracker = to_remove_matched.union(to_remove_unmatched)

    tracker_keep_mask = [i not in to_remove_tracker for i in range(len(tracker_ids))]
    final_tracker_ids = [tracker_ids[i] for i in range(len(tracker_ids)) if tracker_keep_mask[i]]
    final_tracker_boxes = [tracker_boxes[i] for i in range(len(tracker_boxes)) if tracker_keep_mask[i]]

    gt_keep_mask = [
        (gt_classes[i] == cls_id) and (gt_occ[i] <= 2) and (gt_trunc[i] <= 0)
        for i in range(len(gt_ids))
    ]
    final_gt_ids = [gt_ids[i] for i in range(len(gt_ids)) if gt_keep_mask[i]]
    final_gt_boxes = [gt_boxes[i] for i in range(len(gt_boxes)) if gt_keep_mask[i]]

    final_similarity = box_iou_matrix(final_gt_boxes, final_tracker_boxes)
    return final_gt_ids, final_tracker_ids, final_similarity


def analyze_sequence(seq, num_timesteps, assoc_map):
    gt_path = os.path.join(GT_ROOT, "label_02", seq + ".txt")
    tracker_path = os.path.join(TRACKER_DATA_DIR, seq + ".txt")
    gt_frames = parse_kitti_file(gt_path, is_gt=True)
    tracker_frames = parse_kitti_file(tracker_path, is_gt=False)

    prev_tracker_id = {}
    prev_timestep_tracker_id = {}
    events = []

    for t in range(num_timesteps):
        gt_ids_t, tracker_ids_t, similarity = preprocess_frame(gt_frames.get(t, []), tracker_frames.get(t, []))
        if len(gt_ids_t) == 0:
            continue
        if len(tracker_ids_t) == 0:
            prev_timestep_tracker_id = {}
            continue

        gt_ids_arr = np.asarray(gt_ids_t, dtype=np.int64)
        tracker_ids_arr = np.asarray(tracker_ids_t, dtype=np.int64)
        score_mat = (tracker_ids_arr[np.newaxis, :] == np.asarray([prev_timestep_tracker_id.get(gt_id, math.nan) for gt_id in gt_ids_arr])[:, np.newaxis]).astype(np.float64)
        score_mat = 1000.0 * score_mat + similarity
        score_mat[similarity < 0.5 - np.finfo(float).eps] = 0.0

        rows, cols = linear_sum_assignment(-score_mat)
        keep = score_mat[rows, cols] > 0 + np.finfo(float).eps
        rows = rows[keep]
        cols = cols[keep]

        matched_gt_ids = gt_ids_arr[rows]
        matched_tracker_ids = tracker_ids_arr[cols]

        for gt_id, tracker_id in zip(matched_gt_ids.tolist(), matched_tracker_ids.tolist()):
            if gt_id in prev_tracker_id and prev_tracker_id[gt_id] != tracker_id:
                assoc_info = assoc_map.get((t, tracker_id), None)
                events.append({
                    "seq": seq,
                    "frame": t,
                    "gt_id": gt_id,
                    "prev_tracker_id": int(prev_tracker_id[gt_id]),
                    "new_tracker_id": int(tracker_id),
                    "assoc_level": assoc_info["assoc_level"] if assoc_info else "UNKNOWN",
                    "assoc_frame": assoc_info["assoc_frame"] if assoc_info else -1,
                    "time_since_update": assoc_info["time_since_update"] if assoc_info else -1,
                    "hits": assoc_info["hits"] if assoc_info else -1,
                })

        for gt_id, tracker_id in zip(matched_gt_ids.tolist(), matched_tracker_ids.tolist()):
            prev_tracker_id[gt_id] = tracker_id
        prev_timestep_tracker_id = {gt_id: tracker_id for gt_id, tracker_id in zip(matched_gt_ids.tolist(), matched_tracker_ids.tolist())}

    return events


def main():
    seqs, lengths = load_seqmap(SEQMAP_PATH)
    all_events = []
    for seq in seqs:
        assoc_path = os.path.join(ASSOC_DIR, seq + ".csv")
        assoc_map = load_assoc_levels(assoc_path)
        seq_events = analyze_sequence(seq, lengths[seq], assoc_map)
        all_events.extend(seq_events)

    level_counter = Counter(event["assoc_level"] for event in all_events)
    seq_counter = Counter(event["seq"] for event in all_events)

    with open(EVENTS_OUT, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seq",
                "frame",
                "gt_id",
                "prev_tracker_id",
                "new_tracker_id",
                "assoc_level",
                "assoc_frame",
                "time_since_update",
                "hits",
            ],
        )
        writer.writeheader()
        writer.writerows(all_events)

    with open(SUMMARY_OUT, "w", encoding="utf-8") as f:
        f.write("Total IDSW events: {}\n".format(len(all_events)))
        f.write("By association level:\n")
        for level, count in sorted(level_counter.items(), key=lambda kv: (-kv[1], kv[0])):
            f.write("  {}: {}\n".format(level, count))
        f.write("By sequence:\n")
        for seq, count in sorted(seq_counter.items(), key=lambda kv: (-kv[1], kv[0])):
            f.write("  {}: {}\n".format(seq, count))

    print("Total IDSW events:", len(all_events))
    print("By association level:")
    for level, count in sorted(level_counter.items(), key=lambda kv: (-kv[1], kv[0])):
        print("  {}: {}".format(level, count))


if __name__ == "__main__":
    main()
