from __future__ import annotations

import ast
import csv
import os
import re
from collections import defaultdict

from analyze_idsw_heading import (
    match_detections,
    preprocess_kitti_frame,
    read_kitti_objects,
)


RESULT_ROOT = os.path.join("results", "virconv_OCM", "data")
GT_ROOT = os.path.join("datasets", "kitti", "train", "label_02")
LOG_PATH = os.path.join("logs", "l25_hit_final_events.log")
CSV_PATH = os.path.join("logs", "l25_hit_quality_events.csv")
IOU_THR = 0.5

LINE_RE = re.compile(
    r"\[L2\.5 FinalHit\] seq=(?P<seq>\S+) frame=(?P<frame>-?\d+) "
    r"initial_track_id=(?P<initial_track_id>-?\d+) final_track_id=(?P<track_id>-?\d+) "
    r"confirmed=(?P<confirmed>\d+) state=(?P<state>-?\d+) hits=(?P<hits>-?\d+) "
    r"tsu=(?P<tsu>-?\d+) dt=(?P<dt>\d+) decay=(?P<decay>[0-9eE+\-.]+)"
)


def parse_events(path):
    events = []
    if not os.path.isfile(path):
        return events
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = LINE_RE.search(line.strip())
            if not m:
                continue
            events.append(
                {
                    "seq": m.group("seq"),
                    "frame": int(m.group("frame")),
                    "initial_track_id": int(m.group("initial_track_id")),
                    "track_id": int(m.group("track_id")),
                    "dt": int(m.group("dt")),
                    "decay": float(m.group("decay")),
                    "tsu": int(m.group("tsu")),
                    "confirmed": int(m.group("confirmed")),
                    "state": int(m.group("state")),
                    "hits": int(m.group("hits")),
                }
            )
    return events


def build_frame_maps(seq):
    gt_path = os.path.join(GT_ROOT, f"{seq}.txt")
    trk_path = os.path.join(RESULT_ROOT, f"{seq}.txt")
    gt_frames = read_kitti_objects(gt_path)
    trk_frames = read_kitti_objects(trk_path)
    frame_ids = sorted(set(gt_frames.keys()) | set(trk_frames.keys()))

    gt_to_trk = {}
    trk_to_gt = {}
    idsw_by_gt = defaultdict(list)
    prev_match_by_gt = {}

    for frame_id in frame_ids:
        gt_final, trk_final = preprocess_kitti_frame(
            gt_frames.get(frame_id, []),
            trk_frames.get(frame_id, []),
            target_cls="car",
            iou_thr=IOU_THR,
        )
        matches = match_detections(gt_final, trk_final, IOU_THR)

        curr_gt_to_trk = {}
        curr_trk_to_gt = {}
        for gt_idx, trk_idx in matches:
            gt_id = int(gt_final[int(gt_idx)]["id"])
            trk_id = int(trk_final[int(trk_idx)]["id"])
            curr_gt_to_trk[gt_id] = trk_id
            curr_trk_to_gt[trk_id] = gt_id

        for gt_id, trk_id in curr_gt_to_trk.items():
            prev_trk_id = prev_match_by_gt.get(gt_id)
            if prev_trk_id is not None and prev_trk_id != trk_id:
                idsw_by_gt[gt_id].append(
                    {
                        "frame": int(frame_id),
                        "prev_trk_id": int(prev_trk_id),
                        "curr_trk_id": int(trk_id),
                    }
                )
        prev_match_by_gt.update(curr_gt_to_trk)
        gt_to_trk[int(frame_id)] = curr_gt_to_trk
        trk_to_gt[int(frame_id)] = curr_trk_to_gt

    return gt_to_trk, trk_to_gt, idsw_by_gt


def find_previous_gt_for_track(trk_to_gt, track_id, current_frame):
    prev_frames = sorted(fid for fid in trk_to_gt.keys() if fid < current_frame)
    for fid in reversed(prev_frames):
        gt_id = trk_to_gt[fid].get(track_id)
        if gt_id is not None:
            return fid, gt_id
    return None, None


def find_next_gt_for_track(trk_to_gt, track_id, current_frame):
    next_frames = sorted(fid for fid in trk_to_gt.keys() if fid >= current_frame)
    for fid in next_frames:
        gt_id = trk_to_gt[fid].get(track_id)
        if gt_id is not None:
            return fid, gt_id
    return None, None


def enrich_events(events):
    by_seq = defaultdict(list)
    for ev in events:
        by_seq[ev["seq"]].append(ev)

    enriched = []
    for seq, seq_events in sorted(by_seq.items()):
        gt_to_trk, trk_to_gt, idsw_by_gt = build_frame_maps(seq)
        for ev in seq_events:
            curr_gt_id = trk_to_gt.get(ev["frame"], {}).get(ev["track_id"])
            prev_gt_frame, prev_gt_id = find_previous_gt_for_track(
                trk_to_gt, ev["track_id"], ev["frame"]
            )
            visible_frame, visible_gt_id = find_next_gt_for_track(
                trk_to_gt, ev["track_id"], ev["frame"]
            )

            effective_gt_id = curr_gt_id if curr_gt_id is not None else visible_gt_id
            effective_frame = ev["frame"] if curr_gt_id is not None else visible_frame

            continued_from_prev = (
                effective_gt_id is not None and prev_gt_id is not None and effective_gt_id == prev_gt_id
            )

            future_idsw_frame = None
            future_idsw_new_trk = None
            if effective_gt_id is not None:
                for item in idsw_by_gt.get(effective_gt_id, []):
                    if item["frame"] > max(ev["frame"], effective_frame if effective_frame is not None else ev["frame"]):
                        future_idsw_frame = item["frame"]
                        future_idsw_new_trk = item["curr_trk_id"]
                        break

            row = dict(ev)
            row.update(
                {
                    "curr_gt_id": curr_gt_id,
                    "visible_frame": visible_frame,
                    "visible_gt_id": visible_gt_id,
                    "effective_frame": effective_frame,
                    "effective_gt_id": effective_gt_id,
                    "prev_gt_frame": prev_gt_frame,
                    "prev_gt_id": prev_gt_id,
                    "continued_from_prev_gt": bool(continued_from_prev),
                    "future_idsw": future_idsw_frame is not None,
                    "future_idsw_frame": future_idsw_frame,
                    "future_idsw_new_trk": future_idsw_new_trk,
                }
            )
            enriched.append(row)
    return enriched


def print_summary(rows):
    print("=" * 80)
    print("L2.5 Hit Quality Analysis")
    print("=" * 80)
    print(f"Total L2.5 hit events: {len(rows)}")
    current_matchable = [r for r in rows if r["curr_gt_id"] is not None]
    analyzable = [r for r in rows if r["effective_gt_id"] is not None]
    print(f"Events with current-frame GT match: {len(current_matchable)}")
    print(f"Events analyzable via first visible tracker frame: {len(analyzable)}")
    print("-" * 80)

    by_dt = defaultdict(list)
    for row in rows:
        by_dt[row["dt"]].append(row)

    print("Per-dt summary:")
    for dt in sorted(by_dt):
        group = by_dt[dt]
        analyzable_group = [r for r in group if r["effective_gt_id"] is not None]
        same_prev = sum(1 for r in analyzable_group if r["continued_from_prev_gt"])
        future_idsw = sum(1 for r in analyzable_group if r["future_idsw"])
        print(
            "  dt={}: hits={} analyzable={} continue_prev_gt={} ({:.1f}%) "
            "future_idsw={} ({:.1f}%)".format(
                dt,
                len(group),
                len(analyzable_group),
                same_prev,
                100.0 * same_prev / max(len(analyzable_group), 1),
                future_idsw,
                100.0 * future_idsw / max(len(analyzable_group), 1),
            )
        )

    print("-" * 80)
    bad_cases = [
        r for r in analyzable
        if (not r["continued_from_prev_gt"]) or r["future_idsw"]
    ]
    print(f"Potentially risky hit events: {len(bad_cases)}")
    for row in bad_cases[:15]:
        print(
            "  seq={} frame={} dt={} initial_track_id={} final_track_id={} curr_gt={} prev_gt={} future_idsw={}".format(
                row["seq"],
                row["frame"],
                row["dt"],
                row["initial_track_id"],
                row["track_id"],
                row["effective_gt_id"],
                row["prev_gt_id"],
                row["future_idsw_frame"] if row["future_idsw"] else "",
            )
        )


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "seq",
        "frame",
        "initial_track_id",
        "track_id",
        "dt",
        "decay",
        "tsu",
        "confirmed",
        "state",
        "hits",
        "curr_gt_id",
        "visible_frame",
        "visible_gt_id",
        "effective_frame",
        "effective_gt_id",
        "prev_gt_frame",
        "prev_gt_id",
        "continued_from_prev_gt",
        "future_idsw",
        "future_idsw_frame",
        "future_idsw_new_trk",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def main():
    events = parse_events(LOG_PATH)
    if not events:
        print("No L2.5 hit event log found or no valid events parsed.")
        print(f"Expected log path: {os.path.abspath(LOG_PATH)}")
        return

    enriched = enrich_events(events)
    print_summary(enriched)
    write_csv(CSV_PATH, enriched)
    print("-" * 80)
    print(f"Saved per-event CSV to: {os.path.abspath(CSV_PATH)}")


if __name__ == "__main__":
    main()
