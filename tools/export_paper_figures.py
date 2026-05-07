import argparse
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.coordinate_transformation import compute_box_3dto2d
from visualization.visualization_3d import draw_projected_box3d


def compute_color_for_id(track_id):
    palette = (2**11 - 1, 2**15 - 1, 2**20 - 1)
    return tuple(int((p * (track_id * track_id - track_id + 1)) % 255) for p in palette)


def parse_frame_spec(frame_spec):
    frames = set()
    for part in frame_spec.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start, end = token.split("-", 1)
            start_i, end_i = int(start), int(end)
            if end_i < start_i:
                start_i, end_i = end_i, start_i
            frames.update(range(start_i, end_i + 1))
        else:
            frames.add(int(token))
    return sorted(frames)


def parse_id_spec(id_spec):
    if not id_spec:
        return None
    return {int(token.strip()) for token in id_spec.split(",") if token.strip()}


def load_tracking_result(result_file):
    frame_rows = defaultdict(list)
    with open(result_file, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) < 18:
                continue
            frame_id = int(fields[0])
            frame_rows[frame_id].append(
                {
                    "track_id": int(fields[1]),
                    "type": fields[2],
                    "ori": float(fields[5]),
                    "bbox2d": [float(v) for v in fields[6:10]],
                    "bbox3d": [float(v) for v in fields[10:17]],
                    "score": float(fields[17]),
                }
            )
    return frame_rows


def load_assoc_levels(assoc_file):
    assoc = {}
    if not os.path.exists(assoc_file):
        return assoc
    with open(assoc_file, "r", encoding="utf-8") as f:
        next(f, None)
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 3:
                continue
            assoc[(int(parts[0]), int(parts[1]))] = parts[2]
    return assoc


def draw_label_box(image, anchor, text, color):
    x1, y1 = anchor
    font_scale = 0.5
    thickness = 1
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    top_left = (x1, max(0, y1 - text_size[1] - 6))
    bottom_right = (x1 + text_size[0] + 4, max(0, y1))
    cv2.rectangle(image, top_left, bottom_right, color, -1, cv2.LINE_AA)
    cv2.putText(
        image,
        text,
        (top_left[0] + 2, max(text_size[1] + 1, bottom_right[1] - 3)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def render_single_view(image, calib_file, frame_rows, assoc_levels, frame_id, draw_2d, draw_3d, only_ids):
    canvas = image.copy()
    for row in frame_rows:
        track_id = row["track_id"]
        if only_ids is not None and track_id not in only_ids:
            continue

        color = compute_color_for_id(track_id)
        level = assoc_levels.get((frame_id, track_id), "NA")
        label = f"{track_id}/{level}"
        anchor = (int(row["bbox2d"][0]), int(row["bbox2d"][1]))

        if draw_3d:
            box3d_pts_2d = compute_box_3dto2d(np.asarray(row["bbox3d"], dtype=np.float32), calib_file)
            canvas = draw_projected_box3d(canvas, box3d_pts_2d, color=color, thickness=2)
            if box3d_pts_2d is not None:
                anchor = (int(box3d_pts_2d[4, 0]), int(box3d_pts_2d[4, 1]))

        if draw_2d:
            x1, y1, x2, y2 = [int(v) for v in row["bbox2d"]]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        draw_label_box(canvas, anchor, label, color)
    return canvas


def add_title_bar(image, title):
    bar_h = 36
    canvas = np.full((image.shape[0] + bar_h, image.shape[1], 3), 245, dtype=np.uint8)
    canvas[bar_h:, :, :] = image
    cv2.putText(
        canvas,
        title,
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )
    return canvas


def build_view_specs(args):
    specs = [(args.result_root, args.title or os.path.basename(os.path.normpath(args.result_root)))]
    for extra_root in args.compare_root:
        specs.append((extra_root, os.path.basename(os.path.normpath(extra_root))))
    return specs


def main():
    parser = argparse.ArgumentParser(description="Export targeted qualitative paper figures from saved MOT results.")
    parser.add_argument("--dataset", choices=["train", "test"], default="train")
    parser.add_argument("--seq", required=True, help="KITTI sequence id, e.g. 0000")
    parser.add_argument("--frames", required=True, help="Frame spec, e.g. 70,73,101-123")
    parser.add_argument("--result-root", default=os.path.join(PROJECT_ROOT, "results", "virconv_OCM"))
    parser.add_argument("--compare-root", action="append", default=[], help="Optional extra result root(s) for side-by-side comparison.")
    parser.add_argument("--output-dir", default=os.path.join(PROJECT_ROOT, "results", "paper_figures"))
    parser.add_argument("--title", default="", help="Display title for the primary result root.")
    parser.add_argument("--only-ids", default="", help="Optional comma-separated track ids to render.")
    parser.add_argument("--draw-2d", action="store_true")
    parser.add_argument("--draw-3d", action="store_true")
    args = parser.parse_args()

    draw_2d = True if not args.draw_2d and not args.draw_3d else args.draw_2d
    draw_3d = True if not args.draw_2d and not args.draw_3d else args.draw_3d
    seq = args.seq
    frames = parse_frame_spec(args.frames)
    only_ids = parse_id_spec(args.only_ids)

    data_root = os.path.join(PROJECT_ROOT, "datasets", "kitti", args.dataset)
    image_dir = os.path.join(data_root, "image_02", seq)
    calib_file = os.path.join(data_root, "calib", f"{seq}.txt")

    view_specs = build_view_specs(args)
    loaded_results = []
    for result_root, title in view_specs:
        result_file = os.path.join(result_root, "data", f"{seq}.txt")
        assoc_file = os.path.join(result_root, "assoc_levels", f"{seq}.csv")
        if not os.path.exists(result_file):
            raise FileNotFoundError(f"Tracking result not found: {result_file}")
        loaded_results.append(
            {
                "title": title,
                "result_root": result_root,
                "frames": load_tracking_result(result_file),
                "assoc": load_assoc_levels(assoc_file),
            }
        )

    out_dir = os.path.join(args.output_dir, seq)
    os.makedirs(out_dir, exist_ok=True)

    for frame_id in frames:
        img_path = os.path.join(image_dir, f"{frame_id:06d}.png")
        image = cv2.imread(img_path)
        if image is None:
            print(f"[skip] image not found: {img_path}")
            continue

        panels = []
        for bundle in loaded_results:
            rendered = render_single_view(
                image,
                calib_file,
                bundle["frames"].get(frame_id, []),
                bundle["assoc"],
                frame_id,
                draw_2d,
                draw_3d,
                only_ids,
            )
            panels.append(add_title_bar(rendered, f'{bundle["title"]} | seq {seq} frame {frame_id:06d}'))

        merged = panels[0] if len(panels) == 1 else cv2.hconcat(panels)
        out_path = os.path.join(out_dir, f"{frame_id:06d}.png")
        cv2.imwrite(out_path, merged)
        print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
