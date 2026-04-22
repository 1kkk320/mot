from __future__ import annotations

import ast
import math
import os
import re
from collections import defaultdict
from statistics import mean, median


LOG_PATH = os.path.join("logs", "l25_covariance_diag.log")

LINE_RE = re.compile(
    r"\[L2\.5 CovDiag\] seq=(?P<seq>\S+) frame=(?P<frame>-?\d+) "
    r"track_id=(?P<track_id>-?\d+) det_frame=(?P<det_frame>-?\d+) "
    r"dt=(?P<dt>\d+) decay=(?P<decay>[0-9eE+\-.]+) "
    r"diag_before=(?P<before>\[[^\]]*\]) "
    r"diag_after_update=(?P<after_update>\[[^\]]*\]) "
    r"diag_after_fast_forward=(?P<after_ff>\[[^\]]*\])"
)


def safe_ratio(num: float, den: float) -> float:
    if abs(den) < 1e-12:
        return math.inf if num > 0 else 1.0
    return num / den


def parse_line(line: str):
    m = LINE_RE.search(line.strip())
    if not m:
        return None
    try:
        before = list(ast.literal_eval(m.group("before")))
        after_update = list(ast.literal_eval(m.group("after_update")))
        after_ff = list(ast.literal_eval(m.group("after_ff")))
    except Exception:
        return None

    return {
        "seq": m.group("seq"),
        "frame": int(m.group("frame")),
        "track_id": int(m.group("track_id")),
        "det_frame": int(m.group("det_frame")),
        "dt": int(m.group("dt")),
        "decay": float(m.group("decay")),
        "before": before,
        "after_update": after_update,
        "after_ff": after_ff,
    }


def summarize_ratios(events):
    ff_ratios = []
    update_ratios = []
    max_ff_ratios = []
    suspicious = []
    by_dt = defaultdict(list)

    for ev in events:
        before = ev["before"]
        after_update = ev["after_update"]
        after_ff = ev["after_ff"]
        dim = min(len(before), len(after_update), len(after_ff))
        if dim == 0:
            continue

        ff_vec = [safe_ratio(after_ff[i], before[i]) for i in range(dim)]
        update_vec = [safe_ratio(after_update[i], before[i]) for i in range(dim)]

        finite_ff = [x for x in ff_vec if math.isfinite(x)]
        finite_update = [x for x in update_vec if math.isfinite(x)]
        max_ff = max(ff_vec) if ff_vec else 1.0

        if finite_ff:
            ff_ratios.extend(finite_ff)
            by_dt[ev["dt"]].extend(finite_ff)
        if finite_update:
            update_ratios.extend(finite_update)
        max_ff_ratios.append(max_ff)

        if max_ff >= 5.0:
            suspicious.append((max_ff, ev))

    suspicious.sort(key=lambda x: x[0], reverse=True)
    return ff_ratios, update_ratios, max_ff_ratios, suspicious, by_dt


def fmt_stats(values):
    if not values:
        return "NA"
    return "{:.3f} / {:.3f} / {:.3f} / {:.3f}".format(
        mean(values), median(values), min(values), max(values)
    )


def main():
    print("=" * 80)
    print("L2.5 Covariance Diagnostic Analysis")
    print("=" * 80)

    if not os.path.exists(LOG_PATH):
        print(f"Log file not found: {os.path.abspath(LOG_PATH)}")
        print("Run `python main.py` first to generate L2.5 covariance diagnostics.")
        return

    events = []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            parsed = parse_line(line)
            if parsed is not None:
                events.append(parsed)

    if not events:
        print("No valid L2.5 covariance diagnostic events were found.")
        return

    ff_ratios, update_ratios, max_ff_ratios, suspicious, by_dt = summarize_ratios(events)

    print(f"Total matched L2.5 events logged: {len(events)}")
    print(f"Unique sequences: {len(set(ev['seq'] for ev in events))}")
    print(f"Unique tracks: {len(set((ev['seq'], ev['track_id']) for ev in events))}")
    print("-" * 80)
    print("Per-dimension ratio stats (after_update / before): mean / median / min / max")
    print(fmt_stats(update_ratios))
    print("Per-dimension ratio stats (after_fast_forward / before): mean / median / min / max")
    print(fmt_stats(ff_ratios))
    print("Event-wise max ratio stats (max over dimensions of after_fast_forward / before):")
    print(fmt_stats(max_ff_ratios))
    print("-" * 80)

    for thr in (2.0, 3.0, 5.0, 10.0):
        count = sum(1 for x in max_ff_ratios if x >= thr)
        pct = 100.0 * count / max(len(max_ff_ratios), 1)
        print(f"Events with max fast-forward ratio >= {thr:>4.1f}: {count:4d} ({pct:5.1f}%)")

    print("-" * 80)
    print("By dt: mean / median / min / max of per-dimension fast-forward ratios")
    for dt in sorted(by_dt):
        print(f"  dt={dt}: {fmt_stats(by_dt[dt])}")

    print("-" * 80)
    if suspicious:
        print("Top suspicious events (max fast-forward ratio >= 5):")
        for max_ff, ev in suspicious[:10]:
            print(
                "  seq={} frame={} track_id={} dt={} max_ratio={:.3f} "
                "before={} after_ff={}".format(
                    ev["seq"],
                    ev["frame"],
                    ev["track_id"],
                    ev["dt"],
                    max_ff,
                    ev["before"][:6],
                    ev["after_ff"][:6],
                )
            )
    else:
        print("No obviously suspicious covariance inflation events were found.")

    print("-" * 80)
    severe_pct = 100.0 * sum(1 for x in max_ff_ratios if x >= 5.0) / max(len(max_ff_ratios), 1)
    if severe_pct >= 20.0:
        print("Judgment: covariance inflation is significant; add covariance clipping or virtual update.")
    elif severe_pct >= 5.0:
        print("Judgment: covariance inflation exists in a non-trivial minority; clipping is worth trying.")
    else:
        print("Judgment: no strong evidence of widespread covariance explosion from fast-forward alone.")


if __name__ == "__main__":
    main()
