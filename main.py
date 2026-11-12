"""Predict Roblox's Up-and-Coming top 10 (make.fun bounty roblox-up-and-coming-forecast-01).

Reads $MAKEFUN_INPUT_DIR (default /input): task.json + history.ndjson.
Writes $MAKEFUN_OUTPUT_DIR/forecast.json (default /output), or the output name
from task.json: {"probabilities": [{"universe_id", "place", "p"}], "top10": [10 ids]}.

Model: rank-transition kernel.
  The chart is volatile (median top-10 order change: 6 places between reads a few
  hours apart), so persistence loses badly over a 72 h window. Instead, every
  pair of history reads (i -> j) is a sample of how a game at rank r moves after
  dt hours. Those transitions give P(rank k at T+dt | rank r at T), smoothed with
  Gaussian kernels over dt (bandwidth 9 h) and over the starting rank
  (bandwidth max(1.5, 0.15*r)). The forecast is that distribution averaged over
  the 72 hourly slots of the scored window, so one static forecast represents
  the time-averaged place distribution of every game. Games absent from the
  last chart are diffused from their last-seen rank over (time since seen +
  window). A small momentum term shifts each game's distribution by its recent
  rank velocity (last ~6 h), capped at 6 ranks. Per place, named probabilities
  are capped at 0.75 in total (the rest is left unnamed: brand-new games that
  never appeared in the history do crack the top 10, so the unnamed share is
  real insurance), and only rows with p >= 0.005 are named.

  Pure standard library, no network, no randomness, no prints: two runs on the
  same input produce byte-identical output. Runs in seconds on the practice
  history and well inside the 300 s limit on the denser final history.
"""

import json
import math
import os
from datetime import datetime, timezone

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

PLACES = 10
MAX_RANK = 100          # ranks above this are bucketed at 100; 0 = left the chart
DT_BUCKET_H = 6         # transition dt bucket width (hours)
DT_BANDWIDTH = 9.0      # Gaussian bandwidth over dt (hours)
HR_FACTOR = 0.15        # rank kernel bandwidth = max(1.5, HR_FACTOR * rank)
CAP = 0.75              # max total named probability per place
THRESH = 0.005          # only name rows with p >= THRESH
MOMENTUM = 0.08         # drift = MOMENTUM * velocity(rank/h) * 36 h, capped at +-6
VEL_WINDOW_H = 6        # velocity measured over roughly this many hours
MAX_AGE_H = 7 * 24      # ignore games last seen longer ago than this


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def load_history(input_dir):
    rows = []
    with open(os.path.join(input_dir, "history.ndjson"), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if not d.get("ok", True):
                continue
            chart = {}
            for g in d.get("up_and_coming") or []:
                try:
                    rank = int(g["rank"])
                    uid = int(g["universe_id"])
                except (KeyError, TypeError, ValueError):
                    continue
                chart[uid] = min(rank, MAX_RANK)
            if chart:
                rows.append({"at": parse_time(d["at"]), "chart": chart})
    rows.sort(key=lambda r: r["at"])
    return rows


def load_task(input_dir):
    with open(os.path.join(input_dir, "task.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def gauss(x, h):
    return math.exp(-0.5 * (x / h) ** 2)


class Kernel:
    """P(rank k at T+dt | rank r at T), from every (read i -> read j) pair."""

    def __init__(self, rows):
        # counters[(dt_bucket, r_start)][r_end] = count; r_end 0 = left the chart
        self.counters = {}
        n = len(rows)
        for i in range(n):
            ti = rows[i]["at"]
            ci = rows[i]["chart"]
            for j in range(i + 1, n):
                dt = (rows[j]["at"] - ti).total_seconds() / 3600.0
                db = min(int(dt // DT_BUCKET_H), 12)
                cj = rows[j]["chart"]
                for uid, r in ci.items():
                    key = (db, r)
                    c = self.counters.get(key)
                    if c is None:
                        c = self.counters[key] = {}
                    rj = cj.get(uid, 0)
                    c[rj] = c.get(rj, 0) + 1
        # bucket centres: 3, 9, ..., 75 (bucket 12 holds dt >= 72 h, centre 75)
        self.centres = [DT_BUCKET_H * b + DT_BUCKET_H / 2.0 for b in range(13)]
        # smoothed matrix M[bucket][r] = {rank: prob} incl. 0 = gone
        self.mat = {}
        for db in range(13):
            for r in range(1, MAX_RANK + 1):
                hr = max(1.5, HR_FACTOR * r)
                acc = {}
                tot = 0.0
                span = int(3 * hr) + 1
                for rp in range(max(1, r - span), min(MAX_RANK, r + span) + 1):
                    w = gauss(rp - r, hr)
                    if w < 0.01:
                        continue
                    c = self.counters.get((db, rp))
                    if not c:
                        continue
                    for rj, cnt in c.items():
                        acc[rj] = acc.get(rj, 0.0) + w * cnt
                        tot += w * cnt
                if tot > 0:
                    self.mat[(db, r)] = {k: v / tot for k, v in acc.items()}

    def dist(self, r, deltas):
        """Average P(rank | start rank r) over the given horizon deltas (hours).
        Returns a list over ranks 0..100 (0 = gone)."""
        out = [0.0] * (MAX_RANK + 1)
        if not deltas:
            return out
        for d in deltas:
            bw = [gauss(c - d, DT_BANDWIDTH) for c in self.centres]
            wsum = sum(bw)
            if wsum <= 0:
                continue
            for db, w in enumerate(bw):
                if w < 0.01:
                    continue
                m = self.mat.get((db, r))
                if not m:
                    continue
                f = w / wsum
                for k, p in m.items():
                    out[k] += f * p
        n = float(len(deltas))
        return [v / n for v in out]


def shift_dist(d, drift):
    """Shift a rank distribution (index 0 = gone) by `drift` ranks (float)."""
    if drift == 0.0:
        return d
    out = [0.0] * len(d)
    out[0] = d[0]
    src = d[1:]
    n = len(src)
    for i, p in enumerate(src):
        if p == 0.0:
            continue
        pos = i + drift
        lo = math.floor(pos)
        frac = pos - lo
        if 0 <= lo < n:
            out[1 + lo] += p * (1.0 - frac)
        if 0 <= lo + 1 < n:
            out[1 + lo + 1] += p * frac
    return out


def mean_rank(d):
    """Expected rank of a game from its place distribution (index 0 = gone).

    Ranks 1..100 count as themselves; a game that has left the chart (rank 0)
    counts as just off it, 101. The distribution is normalised first: the
    kernel can return less than full mass for a rank with sparse data.
    (The old version averaged only ranks 1..60 and lumped everything else,
    gone included, at 45 — which let a half-gone game outrank a solid
    mid-chart game in the top10 tie-break.)"""
    tot = sum(d)
    if tot <= 0:
        return float(MAX_RANK + 1)
    s = sum(d[k] * k for k in range(1, MAX_RANK + 1)) + d[0] * (MAX_RANK + 1)
    return s / tot


def build_forecast(rows, task):
    places = int(task.get("places", PLACES))
    last = rows[-1]["chart"]
    t_end = rows[-1]["at"]
    try:
        until = parse_time(task["history_until"])
        offset = max(0.0, (until - t_end).total_seconds() / 3600.0)
    except (KeyError, TypeError, ValueError):
        offset = 0.0
    deltas = [offset + 6 * i for i in range(13)]  # 0..72 h step 6 (+offset)

    kern = Kernel(rows)
    has_pairs = bool(kern.counters)

    # last-seen info for games absent from the last chart
    last_seen = {}
    for row in rows:
        for uid, r in row["chart"].items():
            last_seen[uid] = (row["at"], r)

    # recent rank velocity (rank/hour, positive = falling), over ~VEL_WINDOW_H
    vel = {}
    if has_pairs:
        target = t_end.timestamp() - VEL_WINDOW_H * 3600.0
        for uid in last:
            rs = [(row["at"], row["chart"][uid]) for row in rows if uid in row["chart"]]
            if not rs:
                continue
            past = [x for x in rs if x[0].timestamp() <= target]
            if past:
                (t1, r1), (t2, r2) = past[-1], rs[-1]
            elif len(rs) >= 2:
                (t1, r1), (t2, r2) = rs[-2], rs[-1]
            else:
                continue
            dtv = max((t2 - t1).total_seconds() / 3600.0, 0.1)
            vel[uid] = (r2 - r1) / dtv

    dists = {}
    if has_pairs:
        for uid, r in last.items():
            d = kern.dist(r, deltas)
            if uid in vel:
                drift = max(-6.0, min(6.0, MOMENTUM * vel[uid] * 36.0))
                d = shift_dist(d, drift)
            dists[uid] = d
        for uid, (at, r) in last_seen.items():
            if uid in dists:
                continue
            s = (t_end - at).total_seconds() / 3600.0
            if s > MAX_AGE_H:
                continue
            d = kern.dist(r, [offset + s + 6 * i for i in range(13)])
            dists[uid] = d

    if not dists:
        # degenerate history (a single read): fall back to persistence
        chart = sorted(last, key=lambda u: last[u])
        probs = []
        for place in range(1, places + 1):
            probs.append({"universe_id": chart[place - 1], "place": place, "p": 0.5})
            if place == 1:
                probs.append({"universe_id": chart[1], "place": place, "p": 0.3})
            else:
                probs.append({"universe_id": chart[place - 2], "place": place, "p": 0.15})
                if place < len(chart):
                    probs.append({"universe_id": chart[place], "place": place, "p": 0.15})
        return {"probabilities": probs, "top10": chart[:places]}

    mr = {uid: mean_rank(d) for uid, d in dists.items()}
    probs = []
    for k in range(1, places + 1):
        cands = [(dists[uid][k], uid) for uid in dists if dists[uid][k] >= THRESH]
        raw = sum(p for p, _ in cands)
        if raw > CAP:
            scale = CAP / raw
            cands = [(p * scale, uid) for p, uid in cands]
        for p, uid in sorted(cands, key=lambda x: (-x[0], x[1])):
            probs.append({"universe_id": int(uid), "place": k, "p": round(float(p), 6)})
    # final safety: no place may sum above 1 after rounding
    by_place = {}
    for row in probs:
        by_place.setdefault(row["place"], []).append(row)
    for k, rs in by_place.items():
        s = sum(r["p"] for r in rs)
        if s > 1.0:
            f = 1.0 / s
            for r in rs:
                r["p"] = round(r["p"] * f, 6)
    top10 = sorted(mr, key=lambda u: (mr[u], u))[:places]
    return {"probabilities": probs, "top10": [int(u) for u in top10]}


def main():
    task = load_task(INPUT_DIR)
    rows = load_history(INPUT_DIR)
    forecast = build_forecast(rows, task)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, task.get("output", "forecast.json"))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(forecast, sort_keys=True))
        f.write("\n")


if __name__ == "__main__":
    main()
