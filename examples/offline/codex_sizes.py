"""Check historical req_bytes deltas using no-truncation controls of the same item shape.

A route count is not a Responses body. Compare increments between adjacent uniquely
joined requests within one rollout, holding route, item structure and message counts
fixed across the control and test increments. Emit only lengths and aggregate counts.
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from agent_snapshot_resume.agents.codex import DEFAULT_CATALOG, project_item, read_catalog
from agent_snapshot_resume.proxy.native_wire import codex_item


def size(obj):
    return len(json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode())


def lengths(item, limit):
    plain = codex_item(item)
    content = plain.get("content")
    blocks = tuple(b.get("type") for b in content if isinstance(b, dict)) if isinstance(content, list) else ()
    shape = (plain["type"], plain.get("role"), tuple(sorted(plain)), blocks)
    output = plain.get("output")
    if isinstance(output, str):
        texts = [output]
    elif isinstance(output, list):
        texts = [block["text"] for block in output if block["type"] == "input_text"]
    else:
        texts = []
    output_bytes = sum(len(text.encode("utf-8")) for text in texts)
    token_marker = any(re.search(r"…[0-9]+ tokens truncated…", text) for text in texts)
    return shape, size(plain), size(codex_item(project_item(item, limit))), output_bytes, token_marker


def usage_key(row):
    usage = row.get("usage") or {}
    return usage.get("input_tokens"), usage.get("output_tokens")


def route(row):
    return (row["model"], row.get("effort"), row.get("n_tools"), tuple(row.get("dropped_fields") or []),
            tuple(row.get("dropped_tools") or []))


def trial_intervals(trial, stats, model, limit):
    records = [json.loads(line) for path in sorted(trial.glob("agent/model-proxy/requests-*.jsonl"))
               for line in path.open() if line.strip()]
    matches = defaultdict(list)
    for i, r in enumerate(records):
        if isinstance(r.get("req_bytes"), int):
            stats["size_records"] += 1
            if r.get("model") == model and r.get("status") == 200 and None not in usage_key(r):
                matches[(r.get("session"), usage_key(r))].append((i, r))
    for path in sorted(trial.glob("agent/sessions/*/*/*/rollout-*.jsonl")):
        stats["rollouts"] += 1
        prev = None
        history = []
        with path.open("rb") as stream:
            for line in stream:
                row = json.loads(line)
                p = row["payload"]
                if row["type"] == "session_meta":
                    sid = p["id"]
                elif row["type"] == "compacted":
                    prev = None
                    history = [lengths(item, limit) for item in p.get("replacement_history") or []]
                elif row["type"] == "response_item":
                    item = lengths(p, limit)
                    history.append(item)
                    if p.get("type") in {"function_call_output", "custom_tool_call_output"}:
                        stats["stored_outputs"] += 1
                        stats["max_stored_text_bytes"] = max(stats["max_stored_text_bytes"], item[3])
                        stats["stored_outputs_with_token_marker"] += item[4]
                        stats["stored_outputs_changed_size"] += item[1] != item[2]
                elif row["type"] == "token_usage_record":
                    hits = matches[(sid, usage_key(p))]
                    if len(hits) != 1:
                        prev = None
                        continue
                    i, rec = hits[0]
                    n = rec["n_items"]
                    if not 0 <= n < len(history) or any(
                        h[0][0] not in ("function_call", "custom_tool_call", "reasoning", "message")
                        or (h[0][0] == "message" and h[0][1] != "assistant") for h in history[n:]
                    ):
                        prev = None
                        continue
                    stats["joined"] += 1
                    if prev and prev[0] + 1 == i and prev[1]["n_items"] < n and route(prev[1]) == route(rec):
                        old = prev[1]
                        added = history[old["n_items"]:n]
                        shape = (route(rec), rec["n_msgs"] - old["n_msgs"],
                                 rec["n_reasoning_replayed"] - old["n_reasoning_replayed"],
                                 tuple(h[0] for h in added))
                        yield (shape, rec["req_bytes"] - old["req_bytes"],
                               sum(h[1] for h in added), sum(h[2] for h in added), max(h[3] for h in added))
                    prev = i, rec


def compare(controls, intervals):
    stats = Counter(intervals=len(intervals))
    errors = defaultdict(list)
    typical_overhead = {shape: median(c.elements()) for shape, c in controls.items()}
    for shape, delta, plain, projected in intervals:
        control = controls[shape]
        if control.total() >= 2:
            stats["range_tested"] += 1
            lo, hi = min(control), max(control)
            raw_fits = lo <= delta - plain <= hi
            cut_fits = lo <= delta - projected <= hi
            stats["control_range_" + ("both" if raw_fits and cut_fits else
                                       "untruncated_only" if raw_fits else
                                       "projected_only" if cut_fits else "neither")] += 1
            typical = typical_overhead[shape]
            for name, length in (("untruncated", plain), ("projected", projected)):
                error = abs(delta - length - typical)
                errors["typical_" + name].append(error)
                stats[name + "_within_64_bytes"] += error <= 64
        # An invariant observed overhead also permits a stricter exact-byte comparison.
        if len(control) == 1 and control.total() >= 2:
            overhead = next(iter(control))
            stats["calibrated"] += 1
            for name, length in (("untruncated", plain), ("projected", projected)):
                error = delta - length - overhead
                errors[name].append(abs(error))
                stats[name + "_exact"] += error == 0
    return dict(stats=stats, median_absolute_error={k: median(v) for k, v in errors.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("jobs", type=Path, nargs="+")
    ap.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    ap.add_argument("--model", default="deepseek-flash")
    args = ap.parse_args()
    _, limit = read_catalog(args.catalog, args.model)
    stats = Counter()
    controls, affected = defaultdict(Counter), []
    short_controls, held_out = defaultdict(Counter), []
    for job in args.jobs:
        for trial in sorted(job.iterdir()):
            if not (trial / "agent/model-proxy").is_dir():
                continue
            for shape, delta, plain, projected, largest in trial_intervals(trial, stats, args.model, limit):
                stats["adjacent"] += 1
                if largest > 12000:
                    held_out.append((shape, delta, plain, projected))
                elif plain == projected:
                    short_controls[shape][delta - plain] += 1
                if plain == projected:
                    controls[shape][delta - plain] += 1
                else:
                    affected.append((shape, delta, plain, projected))
    stats["affected"] = len(affected)
    print(json.dumps(dict(model=args.model, catalog_tokens=limit, stats=stats,
                          size_changing=compare(controls, affected),
                          held_out_above_12000_bytes=compare(short_controls, held_out)), indent=2))


if __name__ == "__main__":
    main()
