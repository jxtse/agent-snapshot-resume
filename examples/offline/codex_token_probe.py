"""Verify the pinned CLI's token renderer with synthetic outputs in an isolated seed."""
import argparse
import json
import shutil
from pathlib import Path

from codex_probe import capture, pinned_cli
from agent_snapshot_resume.agents.codex import read_catalog, render_output


def block(text):
    return {"type": "input_text", "text": text}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, default=Path(".cache"))
    ap.add_argument("source", type=Path, help="catalog-bearing bundle")
    ap.add_argument("out", type=Path, help="fresh directory for the synthetic boundary probe")
    args = ap.parse_args()
    cli = pinned_cli(args.cache.resolve(), cached_only=True)
    meta = json.loads((args.source / "bundle.json").read_bytes())
    _, limit = read_catalog(args.source / "aux/model-catalog.json", meta["native_model"])
    cases = [
        ("ascii-40001", "a" * 40001), ("below", "a" * 47999), ("at", "a" * 48000),
        ("over-one", "a" * 48001), ("over-four", "a" * 48004), ("over-five", "a" * 48005),
        ("long", "b" * 80000), ("four-byte-scalars", "🙂" * 12001),
        ("split-both-ends", "a" * 23999 + "🙂🙂" + "b" * 23999),
        ("combining-and-newlines", "e\u0301\n" * 16000),
        ("shared-budget", [block("A" * 30000), block("B" * 30000)]),
        ("rounded-block-cost", [block("ab"), block("x" * 47998), block("y")]),
        ("exhausted", [block("x" * 48001), block("a"), block("bc")]),
        ("empty-blocks", [block(""), block("abc"), block("")]),
    ]
    args.out.mkdir(parents=True)
    for name in ("bundle.json", "proxy_spec.json"):
        shutil.copyfile(args.source / name, args.out / name)
    shutil.copytree(args.source / "aux", args.out / "aux")
    with (args.source / "session_trunc.jsonl").open() as stream:
        rows = [json.loads(line) for line in stream]
    items = [r["payload"] for r in rows if r["type"] == "response_item"]
    outputs = [p for p in items if p["type"] == "function_call_output"]
    if len(outputs) < len(cases):
        raise ValueError("seed has too few function outputs for the boundary suite")
    calls = {p["call_id"]: p for p in items if p["type"] == "function_call"}
    with (args.out / "session_trunc.jsonl").open("w") as stream:
        for i, (p, (_, sample)) in enumerate(zip(outputs, cases)):
            p["output"] = sample
            if i % 2:
                call = calls[p["call_id"]]
                call.update(type="custom_tool_call", name="apply_patch", input=call.pop("arguments"))
                p["type"] = "custom_tool_call_output"
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = capture(cli, args.out, meta)
    source = report.get("capture_source")
    if source not in ("http", "inference_started"):
        raise RuntimeError("no request captured in token probe")
    filename = "first_request_raw.json" if source == "http" else "traced_request_raw.json"
    actual = {p["call_id"]: p for p in json.loads((args.out / filename).read_bytes())["input"] if "output" in p}
    results = {}
    for p, (name, sample) in zip(outputs, cases):
        got = actual.get(p["call_id"], {})
        results[name] = got.get("type") == p["type"] and got.get("output") == render_output(sample, limit)
    print(json.dumps({"source": source, "cases": results, "exact": sum(results.values())}))
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
