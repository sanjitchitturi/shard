import json
import time
import random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shard_core import ShardCore, _hash_embedding, _cosine, _estimate_tokens


PALETTE = {
    "bg": "#0d1117",
    "card": "#161b22",
    "border": "#30363d",
    "muted": "#8b949e",
    "text": "#e6edf3",
    "blue": "#58a6ff",
    "green": "#3fb950",
    "red": "#f78166",
    "yellow": "#d29922",
    "purple": "#a371f7",
}

CORPUS_TOPICS = {
    "python": ["python is a programming language used for data science and web development",
               "python supports object-oriented and functional programming paradigms",
               "the python standard library is extensive and includes many useful modules"],
    "music": ["music theory studies harmony melody and rhythm in compositions",
              "the violin is a stringed instrument used in classical music ensembles",
              "jazz emerged in the early twentieth century in new orleans"],
    "cooking": ["italian cooking emphasizes fresh ingredients and simple preparation",
                "sourdough bread requires a fermented starter culture for leavening",
                "knife skills are essential for efficient kitchen preparation work"],
    "history": ["the renaissance was a period of cultural revival in europe",
                "ancient rome built extensive roads and aqueducts across its empire",
                "world war two reshaped political alliances across the globe"],
    "science": ["quantum mechanics describes physical phenomena at atomic scales",
                "natural selection drives evolutionary change in populations",
                "general relativity describes gravity as curvature of spacetime"],
    "travel": ["japan offers a unique blend of traditional and modern culture",
               "iceland is known for geothermal hot springs and glaciers",
               "patagonia spans both argentina and chile with dramatic landscapes"],
}


def _style_axes(ax):
    ax.set_facecolor(PALETTE["card"])
    ax.tick_params(colors=PALETTE["muted"], labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(PALETTE["border"])
    ax.grid(True, alpha=0.2, color=PALETTE["border"])


def _generate_messages(n: int, seed: int = 42) -> list[tuple[str, str]]:
    random.seed(seed)
    msgs = []
    topics = list(CORPUS_TOPICS.keys())
    for i in range(n):
        topic = random.choice(topics)
        base = random.choice(CORPUS_TOPICS[topic])
        msg = f"{base} also message number {i} continuing the discussion"
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append((role, msg))
    return msgs


def naive_token_count(messages: list[tuple[str, str]]) -> int:
    return sum(_estimate_tokens(content) for _, content in messages)


def benchmark_compression():
    print("[1/5] benchmarking compression at multiple scales")
    sizes = [50, 100, 200, 500, 1000]
    results = []
    for n in sizes:
        s = ShardCore(max_active_tokens=500, block_size=8)
        msgs = _generate_messages(n)
        for role, content in msgs:
            s.add_message(content, role)
        m = s.get_metrics()
        naive = naive_token_count(msgs)
        results.append({
            "n_messages": n,
            "raw_tokens": m.raw_tokens,
            "managed_tokens": m.active_tokens + m.compressed_tokens,
            "naive_tokens": naive,
            "compression_ratio": m.compression_ratio,
            "memory_blocks": m.memory_blocks,
            "tokens_saved": m.tokens_saved,
        })
        print(f"  n={n:5d} raw={m.raw_tokens:6d} managed={m.active_tokens + m.compressed_tokens:5d} ratio={m.compression_ratio:.3f}")
    return results


def benchmark_retrieval():
    print("[2/5] benchmarking retrieval accuracy and speed")
    sizes = [100, 500, 1000, 2000]
    results = []
    queries = [
        ("python programming language code", "python"),
        ("music violin classical theory", "music"),
        ("cooking italian recipe ingredients", "cooking"),
        ("history ancient rome europe", "history"),
        ("science quantum physics theory", "science"),
        ("travel japan culture tourism", "travel"),
    ]
    for n in sizes:
        s = ShardCore(max_active_tokens=500, block_size=8)
        for role, content in _generate_messages(n):
            s.add_message(content, role)

        correct = 0
        latencies = []
        for query, expected_topic in queries:
            t0 = time.perf_counter()
            top = s.retrieve_relevant(query, top_k=3)
            latencies.append((time.perf_counter() - t0) * 1000)
            if top and expected_topic in top[0]["content"].lower():
                correct += 1

        results.append({
            "n_messages": n,
            "accuracy": correct / len(queries),
            "avg_latency_ms": round(np.mean(latencies), 3),
            "p95_latency_ms": round(np.percentile(latencies, 95), 3),
        })
        print(f"  n={n:5d} accuracy={correct/len(queries):.2%} avg_latency={np.mean(latencies):.2f}ms")
    return results


def benchmark_memory_efficiency():
    print("[3/5] comparing SHARD vs naive memory usage")
    sizes = [100, 500, 1000, 2000, 5000]
    results = []
    for n in sizes:
        s = ShardCore(max_active_tokens=500, block_size=8)
        msgs = _generate_messages(n)
        for role, content in msgs:
            s.add_message(content, role)
        m = s.get_metrics()
        naive = naive_token_count(msgs)
        managed = m.active_tokens + m.compressed_tokens
        results.append({
            "n_messages": n,
            "naive_tokens": naive,
            "shard_tokens": managed,
            "savings_pct": round(100 * (1 - managed / naive), 1),
            "memory_blocks": m.memory_blocks,
        })
        print(f"  n={n:5d} naive={naive:6d} shard={managed:5d} savings={100*(1-managed/naive):.1f}%")
    return results


def benchmark_speed():
    print("[4/5] benchmarking per-message processing speed")
    s = ShardCore(max_active_tokens=500, block_size=8)
    msgs = _generate_messages(2000)
    timings = []
    for role, content in msgs:
        t0 = time.perf_counter()
        s.add_message(content, role)
        timings.append((time.perf_counter() - t0) * 1000)
    timings = np.array(timings)
    result = {
        "n_messages": len(msgs),
        "avg_ms": round(float(np.mean(timings)), 3),
        "median_ms": round(float(np.median(timings)), 3),
        "p95_ms": round(float(np.percentile(timings, 95)), 3),
        "p99_ms": round(float(np.percentile(timings, 99)), 3),
        "max_ms": round(float(np.max(timings)), 3),
        "timings": timings.tolist(),
    }
    print(f"  avg={result['avg_ms']}ms p95={result['p95_ms']}ms max={result['max_ms']}ms")
    return result


def benchmark_approach_comparison():
    print("[5/5] comparing SHARD vs simplified vs naive approaches")
    n = 1000
    msgs = _generate_messages(n)

    s_full = ShardCore(max_active_tokens=500, alpha=0.4, beta=0.3, gamma=0.3, block_size=8)
    for role, c in msgs:
        s_full.add_message(c, role)
    m_full = s_full.get_metrics()

    s_simple = ShardCore(max_active_tokens=500, alpha=1.0, beta=0.0, gamma=0.0, block_size=8)
    for role, c in msgs:
        s_simple.add_message(c, role)
    m_simple = s_simple.get_metrics()

    naive = naive_token_count(msgs)

    return {
        "naive_tokens": naive,
        "simplified_tokens": m_simple.active_tokens + m_simple.compressed_tokens,
        "shard_tokens": m_full.active_tokens + m_full.compressed_tokens,
        "naive_blocks": 0,
        "simplified_blocks": m_simple.memory_blocks,
        "shard_blocks": m_full.memory_blocks,
    }


def plot_compression(data, out_path):
    plt.rcParams.update({"figure.facecolor": PALETTE["bg"], "axes.facecolor": PALETTE["card"], "text.color": PALETTE["text"]})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), facecolor=PALETTE["bg"])

    sizes = [d["n_messages"] for d in data]
    naive = [d["naive_tokens"] for d in data]
    managed = [d["managed_tokens"] for d in data]
    ratios = [d["compression_ratio"] for d in data]

    x = np.arange(len(sizes))
    w = 0.35
    ax1.bar(x - w/2, naive, w, label="naive", color=PALETTE["red"])
    ax1.bar(x + w/2, managed, w, label="SHARD", color=PALETTE["green"])
    ax1.set_xticks(x)
    ax1.set_xticklabels(sizes)
    ax1.set_xlabel("n messages", color=PALETTE["text"])
    ax1.set_ylabel("tokens", color=PALETTE["text"])
    ax1.set_title("token usage: SHARD vs naive", color=PALETTE["text"], fontsize=11)
    ax1.legend(facecolor=PALETTE["card"], edgecolor=PALETTE["border"], labelcolor=PALETTE["text"])
    _style_axes(ax1)

    ax2.plot(sizes, ratios, "o-", color=PALETTE["blue"], linewidth=2, markersize=8)
    ax2.set_xlabel("n messages", color=PALETTE["text"])
    ax2.set_ylabel("compression ratio", color=PALETTE["text"])
    ax2.set_title("compression ratio across scales", color=PALETTE["text"], fontsize=11)
    ax2.set_ylim(0, 1.0)
    _style_axes(ax2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, facecolor=PALETTE["bg"], bbox_inches="tight")
    plt.close()


def plot_retrieval(data, out_path):
    plt.rcParams.update({"figure.facecolor": PALETTE["bg"], "axes.facecolor": PALETTE["card"], "text.color": PALETTE["text"]})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), facecolor=PALETTE["bg"])

    sizes = [d["n_messages"] for d in data]
    accuracy = [d["accuracy"] * 100 for d in data]
    latencies = [d["avg_latency_ms"] for d in data]

    ax1.bar(range(len(sizes)), accuracy, color=PALETTE["green"])
    ax1.set_xticks(range(len(sizes)))
    ax1.set_xticklabels(sizes)
    ax1.set_xlabel("n messages", color=PALETTE["text"])
    ax1.set_ylabel("accuracy (%)", color=PALETTE["text"])
    ax1.set_title("retrieval accuracy across scales", color=PALETTE["text"], fontsize=11)
    ax1.set_ylim(0, 105)
    _style_axes(ax1)

    ax2.plot(sizes, latencies, "o-", color=PALETTE["yellow"], linewidth=2, markersize=8)
    ax2.set_xlabel("n messages", color=PALETTE["text"])
    ax2.set_ylabel("avg latency (ms)", color=PALETTE["text"])
    ax2.set_title("retrieval latency vs corpus size", color=PALETTE["text"], fontsize=11)
    _style_axes(ax2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, facecolor=PALETTE["bg"], bbox_inches="tight")
    plt.close()


def plot_memory(data, out_path):
    plt.rcParams.update({"figure.facecolor": PALETTE["bg"], "axes.facecolor": PALETTE["card"], "text.color": PALETTE["text"]})
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=PALETTE["bg"])

    sizes = [d["n_messages"] for d in data]
    naive = [d["naive_tokens"] for d in data]
    shard = [d["shard_tokens"] for d in data]

    ax.fill_between(sizes, naive, alpha=0.3, color=PALETTE["red"], label="naive")
    ax.fill_between(sizes, shard, alpha=0.3, color=PALETTE["green"], label="SHARD")
    ax.plot(sizes, naive, color=PALETTE["red"], linewidth=2)
    ax.plot(sizes, shard, color=PALETTE["green"], linewidth=2)

    ax.set_xlabel("n messages", color=PALETTE["text"])
    ax.set_ylabel("token usage", color=PALETTE["text"])
    ax.set_title("memory usage scaling: SHARD vs naive", color=PALETTE["text"], fontsize=11)
    ax.legend(facecolor=PALETTE["card"], edgecolor=PALETTE["border"], labelcolor=PALETTE["text"])
    _style_axes(ax)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, facecolor=PALETTE["bg"], bbox_inches="tight")
    plt.close()


def plot_speed(data, out_path):
    plt.rcParams.update({"figure.facecolor": PALETTE["bg"], "axes.facecolor": PALETTE["card"], "text.color": PALETTE["text"]})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), facecolor=PALETTE["bg"])

    timings = data["timings"]
    ax1.plot(timings, color=PALETTE["blue"], linewidth=0.7, alpha=0.6)
    ax1.set_xlabel("message index", color=PALETTE["text"])
    ax1.set_ylabel("processing time (ms)", color=PALETTE["text"])
    ax1.set_title(f"per-message latency over {len(timings)} messages", color=PALETTE["text"], fontsize=11)
    _style_axes(ax1)

    labels = ["avg", "median", "p95", "p99", "max"]
    values = [data["avg_ms"], data["median_ms"], data["p95_ms"], data["p99_ms"], data["max_ms"]]
    colors = [PALETTE["blue"], PALETTE["green"], PALETTE["yellow"], PALETTE["red"], PALETTE["purple"]]
    ax2.bar(labels, values, color=colors)
    ax2.set_ylabel("latency (ms)", color=PALETTE["text"])
    ax2.set_title("latency percentiles", color=PALETTE["text"], fontsize=11)
    for i, v in enumerate(values):
        ax2.text(i, v + max(values) * 0.02, f"{v:.2f}", ha="center", color=PALETTE["text"], fontsize=8)
    _style_axes(ax2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, facecolor=PALETTE["bg"], bbox_inches="tight")
    plt.close()


def write_report(all_results: dict, out_path: str):
    comp = all_results["compression"]
    ret = all_results["retrieval"]
    mem = all_results["memory"]
    speed = all_results["speed"]
    cmp = all_results["comparison"]

    avg_ratio = np.mean([d["compression_ratio"] for d in comp])
    avg_acc = np.mean([d["accuracy"] for d in ret])
    max_savings = max(d["savings_pct"] for d in mem)

    md = f"""# SHARD Analysis Report

## Executive Summary

- average compression ratio across scales: {avg_ratio:.3f}
- average retrieval accuracy: {avg_acc:.1%}
- maximum memory savings vs naive: {max_savings:.1f}%
- average per-message latency: {speed["avg_ms"]} ms
- p95 latency: {speed["p95_ms"]} ms
- tested up to {max(d["n_messages"] for d in mem):,} messages

## Compression Analysis

| messages | raw tokens | managed | ratio | blocks |
|---|---|---|---|---|
"""
    for d in comp:
        md += f"| {d['n_messages']} | {d['raw_tokens']} | {d['managed_tokens']} | {d['compression_ratio']:.3f} | {d['memory_blocks']} |\n"

    md += "\n## Retrieval Performance\n\n| messages | accuracy | avg latency (ms) | p95 latency (ms) |\n|---|---|---|---|\n"
    for d in ret:
        md += f"| {d['n_messages']} | {d['accuracy']:.1%} | {d['avg_latency_ms']} | {d['p95_latency_ms']} |\n"

    md += "\n## Memory Efficiency\n\n| messages | naive tokens | SHARD tokens | savings |\n|---|---|---|---|\n"
    for d in mem:
        md += f"| {d['n_messages']} | {d['naive_tokens']} | {d['shard_tokens']} | {d['savings_pct']}% |\n"

    md += f"""\n## Processing Speed (n={speed["n_messages"]})

- avg: {speed["avg_ms"]} ms
- median: {speed["median_ms"]} ms
- p95: {speed["p95_ms"]} ms
- p99: {speed["p99_ms"]} ms
- max: {speed["max_ms"]} ms

## Approach Comparison (n=1000)

| approach | tokens | memory blocks |
|---|---|---|
| naive (no compression) | {cmp["naive_tokens"]} | {cmp["naive_blocks"]} |
| simplified (relevance only) | {cmp["simplified_tokens"]} | {cmp["simplified_blocks"]} |
| SHARD (full) | {cmp["shard_tokens"]} | {cmp["shard_blocks"]} |

## Conclusion

SHARD reduces token usage by {(1 - cmp["shard_tokens"]/cmp["naive_tokens"])*100:.1f}% compared to naive context handling at 1000 messages, while maintaining {avg_acc:.0%} retrieval accuracy and sub-{max(speed["avg_ms"], 1)}ms per-message processing latency.
"""
    Path(out_path).write_text(md)


def main():
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("SHARD comprehensive analysis")
    print("=" * 60 + "\n")

    results = {
        "compression": benchmark_compression(),
        "retrieval": benchmark_retrieval(),
        "memory": benchmark_memory_efficiency(),
        "speed": benchmark_speed(),
        "comparison": benchmark_approach_comparison(),
    }

    print("\ngenerating visualizations...")
    plot_compression(results["compression"], out_dir / "compression_comparison.png")
    plot_retrieval(results["retrieval"], out_dir / "retrieval_performance.png")
    plot_memory(results["memory"], out_dir / "memory_efficiency.png")
    plot_speed(results["speed"], out_dir / "speed_analysis.png")

    speed_for_json = {k: v for k, v in results["speed"].items() if k != "timings"}
    results_for_json = {**results, "speed": speed_for_json}
    (out_dir / "comprehensive_results.json").write_text(json.dumps(results_for_json, indent=2))

    print("writing report...")
    write_report(results, out_dir / "SHARD_Analysis_Report.md")

    print(f"\nall outputs written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
