import subprocess
import sys
import time
from pathlib import Path


def run(label: str, cmd: list[str]) -> bool:
    print("\n" + "=" * 70)
    print(f"running: {label}")
    print("=" * 70)
    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.perf_counter() - start
    status = "ok" if result.returncode == 0 else f"failed (exit {result.returncode})"
    print(f"\n{label}: {status} in {elapsed:.2f}s")
    return result.returncode == 0


def main():
    here = Path(__file__).parent
    scripts = [
        ("demo", [sys.executable, str(here / "demo.py")]),
        ("test_scenarios", [sys.executable, str(here / "test_scenarios.py")]),
        ("comprehensive_analysis", [sys.executable, str(here / "comprehensive_analysis.py")]),
    ]

    print("SHARD: running all scripts in sequence\n")
    results = []
    for label, cmd in scripts:
        ok = run(label, cmd)
        results.append((label, ok))

    print("\n" + "=" * 70)
    print("summary")
    print("=" * 70)
    for label, ok in results:
        marker = "PASS" if ok else "FAIL"
        print(f"  [{marker}] {label}")

    if all(ok for _, ok in results):
        print("\nall scripts completed successfully")
        print("see ./output/ for analysis artifacts")
    else:
        print("\nsome scripts failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
