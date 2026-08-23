"""Resume-driver for wvs_values_prediction: relaunch from the last saved user
until `ending_row` users exist in the output. Makes transient API timeouts
non-fatal (the base runner aborts the whole job on an unretried timeout).

Requires storage_step=1 in the config so each completed user is checkpointed.
Output is append-only and one line per user, so #lines == #users done.

Usage: python scripts/run_with_resume.py --config <cfg.yaml> [--max-stuck 6]
"""
import argparse
import os
import subprocess
import sys
import tempfile

import yaml

RUNNER = "llm_behavior_adaptation.value_measurement.wvs_values_prediction"


def count_done(path):
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return sum(1 for _ in f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-stuck", type=int, default=6, help="abort after this many no-progress attempts")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    out = cfg["dialogue_output_file_path"]
    target = int(cfg["ending_row"])
    assert int(cfg.get("storage_step", 0)) == 1, "set storage_step: 1 for safe resume"

    stuck = 0
    while True:
        done = count_done(out)
        if done >= target:
            print(f"[resume] DONE: {done}/{target} users in {out}")
            return 0
        print(f"[resume] {done}/{target} done -> launching from row {done}")
        cfg["starting_row"] = done
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tf:
            yaml.safe_dump(cfg, tf)
            tmp = tf.name
        subprocess.run([sys.executable, "-m", RUNNER, "--config", tmp])  # exit ignored; we re-count
        os.unlink(tmp)

        if count_done(out) <= done:          # no forward progress this attempt
            stuck += 1
            print(f"[resume] no progress ({stuck}/{args.max_stuck})")
            if stuck >= args.max_stuck:
                print(f"[resume] ABORT: stuck at {done}/{target}")
                return 1
        else:
            stuck = 0


if __name__ == "__main__":
    sys.exit(main())
