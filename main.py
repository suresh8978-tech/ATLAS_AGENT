from __future__ import annotations

import argparse
import sys

from agent import analyze_job


def main() -> int:
    # parser = argparse.ArgumentParser(
    #     description="Analyze Ansible Automation Platform job events using a LangGraph agent."
    # )
    # parser.add_argument("--job-id", type=int, required=True, help="Automation job id")
    # args = parser.parse_args()

    try:
        # report = analyze_job(args.job_id)
        report = analyze_job(8295005)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
