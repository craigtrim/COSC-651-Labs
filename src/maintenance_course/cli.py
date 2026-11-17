"""Course commands. No model service is called unless --mode live is supplied."""

import argparse
import asyncio
import importlib
import json
import sys
from pathlib import Path
from .data import LAB_ROOT, cases, case_by_id
from .experiments import run_case, save_runs
from . import baselines


# Issue #32: Command documentation makes the fixture default explicit.
def main():
    """Dispatch course commands, using fixture mode unless live mode is requested."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--workflow", choices=["direct", "fixed"], default="fixed")
    run.add_argument("--student", help="MODULE:FUNCTION in this lab's student_work")
    run.add_argument("--mode", choices=["fixture", "live"], default="fixture")
    run.add_argument("--case", default="D01", help="Development ID or all")
    run.add_argument("--output", default="artifacts/runs.json")
    sub.add_parser("mcp")
    sub.add_parser("flow")
    sub.add_parser("crew")
    for name in ("approve", "apply"):
        action = sub.add_parser(name)
        action.add_argument("--database", required=True)
        action.add_argument("--session", required=True)
        action.add_argument("--report", required=True)
        if name == "approve":
            action.add_argument("--reviewer", required=True)
    args = parser.parse_args()
    if args.command == "run":
        if args.student:
            sys.path.insert(0, str(LAB_ROOT))
            module, function = args.student.split(":", 1)
            if (
                module not in [f"week{i:02d}" for i in range(1, 9)]
                or not function.isidentifier()
            ):
                parser.error("Invalid student function")
            workflow = getattr(
                importlib.import_module("student_work." + module), function
            )
        else:
            workflow = getattr(baselines, args.workflow)
        selected = cases() if args.case == "all" else [case_by_id(args.case)]
        runs = [run_case(workflow, case, mode=args.mode) for case in selected]
        save_runs(args.output, runs)
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "cases": len(runs),
                    "errors": sum(bool(r["error"]) for r in runs),
                    "output": str(Path(args.output).resolve()),
                },
                indent=2,
            )
        )
        if any(r["error"] for r in runs):
            sys.exit(1)
    elif args.command == "mcp":
        from .mcp_client import inspect_server

        print(json.dumps(asyncio.run(inspect_server()), indent=2))
    elif args.command == "flow":
        from .crewai_bridge import DemonstrationFlow

        print(json.dumps(DemonstrationFlow().kickoff(), indent=2))
    elif args.command == "crew":
        from .crewai_bridge import run_crew

        print(json.dumps(run_case(run_crew, case_by_id("D01")), indent=2))
    else:
        from .state import StateStore
        from .contracts import checked_report

        report = checked_report(
            json.loads(Path(args.report).read_text(encoding="utf-8"))
        )
        store = StateStore(args.database)
        try:
            result = (
                store.approve(args.session, report, args.reviewer)
                if args.command == "approve"
                else store.apply_simulated(args.session, report)
            )
            print(json.dumps(result, indent=2))
        finally:
            store.close()


if __name__ == "__main__":
    main()
