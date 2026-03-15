"""
End-to-end agentic benchmark: Micro instruction policy vs GPT text generator.

This benchmark runs the full A2A execution path:
JobAgent -> SchedulerAgent -> MachineAgent

It compares two modes:
- micro: USE_MICRO_LM=1 + USE_INSTRUCTION_POLICY=1
- gpt:   USE_MICRO_LM=0 + USE_INSTRUCTION_POLICY=0

Outputs:
- markdown table with end-to-end metrics
- optional markdown file via --output

Usage:
    python benchmark_end_to_end_agentic.py
    python benchmark_end_to_end_agentic.py --mode micro
    python benchmark_end_to_end_agentic.py --mode gpt
    python benchmark_end_to_end_agentic.py --output benchmark_e2e_results.md
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import time
from typing import Dict, List, Sequence

from a2a import A2AClient
from agents import JobAgent, MachineAgent, SchedulerAgent
from knowledge_base import initialize_knowledge_base


@dataclass(frozen=True)
class JobScenario:
    job_id: str
    job_type: str
    duration_minutes: int
    priority: str
    rush: bool


SCENARIOS: List[JobScenario] = [
    JobScenario("J9001", "cutting", 35, "normal", False),
    JobScenario("J9002", "welding", 24, "high", True),
    JobScenario("J9003", "painting", 42, "normal", False),
    JobScenario("J9004", "cutting", 18, "high", True),
    JobScenario("J9005", "welding", 30, "normal", False),
    JobScenario("J9006", "painting", 28, "high", True),
    JobScenario("J9007", "cutting", 40, "normal", False),
    JobScenario("J9008", "welding", 16, "high", True),
    JobScenario("J9009", "painting", 33, "normal", False),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run end-to-end Micro vs GPT benchmark.")
    parser.add_argument("--server-url", default="http://localhost:8000", help="A2A server URL")
    parser.add_argument(
        "--mode",
        choices=["micro", "gpt", "both", "cli"],
        default="cli",
        help="Execution mode: micro, gpt, both, or cli prompt.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_e2e_results.md"),
        help="Where to save markdown output table.",
    )
    return parser.parse_args()


def select_modes(mode_arg: str) -> List[str]:
    if mode_arg == "micro":
        return ["micro"]
    if mode_arg == "gpt":
        return ["gpt"]
    if mode_arg == "both":
        return ["micro", "gpt"]

    while True:
        print("\nSelect model mode:")
        print("1) Micro LM")
        print("2) GPT-2")
        choice = input("Enter choice (1 or 2): ").strip()
        if choice == "1":
            return ["micro"]
        if choice == "2":
            return ["gpt"]
        print("Invalid choice. Please enter 1 or 2.")


def ensure_server(server_url: str) -> None:
    client = A2AClient(server_url)
    try:
        _ = client.health_check()
    finally:
        client.close()


def machine_capability_map(machines: Sequence[MachineAgent]) -> Dict[str, List[str]]:
    return {m.id: list(m.capabilities) for m in machines}


def valid_plan(plan: Dict) -> bool:
    required = {"plan_start_offset_minutes", "plan_duration_minutes", "preconditions"}
    if not isinstance(plan, dict):
        return False
    if not required.issubset(plan.keys()):
        return False
    if not isinstance(plan.get("plan_start_offset_minutes"), int):
        return False
    if not isinstance(plan.get("plan_duration_minutes"), int):
        return False
    if not isinstance(plan.get("preconditions"), list):
        return False
    return True


def run_single_mode(mode: str, server_url: str) -> Dict[str, float]:
    if mode == "micro":
        os.environ["USE_MICRO_LM"] = "1"
        os.environ["USE_INSTRUCTION_POLICY"] = "1"
    elif mode == "gpt":
        os.environ["USE_MICRO_LM"] = "0"
        os.environ["USE_INSTRUCTION_POLICY"] = "0"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    vector_store = initialize_knowledge_base()

    machines = [
        MachineAgent(id="M1", name="Cutter-1", capabilities=["cutting"]),
        MachineAgent(id="M2", name="Welder-1", capabilities=["welding"]),
        MachineAgent(id="M3", name="Painter-1", capabilities=["painting"]),
    ]
    for machine in machines:
        machine.register_with_server(server_url)

    scheduler = SchedulerAgent(id="Scheduler", name="MasterScheduler")
    scheduler.register_with_server(server_url)
    scheduler.set_vector_store(vector_store)

    job_agent = JobAgent(id="JobGen", name="JobGenerator", scheduler_id="Scheduler")
    job_agent.register_with_server(server_url)

    send_start = time.perf_counter()
    for scenario in SCENARIOS:
        payload = {
            "job_id": scenario.job_id,
            "job_type": scenario.job_type,
            "duration_minutes": scenario.duration_minutes,
            "priority": scenario.priority,
            "rush": scenario.rush,
            "due_date": 20,
        }
        job_agent.send_job_to_scheduler(payload)

    scheduler_processed = scheduler.step()

    total_machine_processed = 0
    for machine in machines:
        total_machine_processed += machine.step()

    elapsed_ms = (time.perf_counter() - send_start) * 1000.0

    cap_map = machine_capability_map(machines)
    scheduled = len(scheduler.jobs_scheduled)

    capability_correct = 0
    for record in scheduler.jobs_scheduled:
        job_type = record.get("job", {}).get("job_type", "")
        machine_id = record.get("machine", "")
        if job_type in cap_map.get(machine_id, []):
            capability_correct += 1

    client = A2AClient(server_url)
    completed = 0
    valid_plans = 0
    plan_count = 0

    try:
        for record in scheduler.jobs_scheduled:
            task_id = record.get("task_id")
            if not task_id:
                continue
            task = client.get_task(task_id)
            if task.status.value == "completed":
                completed += 1
            # Last message should include execution payload with plan from machine process.
            if task.messages:
                content = task.messages[-1].content
                plan = content.get("plan") if isinstance(content, dict) else None
                if plan is not None:
                    plan_count += 1
                    if valid_plan(plan):
                        valid_plans += 1
    finally:
        client.close()

    total_jobs = len(SCENARIOS)
    success_rate = (100.0 * completed / total_jobs) if total_jobs else 0.0
    capability_accuracy = (100.0 * capability_correct / total_jobs) if total_jobs else 0.0
    plan_valid_rate = (100.0 * valid_plans / max(1, plan_count)) if plan_count else 0.0

    return {
        "mode": mode,
        "jobs_total": float(total_jobs),
        "scheduler_processed": float(scheduler_processed),
        "machine_processed": float(total_machine_processed),
        "jobs_scheduled": float(scheduled),
        "jobs_completed": float(completed),
        "success_rate": success_rate,
        "capability_accuracy": capability_accuracy,
        "plan_valid_rate": plan_valid_rate,
        "elapsed_ms": elapsed_ms,
    }


def format_table(rows: Sequence[Dict[str, float]]) -> str:
    lines = [
        "| Mode | Jobs | Scheduled | Completed | Success % | Capability match % | Valid plan % | End-to-end latency (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['mode']} | {int(row['jobs_total'])} | {int(row['jobs_scheduled'])} | {int(row['jobs_completed'])} | "
            f"{row['success_rate']:.2f} | {row['capability_accuracy']:.2f} | {row['plan_valid_rate']:.2f} | {row['elapsed_ms']:.2f} |"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    ensure_server(args.server_url)
    selected_modes = select_modes(args.mode)

    runs = []
    for mode in selected_modes:
        print(f"Running mode: {mode}")
        result = run_single_mode(mode, args.server_url)
        runs.append(result)

    table = format_table(runs)
    print("\nEnd-to-end benchmark results:\n")
    print(table)

    args.output.write_text(table + "\n", encoding="utf-8")
    print(f"\nSaved results to: {args.output}")


if __name__ == "__main__":
    main()
