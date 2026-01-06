from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json, re
from transformers import pipeline

from langchain.schema import SystemMessage, HumanMessage

from tasks import Task
from agents import MachineAgent, _now
from prompts import Assignment_Prompt, Conflict_Resolution_prompt
from vector_store import VectorStore

class MasterScheduler:
    def __init__(self, machines: List[MachineAgent], vector_store: VectorStore = None):
        self.machines = {m.id: m for m in machines}
        self.llm = pipeline("text-generation", model="gpt2")  # Use default device
        self.vector_store = vector_store
        # attaching the llms to each machine here 
        for m in machines:
            if m.llm is None:
                m.llm = self.llm

    def machines_overview_text(self) -> str:
        parts = []
        for m in self.machines.values():
            parts.append(f"- {m.id} ({m.name}) capabilities={m.capabilities}\n  schedule:\n{m.describe_schedule()}\n")
        return "\n".join(parts)

    def get_rag_context(self, query: str) -> str:
        """Retrieve relevant context from the vector store."""
        if not self.vector_store:
            return ""
        
        results = self.vector_store.search(query, top_k=3)
        if not results:
            return ""
        
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(f"{i}. {result['text']} (relevance: {result['score']:.2f})")
        
        return "\n".join(context_parts)

    def pick_machine_for_task(self, task: Task) -> str:
        # Query RAG system for relevant scheduling knowledge
        rag_query = f"scheduling {task.required_capability} task {task.name} duration {task.duration_minutes} minutes"
        rag_context = self.get_rag_context(rag_query)
        
        # Ensure each machine can handle only one task at a time
        earliest_start = datetime.now()
        best_machine = None
        best_start_time = None

        for m in self.machines.values():
            # Check if the machine has the required capability
            if task.required_capability not in m.capabilities:
                continue

            # Check if the machine is free for the task's duration
            candidate = m.next_free(earliest_start, task.duration_minutes)
            if candidate is not None:
                if best_start_time is None or candidate < best_start_time:
                    best_start_time = candidate
                    best_machine = m

        # Log RAG context if available
        if rag_context:
            print(f"\n[RAG Context for {task.id}]:\n{rag_context}\n")

        # Return the ID of the best machine or an empty string if none are available
        return best_machine.id if best_machine else ""

    def schedule_task(self, task: Task):
        machine_id = self.pick_machine_for_task(task)

        # Handle case where no suitable machine is found
        if not machine_id:
            raise ValueError(f"No suitable machine found for task {task.id} requiring capability '{task.required_capability}'.")

        machine = self.machines[machine_id]
        earliest = task.earliest_start or datetime.now()
        planned_start = machine.next_free(earliest, task.duration_minutes)

        # Ensure the task is not already assigned to the machine
        for s in machine.schedule:
            if s.task.id == task.id:
                raise ValueError(f"Task {task.id} is already assigned to machine {machine_id}.")

        # Get RAG context for machine planning
        machine_rag_query = f"{machine.name} execute {task.name} {task.required_capability}"
        machine_rag_context = self.get_rag_context(machine_rag_query)
        
        # ask machine how it plans to run the task (LLM)
        plan = machine.plan_execution_with_llm(task, machine_rag_context)
        # apply offset suggested by machine (in minutes)
        offset = plan.get("plan_start_offset_minutes", 0)
        planned_start = max(planned_start, datetime.now() + timedelta(minutes=offset))
        machine.assign_task(task, planned_start)
        # after assignment, check for conflicts on that machine (shouldn't happen because we used next_free)
        # but simulate conflict detection for demonstration (if schedule is messy)
        conflicts = self.detect_conflicts_on_machine(machine)
        if conflicts:
            self.resolve_conflicts(conflicts)

        return machine.id, planned_start

    def detect_conflicts_on_machine(self, machine: MachineAgent):
        # naive conflict detection: any overlapping scheduled entries on the machine
        entries = machine.schedule
        conflicts = []
        for i in range(len(entries)):
            for j in range(i+1, len(entries)):
                a = entries[i]; b = entries[j]
                if not (a.end <= b.start or b.end <= a.start):
                    conflicts.append((machine.id, a, b))
        return conflicts

    def resolve_conflicts(self, conflicts):
        # for each conflict, call LLM for a decision with RAG context
        for machine_id, a, b in conflicts:
            desc = f"Conflict on machine {machine_id}: tasks {a.task.id} ({a.start.strftime('%H:%M')}-{a.end.strftime('%H:%M')}) vs {b.task.id} ({b.start.strftime('%H:%M')}-{b.end.strftime('%H:%M')})"
            options = f"Machine schedule:\n{self.machines[machine_id].describe_schedule()}\n"
            
            # Get RAG context for conflict resolution
            rag_query = f"resolve conflict between {a.task.name} and {b.task.name} on machine {machine_id}"
            rag_context = self.get_rag_context(rag_query)
            
            if rag_context:
                options += f"\nRelevant Knowledge:\n{rag_context}\n"
                print(f"\n[RAG Context for Conflict Resolution]:\n{rag_context}\n")
            
            prompt = Conflict_Resolution_prompt.format(
                conflict_description=desc,
                options_overview=options
            )
            sys_msg = SystemMessage(content="You are the master scheduler, resolve conflict.")
            human_msg = HumanMessage(content=prompt)
            resp = self.llm.invoke([sys_msg, human_msg])
            text = getattr(resp, "content", str(resp))
            try:
                decision = json.loads(text)
            except Exception:
                m = re.search(r"(\{.*\})", text, re.S)
                if m:
                    try:
                        decision = json.loads(m.group(1))
                    except Exception:
                        decision = {"action": "delay", "task_to_adjust": b.task.id, "delay_minutes": 10, "reassign_to_machine_id": None}
                else:
                    decision = {"action": "delay", "task_to_adjust": b.task.id, "delay_minutes": 10, "reassign_to_machine_id": None}
            # apply decision
            if decision["action"] == "delay":
                self._apply_delay(machine_id, decision["task_to_adjust"], int(decision.get("delay_minutes", 0)))
            elif decision["action"] == "reassign":
                self._apply_reassign(decision["task_to_adjust"], decision.get("reassign_to_machine_id"))

    def _apply_delay(self, machine_id, task_id, minutes):
        machine = self.machines[machine_id]
        for s in machine.schedule:
            if s.task.id == task_id:
                s.start = s.start + timedelta(minutes=minutes)
                s.end = s.end + timedelta(minutes=minutes)
        machine.schedule.sort(key=lambda x: x.start)

    def _apply_reassign(self, task_id, to_machine_id):
        # remove task from current machine and add to to_machine after earliest possible
        src = None
        entry = None
        for m in self.machines.values():
            for s in m.schedule:
                if s.task.id == task_id:
                    src = m
                    entry = s
                    break
            if src:
                break
        if not src or not entry:
            return
        src.schedule = [s for s in src.schedule if s.task.id != task_id]
        target = self.machines.get(to_machine_id)
        if not target:
            # fallback: pick heuristic machine
            to_machine_id = self._heuristic_pick(entry.task)
            target = self.machines[to_machine_id]
        start = target.next_free_after(entry.start, entry.task.duration_minutes)
        target.assign_task_at(entry.task, start)

