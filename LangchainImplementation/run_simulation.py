# run_simulation.py
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from rich import print
from transformers import pipeline

from tasks import Task
from agents import MachineAgent
from scheduler import MasterScheduler
from knowledge_base import initialize_knowledge_base

load_dotenv()  # loads .env if present

# Initialize a local model pipeline
llm = pipeline("text-generation", model="gpt2", device=-1)  # Use CPU

# Example usage
prompt = "You are a scheduler. Pick the best machine for the task."
response = llm(prompt, max_length=50, num_return_sequences=1)
print(response[0]["generated_text"])

# Initialize RAG system with knowledge base
print("\n[cyan]Initializing RAG system with factory knowledge...[/cyan]")
vector_store = initialize_knowledge_base()
print(f"[green]Loaded {len(vector_store.documents)} documents into vector store[/green]\n")

# create machines
m1 = MachineAgent(id="M1", name="Cutter-1", capabilities=["cutting"], llm=llm)
m2 = MachineAgent(id="M2", name="Welder-1", capabilities=["welding"], llm=llm)
m3 = MachineAgent(id="M3", name="Painter-1", capabilities=["painting"], llm=llm)

scheduler = MasterScheduler([m1, m2, m3], vector_store=vector_store)

# create tasks (two production lines)
now = datetime.now()
t1 = Task(id="T1", name="Cut metal A", duration_minutes=30, earliest_start=now, required_capability="cutting")
t7 = Task(id="T7", name="Cut metal A", duration_minutes=30, earliest_start=now, required_capability="cutting")
t8 = Task(id="T8", name="Cut metal A", duration_minutes=45, earliest_start=now, required_capability="cutting")
t2 = Task(id="T2", name="Weld metal A", duration_minutes=45, earliest_start=now, required_capability="welding")
t3 = Task(id="T3", name="Paint part A", duration_minutes=20, earliest_start=now, required_capability="painting")

t4 = Task(id="U1", name="Cut metal B", duration_minutes=25, earliest_start=now + timedelta(minutes=5), required_capability="cutting")
t5 = Task(id="U2", name="Weld metal B", duration_minutes=40, earliest_start=now + timedelta(minutes=35), required_capability="welding")
t6 = Task(id="U3", name="Paint part B", duration_minutes=25, earliest_start=now + timedelta(minutes=80), required_capability="painting")

production_line_1 = [t1, t2, t3, t7, t8]
production_line_2 = [t4, t5, t6]

# schedule production lines' tasks sequentially via the master scheduler
print("[blue]Scheduling line 1...[/blue]")
for t in production_line_1:
    machine_id, start = scheduler.schedule_task(t)
    print(f"Assigned {t.id} -> {machine_id} at {start.strftime('%H:%M')}")

print("[red]Scheduling line 2...[/red]")
for t in production_line_2:
    machine_id, start = scheduler.schedule_task(t)
    print(f"Assigned {t.id} -> {machine_id} at {start.strftime('%H:%M')}")

# print final schedules
print("\n[bold]Final schedules:[/bold]")
for m in [m1, m2, m3]:
    print(f"\n[green]{m.id} ({m.name})[/green]")
    print(m.describe_schedule())

