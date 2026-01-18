"""
Test the full multi-agent system with A2A communication

Flow: JobAgent -> Scheduler -> MachineAgent

Run this after starting the A2A server:
    python -m a2a.server

Then run:
    python test_full_system.py
"""

import time
from agents import MachineAgent, SchedulerAgent, JobAgent
from knowledge_base import initialize_knowledge_base


def main():
    print("=" * 70)
    print("  MULTI-AGENT PRODUCTION SYSTEM - FULL TEST")
    print("=" * 70)
    
    # ================================================================
    # 1. INITIALIZE RAG KNOWLEDGE BASE
    # ================================================================
    print("\n[SETUP] Initializing RAG knowledge base...")
    vector_store = initialize_knowledge_base()
    print(f"[SETUP] Loaded {len(vector_store.documents)} documents")
    
    # ================================================================
    # 2. CREATE AND REGISTER MACHINES
    # ================================================================
    print("\n[SETUP] Creating machine agents...")
    
    machines = [
        MachineAgent(id="M1", name="Cutter-1", capabilities=["cutting"]),
        MachineAgent(id="M2", name="Welder-1", capabilities=["welding"]),
        MachineAgent(id="M3", name="Painter-1", capabilities=["painting"]),
    ]
    
    for machine in machines:
        machine.register_with_server("http://localhost:8000")
    
    # ================================================================
    # 3. CREATE AND REGISTER SCHEDULER
    # ================================================================
    print("\n[SETUP] Creating scheduler agent...")
    
    scheduler = SchedulerAgent(id="Scheduler", name="MasterScheduler")
    scheduler.register_with_server("http://localhost:8000")
    scheduler.set_vector_store(vector_store)
    
    # ================================================================
    # 4. CREATE AND REGISTER JOB AGENT
    # ================================================================
    print("\n[SETUP] Creating job agent...")
    
    job_agent = JobAgent(
        id="JobGen", 
        name="JobGenerator",
        scheduler_id="Scheduler"  # Send jobs to this scheduler
    )
    job_agent.register_with_server("http://localhost:8000")
    
    # ================================================================
    # 5. GENERATE AND SEND JOBS
    # ================================================================
    print("\n" + "=" * 70)
    print("  GENERATING JOBS")
    print("=" * 70)
    
    # Generate specific jobs
    job_agent.generate_and_send(job_type="cutting", rush=False)
    job_agent.generate_and_send(job_type="welding", rush=True)
    job_agent.generate_and_send(job_type="painting", rush=False)
    
    # ================================================================
    # 6. SCHEDULER PROCESSES JOBS
    # ================================================================
    print("\n" + "=" * 70)
    print("  SCHEDULER PROCESSING JOBS")
    print("=" * 70)
    
    # Scheduler polls inbox and assigns to machines
    processed = scheduler.step()
    print(f"\n[Scheduler] Processed {processed} job(s)")
    
    # ================================================================
    # 7. MACHINES PROCESS TASKS
    # ================================================================
    print("\n" + "=" * 70)
    print("  MACHINES PROCESSING TASKS")
    print("=" * 70)
    
    for machine in machines:
        processed = machine.step()
        if processed > 0:
            print(f"[{machine.id}] Processed {processed} task(s)")
    
    # ================================================================
    # 8. SYSTEM STATUS
    # ================================================================
    print("\n" + "=" * 70)
    print("  SYSTEM STATUS")
    print("=" * 70)
    
    # Scheduler status
    status = scheduler.get_system_status()
    print(f"\n📊 Scheduler Status:")
    print(f"   Jobs scheduled: {status['jobs_scheduled']}")
    print(f"   Available machines: {status['available_machines']}")
    
    # Job agent stats
    job_stats = job_agent.get_stats()
    print(f"\n📝 Job Agent Stats:")
    print(f"   Total jobs: {job_stats['total_jobs']}")
    print(f"   Rush jobs: {job_stats['rush_jobs']}")
    print(f"   Normal jobs: {job_stats['normal_jobs']}")
    
    # Machine schedules
    print(f"\n🔧 Machine Schedules:")
    for machine in machines:
        schedule = machine.describe_schedule()
        print(f"   {machine.id} ({machine.name}):")
        if schedule == "empty":
            print(f"      (no tasks)")
        else:
            for line in schedule.split("\n"):
                print(f"      {line}")
    
    # ================================================================
    # 9. SIMULATION LOOP (optional)
    # ================================================================
    print("\n" + "=" * 70)
    print("  RUNNING SIMULATION (5 steps)")
    print("=" * 70)
    
    for step in range(5):
        print(f"\n--- Step {step + 1} ---")
        
        # Job agent randomly generates jobs
        jobs_generated = job_agent.step(generate_probability=0.5, rush_probability=0.3)
        if jobs_generated:
            print(f"[JobGen] Generated {jobs_generated} job(s)")
        
        # Scheduler processes inbox
        scheduler.step()
        
        # Machines process their tasks
        for machine in machines:
            machine.step()
        
        time.sleep(0.5)  # Small delay for readability
    
    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)
    
    print(f"\n📊 Jobs Generated: {len(job_agent.jobs_generated)}")
    print(f"📊 Jobs Scheduled: {len(scheduler.jobs_scheduled)}")
    
    print(f"\n🔧 Final Machine Schedules:")
    for machine in machines:
        schedule = machine.describe_schedule()
        tasks_count = len(machine.schedule)
        print(f"   {machine.id}: {tasks_count} task(s)")
        if schedule != "empty":
            for line in schedule.split("\n"):
                print(f"      {line}")
    
    print("\n" + "=" * 70)
    print("  ✓ TEST COMPLETED!")
    print("=" * 70)


if __name__ == "__main__":
    main()
