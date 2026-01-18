"""
Example: How to use the A2A server with factory agents

Run this after starting the server:
    python -m a2a.server

Then run this example:
    python -m a2a.example
"""

from a2a import (
    A2AClient,
    AgentCard,
    Skill,
    TaskStatus,
)


def main():
    # Connect to A2A server
    client = A2AClient("http://localhost:8000")
    
    # Check server is running
    print("Checking server...")
    info = client.health_check()
    print(f"✓ Server running: {info['name']}\n")
    
    # ============================================================
    # 1. REGISTER AGENTS
    # ============================================================
    print("=" * 50)
    print("REGISTERING AGENTS")
    print("=" * 50)
    
    # Register Scheduler Agent
    scheduler_card = AgentCard(
        name="Scheduler",
        description="Master scheduler for production line",
        url="http://localhost:8001",  # Scheduler's own endpoint
        skills=[
            Skill(id="schedule_task", name="Schedule Task", 
                  description="Assign tasks to machines"),
            Skill(id="resolve_conflict", name="Resolve Conflict",
                  description="Resolve scheduling conflicts"),
        ]
    )
    result = client.register(scheduler_card)
    print(f"✓ {result['message']}")
    
    # Register Machine Agents
    for i, (name, skill) in enumerate([
        ("Cutter-01", "cutting"),
        ("Welder-01", "welding"),
        ("Painter-01", "painting"),
    ], start=1):
        machine_card = AgentCard(
            name=name,
            description=f"Machine agent for {skill} operations",
            url=f"http://localhost:{8001 + i}",
            skills=[
                Skill(id=skill, name=skill.title(), 
                      description=f"Performs {skill} operations"),
                Skill(id="status_report", name="Status Report",
                      description="Report current status"),
            ]
        )
        result = client.register(machine_card)
        print(f"✓ {result['message']}")
    
    # ============================================================
    # 2. DISCOVER AGENTS
    # ============================================================
    print("\n" + "=" * 50)
    print("DISCOVERING AGENTS")
    print("=" * 50)
    
    all_agents = client.discover_agents()
    print(f"Found {len(all_agents)} agents:")
    for agent in all_agents:
        skills = [s.id for s in agent.skills]
        print(f"  - {agent.name}: {skills}")
    
    # Find agents with cutting skill
    cutters = client.find_by_skill("cutting")
    print(f"\nAgents with 'cutting' skill: {[a.name for a in cutters]}")
    
    # ============================================================
    # 3. SEND TASKS
    # ============================================================
    print("\n" + "=" * 50)
    print("SENDING TASKS")
    print("=" * 50)
    
    # Scheduler sends a cutting task to Cutter-01
    task = client.send_task(
        receiver="Cutter-01",
        content={
            "action": "execute_job",
            "job_id": "J001",
            "job_type": "cutting",
            "material": "steel",
            "duration_minutes": 30,
        },
        sender="Scheduler"
    )
    print(f"✓ Task {task.id[:8]}... sent to Cutter-01")
    print(f"  Status: {task.status.value}")
    
    # Send another task
    task2 = client.send_task(
        receiver="Welder-01",
        content={
            "action": "execute_job",
            "job_id": "J002",
            "job_type": "welding",
            "material": "steel",
            "duration_minutes": 45,
        },
        sender="Scheduler"
    )
    print(f"✓ Task {task2.id[:8]}... sent to Welder-01")
    
    # ============================================================
    # 4. CHECK AGENT INBOXES
    # ============================================================
    print("\n" + "=" * 50)
    print("CHECKING INBOXES")
    print("=" * 50)
    
    for agent_name in ["Cutter-01", "Welder-01", "Painter-01"]:
        tasks = client.get_pending_tasks(agent_name)
        print(f"{agent_name}: {len(tasks)} pending task(s)")
        for t in tasks:
            content = t.messages[0].content if t.messages else {}
            print(f"  - {t.id[:8]}... : {content.get('action', 'unknown')}")
    
    # ============================================================
    # 5. SIMULATE AGENT PROCESSING
    # ============================================================
    print("\n" + "=" * 50)
    print("SIMULATING TASK PROCESSING")
    print("=" * 50)
    
    # Cutter-01 starts working on the task
    client.update_task(task.id, TaskStatus.IN_PROGRESS, {
        "status": "started",
        "message": "Cutting operation in progress"
    })
    print(f"Cutter-01: Task {task.id[:8]}... -> IN_PROGRESS")
    
    # Cutter-01 completes the task
    client.update_task(task.id, TaskStatus.COMPLETED, {
        "status": "completed",
        "result": "5 pieces cut successfully",
        "actual_duration": 28
    })
    print(f"Cutter-01: Task {task.id[:8]}... -> COMPLETED")
    
    # Check final task state
    final_task = client.get_task(task.id)
    print(f"\nFinal task state:")
    print(f"  Status: {final_task.status.value}")
    print(f"  Messages: {len(final_task.messages)}")
    
    print("\n✓ Example completed!")
    client.close()


if __name__ == "__main__":
    main()
