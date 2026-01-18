"""
Test the updated MachineAgent with A2A integration

Run this after the A2A server is running:
    python -m a2a.server

Then run:
    python test_machine_agent.py
"""

from agents import MachineAgent, MachineStatus
from a2a import A2AClient

def main():
    print("=" * 60)
    print("TESTING MACHINE AGENT WITH A2A")
    print("=" * 60)
    
    # Create a machine agent
    print("\n1. Creating MachineAgent...")
    machine = MachineAgent(
        id="M1",
        name="Cutter-1",
        capabilities=["cutting"]
    )
    print(f"   Created: {machine.id} ({machine.name})")
    print(f"   Status: {machine.status.value}")
    print(f"   Capabilities: {machine.capabilities}")
    
    # Register with A2A server
    print("\n2. Registering with A2A server...")
    machine.register_with_server("http://localhost:8000")
    
    # Check inbox (should be empty)
    print("\n3. Checking inbox...")
    pending = machine.poll_inbox()
    print(f"   Pending tasks: {len(pending)}")
    
    # Now let's send a task to this machine via A2A client
    print("\n4. Sending a task to the machine via A2A...")
    client = A2AClient("http://localhost:8000")
    task = client.send_task(
        receiver="M1",
        content={
            "action": "execute_job",
            "job_id": "J100",
            "job_type": "cutting",
            "material": "steel",
            "duration_minutes": 25,
        },
        sender="TestScheduler"
    )
    print(f"   Task sent: {task.id[:8]}...")
    print(f"   Task status: {task.status.value}")
    
    # Check inbox again
    print("\n5. Checking inbox again...")
    pending = machine.poll_inbox()
    print(f"   Pending tasks: {len(pending)}")
    
    # Process the task (this will use LLM - may take a moment)
    print("\n6. Processing pending tasks...")
    print("   (This may take a moment as it loads the LLM...)")
    processed = machine.step()
    print(f"   Processed {processed} task(s)")
    
    # Check final state
    print("\n7. Final machine state:")
    print(f"   Status: {machine.status.value}")
    print(f"   Schedule:\n   {machine.describe_schedule()}")
    
    # Check task status on server
    print("\n8. Checking task status on A2A server...")
    updated_task = client.get_task(task.id)
    print(f"   Task status: {updated_task.status.value}")
    print(f"   Messages: {len(updated_task.messages)}")
    
    print("\n" + "=" * 60)
    print("✓ TEST COMPLETED!")
    print("=" * 60)
    
    client.close()


if __name__ == "__main__":
    main()
