"""
Test script to demonstrate the multi-agent production system
"""

import time
import random
import importlib.util
import sys

# Import modules with numeric prefixes
def import_module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Import all the agent modules
event_gen_module = import_module_from_file("event_generator", "00_event_generator.py")
job_agent_module = import_module_from_file("job_agent", "01_job_agent.py")
scheduler_module = import_module_from_file("scheduler_agent", "02_scheduler_agent.py")
machine_module = import_module_from_file("machine_agent", "03_machine_agent.py")
production_line_module = import_module_from_file("production_line_agent", "04_production_line_agent.py")
maintenance_module = import_module_from_file("maintenance_agent", "05_MaintenanceAlertAgent.py")

EventGenerator = event_gen_module.EventGenerator
JobAgent = job_agent_module.JobAgent
SchedulerAgent = scheduler_module.SchedulerAgent
MachineAgent = machine_module.MachineAgent
ProductionLineAgent = production_line_module.ProductionLineAgent
MaintenanceAlertAgent = maintenance_module.MaintenanceAlertAgent

def setup_production_system():
    """Setup the complete production system with all agents"""
    print("=" * 60)
    print("SETTING UP MULTI-AGENT PRODUCTION SYSTEM")
    print("=" * 60)
    
    # Create core agents
    event_generator = EventGenerator()
    scheduler = SchedulerAgent("MainScheduler")
    maintenance_agent = MaintenanceAlertAgent("MaintenanceAgent", scheduler)
    
    # Create machines
    machines = []
    machine_names = ['MachineA', 'MachineB', 'MachineC']
    
    for machine_name in machine_names:
        machine = MachineAgent(machine_name, scheduler, maintenance_agent)
        machines.append(machine)
        scheduler.register_machine(machine_name, machine)
        maintenance_agent.register_machine(machine_name, machine)
    
    # Create production line
    production_line = ProductionLineAgent("ProductionLine1", machines, scheduler)
    scheduler.register_production_line("ProductionLine1", production_line)
    
    # Create job agent
    job_agent = JobAgent("JobGenerator", scheduler, event_generator)
    
    # Register maintenance agent with scheduler
    scheduler.register_maintenance_agent(maintenance_agent)
    
    return {
        'event_generator': event_generator,
        'job_agent': job_agent,
        'scheduler': scheduler,
        'machines': machines,
        'production_line': production_line,
        'maintenance_agent': maintenance_agent
    }

def run_simulation(agents, steps=20):
    """Run the production system simulation"""
    print(f"\nStarting simulation for {steps} steps...")
    print("-" * 60)
    
    event_generator = agents['event_generator']
    job_agent = agents['job_agent']
    scheduler = agents['scheduler']
    machines = agents['machines']
    production_line = agents['production_line']
    maintenance_agent = agents['maintenance_agent']
    
    for step in range(steps):
        print(f"\n--- STEP {step + 1} ---")
        
        # Generate events occasionally
        if random.random() < 0.4:  # 40% chance each step
            event_generator.generate_event()
        
        # Sometimes generate rush jobs directly
        if random.random() < 0.2:  # 20% chance each step
            job_agent.simulate_job_generation(rush_probability=0.3)
        
        # Step all agents
        job_agent.step()
        scheduler.step()
        
        for machine in machines:
            machine.step()
        
        production_line.step()
        maintenance_agent.step()
        
        # Print system status every 5 steps
        if (step + 1) % 5 == 0:
            print_system_status(scheduler, machines, production_line, maintenance_agent)
        
        time.sleep(0.5)  # Small delay for readability

def print_system_status(scheduler, machines, production_line, maintenance_agent):
    """Print current system status"""
    print("\n" + "="*50)
    print("SYSTEM STATUS REPORT")
    print("="*50)
    
    # Scheduler status
    scheduler_status = scheduler.get_system_status()
    print(f"📊 SCHEDULER STATUS:")
    print(f"   Jobs in queue: {scheduler_status['jobs_in_queue']}")
    print(f"   Jobs completed: {scheduler_status['completed_jobs']}")
    print(f"   Idle machines: {scheduler_status['idle_machines']}")
    print(f"   Busy machines: {scheduler_status['busy_machines']}")
    print(f"   Broken machines: {scheduler_status['broken_machines']}")
    
    # Machine details
    print(f"\n🔧 MACHINE STATUS:")
    for machine in machines:
        metrics = machine.get_performance_metrics()
        print(f"   {metrics['machine_id']}: {metrics['status']} | "
              f"Jobs: {metrics['jobs_completed']} | "
              f"Breakdowns: {metrics['breakdown_count']} | "
              f"Reliability: {metrics['reliability']:.2f}")
    
    # Production line status
    line_metrics = production_line.get_line_metrics()
    print(f"\n🏭 PRODUCTION LINE:")
    print(f"   Line efficiency: {line_metrics['line_efficiency']:.2f}")
    print(f"   Active jobs: {line_metrics['active_jobs']}")
    print(f"   Bottleneck incidents: {line_metrics['bottleneck_incidents']}")
    
    # Maintenance status
    maintenance_metrics = maintenance_agent.get_maintenance_metrics()
    print(f"\n🔨 MAINTENANCE STATUS:")
    print(f"   Total repairs: {maintenance_metrics['total_repairs']}")
    print(f"   Active alerts: {maintenance_metrics['active_alerts']}")
    print(f"   Emergency incidents: {maintenance_metrics['emergency_incidents']}")
    print(f"   Repair queue: {maintenance_metrics['repair_queue_length']}")
    
    print("="*50)

def demonstrate_emergency_scenario(agents):
    """Demonstrate emergency scenario handling"""
    print("\n" + "🚨" * 20)
    print("DEMONSTRATING EMERGENCY SCENARIO")
    print("🚨" * 20)
    
    maintenance_agent = agents['maintenance_agent']
    machine = agents['machines'][0]  # First machine
    
    # Trigger an emergency
    emergency_msg = {
        'type': 'emergency_alert',
        'emergency_type': 'safety',
        'severity': 'critical',
        'description': 'Safety hazard detected'
    }
    
    maintenance_agent.inbox.append((machine.machine_id, emergency_msg))
    maintenance_agent.step()
    
    print("\nEmergency scenario completed. Check system status above.")

if __name__ == "__main__":
    # Setup the production system
    agents = setup_production_system()
    
    print("\n🎯 System initialized successfully!")
    print("Components:")
    print("✓ Event Generator")
    print("✓ Job Agent (Job Generator)")
    print("✓ Scheduler Agent")
    print("✓ 3 Machine Agents (A, B, C)")
    print("✓ Production Line Agent")
    print("✓ Maintenance & Alert Agent")
    
    # Run simulation
    run_simulation(agents, steps=15)
    
    # Demonstrate emergency handling
    demonstrate_emergency_scenario(agents)
    
    # Final system status
    scheduler = agents['scheduler']
    machines = agents['machines']
    production_line = agents['production_line']
    maintenance_agent = agents['maintenance_agent']
    
    print_system_status(scheduler, machines, production_line, maintenance_agent)
    
    print("\n🏁 Simulation completed!")
    print("=" * 60)
