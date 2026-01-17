# Multi-Agent Production Scheduling System - Foundation
# This is the core agent framework that you'll build upon

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import uuid
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enums and Data Classes
class AgentType(Enum):
    MACHINE = "machine"
    PRODUCTION_LINE = "production_line"
    SCHEDULER = "scheduler"
    COORDINATOR = "coordinator"

class EventType(Enum):
    MACHINE_BREAKDOWN = "machine_breakdown"
    RUSH_ORDER = "rush_order"
    MAINTENANCE_REQUIRED = "maintenance_required"
    PRODUCTION_COMPLETE = "production_complete"
    RESOURCE_SHORTAGE = "resource_shortage"

class JobPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class Job:
    id: str
    product_type: str
    quantity: int
    priority: JobPriority
    deadline: datetime
    estimated_duration: int  # minutes
    requirements: List[str]  # required machine types
    created_at: datetime = datetime.now()
    assigned_to: Optional[str] = None
    status: str = "pending"
    
    def to_dict(self):
        return {
            **asdict(self),
            'deadline': self.deadline.isoformat(),
            'created_at': self.created_at.isoformat(),
            'priority': self.priority.value
        }

@dataclass
class Event:
    id: str
    type: EventType
    source_agent: str
    data: Dict[str, Any]
    timestamp: datetime = datetime.now()
    processed: bool = False
    
    def to_dict(self):
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat(),
            'type': self.type.value
        }

@dataclass
class MachineState:
    id: str
    type: str
    status: str  # idle, running, broken, maintenance
    current_job: Optional[str]
    efficiency: float  # 0.0 to 1.0
    last_maintenance: datetime
    capabilities: List[str]
    queue: List[str]  # job IDs
    
class MessageBus:
    """Simple in-memory message bus for agent communication"""
    
    def __init__(self):
        self.subscribers: Dict[str, List[callable]] = {}
        self.events: List[Event] = []
    
    def subscribe(self, event_type: str, callback: callable):
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
    
    async def publish(self, event: Event):
        self.events.append(event)
        event_type = event.type.value
        
        if event_type in self.subscribers:
            for callback in self.subscribers[event_type]:
                try:
                    await callback(event)
                except Exception as e:
                    logger.error(f"Error in event callback: {e}")
    
    def get_events(self, since: datetime = None) -> List[Event]:
        if since:
            return [e for e in self.events if e.timestamp >= since]
        return self.events.copy()

class BaseAgent(ABC):
    """Base class for all agents in the system"""
    
    def __init__(self, agent_id: str, agent_type: AgentType, message_bus: MessageBus):
        self.id = agent_id
        self.type = agent_type
        self.message_bus = message_bus
        self.state = {}
        self.running = False
        
        # Subscribe to relevant events
        self.setup_subscriptions()
    
    def setup_subscriptions(self):
        """Override this to subscribe to specific events"""
        pass
    
    @abstractmethod
    async def process_event(self, event: Event):
        """Process incoming events"""
        pass
    
    async def send_event(self, event_type: EventType, data: Dict[str, Any], target: str = None):
        """Send an event through the message bus"""
        event = Event(
            id=str(uuid.uuid4()),
            type=event_type,
            source_agent=self.id,
            data={**data, 'target': target} if target else data
        )
        await self.message_bus.publish(event)
        logger.info(f"Agent {self.id} sent {event_type.value} event")
    
    async def start(self):
        """Start the agent"""
        self.running = True
        logger.info(f"Agent {self.id} ({self.type.value}) started")
        
        # Start the agent's main loop
        while self.running:
            try:
                await self.update()
                await asyncio.sleep(1)  # Update every second
            except Exception as e:
                logger.error(f"Error in agent {self.id}: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    async def stop(self):
        """Stop the agent"""
        self.running = False
        logger.info(f"Agent {self.id} stopped")
    
    @abstractmethod
    async def update(self):
        """Main agent update loop - override this"""
        pass

class MachineAgent(BaseAgent):
    """Agent representing a manufacturing machine"""
    
    def __init__(self, agent_id: str, machine_type: str, capabilities: List[str], message_bus: MessageBus):
        super().__init__(agent_id, AgentType.MACHINE, message_bus)
        
        self.machine_state = MachineState(
            id=agent_id,
            type=machine_type,
            status="idle",
            current_job=None,
            efficiency=random.uniform(0.8, 1.0),
            last_maintenance=datetime.now() - timedelta(days=random.randint(1, 30)),
            capabilities=capabilities,
            queue=[]
        )
        
        # Machine-specific parameters
        self.breakdown_probability = 0.001  # per update cycle
        self.maintenance_interval = timedelta(days=30)
    
    def setup_subscriptions(self):
        self.message_bus.subscribe("job_assignment", self.handle_job_assignment)
        self.message_bus.subscribe("maintenance_request", self.handle_maintenance_request)
    
    async def handle_job_assignment(self, event: Event):
        """Handle job assignment events"""
        if event.data.get('target') == self.id:
            job_id = event.data.get('job_id')
            if job_id and self.machine_state.status == "idle":
                self.machine_state.queue.append(job_id)
                logger.info(f"Machine {self.id} received job {job_id}")
    
    async def handle_maintenance_request(self, event: Event):
        """Handle maintenance requests"""
        if event.data.get('target') == self.id:
            self.machine_state.status = "maintenance"
            self.machine_state.last_maintenance = datetime.now()
            logger.info(f"Machine {self.id} entering maintenance")
    
    async def process_event(self, event: Event):
        """Process events relevant to this machine"""
        pass
    
    async def update(self):
        """Main machine update loop"""
        # Check for breakdowns
        if random.random() < self.breakdown_probability and self.machine_state.status == "running":
            await self.trigger_breakdown()
        
        # Check if maintenance is needed
        if datetime.now() - self.machine_state.last_maintenance > self.maintenance_interval:
            if self.machine_state.status == "idle":
                await self.request_maintenance()
        
        # Process jobs in queue
        if self.machine_state.queue and self.machine_state.status == "idle":
            await self.start_job(self.machine_state.queue.pop(0))
        
        # Simulate job completion
        if self.machine_state.status == "running" and self.machine_state.current_job:
            # Simulate job completion (simplified)
            if random.random() < 0.01:  # 1% chance per update to complete
                await self.complete_job()
    
    async def trigger_breakdown(self):
        """Simulate machine breakdown"""
        self.machine_state.status = "broken"
        await self.send_event(EventType.MACHINE_BREAKDOWN, {
            'machine_id': self.id,
            'machine_type': self.machine_state.type,
            'current_job': self.machine_state.current_job
        })
        logger.warning(f"Machine {self.id} broke down!")
    
    async def request_maintenance(self):
        """Request maintenance"""
        await self.send_event(EventType.MAINTENANCE_REQUIRED, {
            'machine_id': self.id,
            'last_maintenance': self.machine_state.last_maintenance.isoformat()
        })
    
    async def start_job(self, job_id: str):
        """Start processing a job"""
        self.machine_state.status = "running"
        self.machine_state.current_job = job_id
        logger.info(f"Machine {self.id} started job {job_id}")
    
    async def complete_job(self):
        """Complete current job"""
        completed_job = self.machine_state.current_job
        self.machine_state.current_job = None
        self.machine_state.status = "idle"
        
        await self.send_event(EventType.PRODUCTION_COMPLETE, {
            'machine_id': self.id,
            'job_id': completed_job
        })
        logger.info(f"Machine {self.id} completed job {completed_job}")

class SchedulerAgent(BaseAgent):
    """Central scheduler agent that coordinates production"""
    
    def __init__(self, agent_id: str, message_bus: MessageBus):
        super().__init__(agent_id, AgentType.SCHEDULER, message_bus)
        self.jobs: Dict[str, Job] = {}
        self.machines: Dict[str, MachineState] = {}
        self.pending_jobs: List[str] = []
    
    def setup_subscriptions(self):
        self.message_bus.subscribe("machine_breakdown", self.handle_breakdown)
        self.message_bus.subscribe("production_complete", self.handle_completion)
        self.message_bus.subscribe("rush_order", self.handle_rush_order)
    
    async def handle_breakdown(self, event: Event):
        """Handle machine breakdown events"""
        machine_id = event.data.get('machine_id')
        current_job = event.data.get('current_job')
        
        logger.warning(f"Handling breakdown of machine {machine_id}")
        
        # Reschedule current job if any
        if current_job:
            await self.reschedule_job(current_job)
        
        # Find alternative machines
        await self.find_alternative_machine(machine_id)
    
    async def handle_completion(self, event: Event):
        """Handle job completion events"""
        job_id = event.data.get('job_id')
        machine_id = event.data.get('machine_id')
        
        if job_id in self.jobs:
            self.jobs[job_id].status = "completed"
            logger.info(f"Job {job_id} completed on machine {machine_id}")
    
    async def handle_rush_order(self, event: Event):
        """Handle rush order events"""
        job_data = event.data.get('job')
        if job_data:
            # Create high priority job
            job = Job(**job_data)
            job.priority = JobPriority.CRITICAL
            self.jobs[job.id] = job
            self.pending_jobs.insert(0, job.id)  # Add to front of queue
            logger.info(f"Rush order {job.id} added to queue")
    
    async def process_event(self, event: Event):
        """Process scheduler events"""
        pass
    
    async def reschedule_job(self, job_id: str):
        """Reschedule a job to another machine"""
        if job_id in self.jobs:
            self.jobs[job_id].assigned_to = None
            self.jobs[job_id].status = "rescheduling"
            self.pending_jobs.append(job_id)
            logger.info(f"Job {job_id} rescheduled")
    
    async def find_alternative_machine(self, broken_machine_id: str):
        """Find alternative machines for jobs"""
        # This is where you'd implement intelligent rescheduling
        # For now, just log the need
        logger.info(f"Finding alternatives for broken machine {broken_machine_id}")
    
    async def update(self):
        """Main scheduler update loop"""
        # Simple job assignment logic
        if self.pending_jobs:
            job_id = self.pending_jobs[0]
            # Try to assign to available machine
            # (This would be more sophisticated in a real system)
            pass

# Factory class to tie everything together
class ProductionFactory:
    """Main factory class that manages the entire system"""
    
    def __init__(self):
        self.message_bus = MessageBus()
        self.agents: Dict[str, BaseAgent] = {}
        self.jobs: Dict[str, Job] = {}
        
    def add_machine(self, machine_id: str, machine_type: str, capabilities: List[str]):
        """Add a machine agent to the factory"""
        machine = MachineAgent(machine_id, machine_type, capabilities, self.message_bus)
        self.agents[machine_id] = machine
        return machine
    
    def add_scheduler(self, scheduler_id: str = "main_scheduler"):
        """Add a scheduler agent"""
        scheduler = SchedulerAgent(scheduler_id, self.message_bus)
        self.agents[scheduler_id] = scheduler
        return scheduler
    
    async def add_job(self, job: Job):
        """Add a new job to the system"""
        self.jobs[job.id] = job
        await self.message_bus.publish(Event(
            id=str(uuid.uuid4()),
            type=EventType.RUSH_ORDER,
            source_agent="factory",
            data={'job': job.to_dict()}
        ))
    
    async def simulate_rush_order(self):
        """Simulate a rush order event"""
        rush_job = Job(
            id=f"RUSH_{uuid.uuid4().hex[:8]}",
            product_type="Widget_A",
            quantity=100,
            priority=JobPriority.CRITICAL,
            deadline=datetime.now() + timedelta(hours=4),
            estimated_duration=120,
            requirements=["CNC", "Assembly"]
        )
        await self.add_job(rush_job)
        logger.info(f"Rush order {rush_job.id} generated!")
    
    async def start_all_agents(self):
        """Start all agents"""
        tasks = []
        for agent in self.agents.values():
            tasks.append(asyncio.create_task(agent.start()))
        return tasks
    
    async def stop_all_agents(self):
        """Stop all agents"""
        for agent in self.agents.values():
            await agent.stop()

# Example usage and demo
async def main():
    """Demo of the multi-agent system"""
    factory = ProductionFactory()
    
    # Add machines
    factory.add_machine("CNC_001", "CNC", ["drilling", "milling", "turning"])
    factory.add_machine("CNC_002", "CNC", ["drilling", "milling"])
    factory.add_machine("ASM_001", "Assembly", ["welding", "screwing", "testing"])
    
    # Add scheduler
    factory.add_scheduler()
    
    # Start all agents
    tasks = await factory.start_all_agents()
    
    # Simulate some events
    await asyncio.sleep(2)
    await factory.simulate_rush_order()
    
    # Run for a demo period
    await asyncio.sleep(10)
    
    # Stop all agents
    await factory.stop_all_agents()
    
    # Cancel all tasks
    for task in tasks:
        task.cancel()
    
    logger.info("Demo completed!")

if __name__ == "__main__":
    asyncio.run(main())