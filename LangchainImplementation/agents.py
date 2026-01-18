from langchain_community.llms import GPT4All
from langchain_core.messages import HumanMessage, SystemMessage
from a2a import A2AClient, AgentCard, Skill, Task as A2ATask, TaskStatus


import json, re 
from typing import List , Dict , Any , Optional
from dataclasses import dataclass, field
from datetime import datetime , timedelta 
from transformers import pipeline
from enum import Enum

from tasks import Task
from prompts import Machine_Execution_Prompt


class MachineStatus(str, Enum):
    """Machine status states"""
    IDLE = "idle"
    BUSY = "busy"
    BROKEN = "broken"
    MAINTENANCE = "maintenance"


def _now():
    return datetime.now()

def _overlaps(a_start , a_end , b_start , b_end):
    return not (a_end <= b_start or b_end <= a_start )

@dataclass
class ScheduledEntry:
    task : Task
    start : datetime
    end : datetime


@dataclass
class MachineAgent:
    id : str
    name : str 
    capabilities : List[str]
    # A2A Communication
    a2a_client: A2AClient = field(default=None)
    agent_card: AgentCard = field(default=None)
    status: MachineStatus = MachineStatus.IDLE
    schedule : List[ScheduledEntry] = field(default_factory=list)
    llm : Any = field(default_factory=lambda: pipeline("text-generation", model="gpt2", device=-1))

    def is_free_between(self, start:datetime , end : datetime) -> bool:
        #loop thru the entire schedule and check if it overlaps or not. If it does return False otherwise return True ezz
        for s in self.schedule:
            if _overlaps(start , end , s.start ,s.end):
                return False
        return True
    
    def next_free(self , earliest: datetime , duration_minutes : int) -> datetime:
        candidate_start = earliest

        while True:
            candidate_end = candidate_start + timedelta(minutes=duration_minutes)
            conflict = False  # Initially, it's false

            for s in self.schedule:
                if _overlaps(candidate_start, candidate_end, s.start, s.end):
                    conflict = True
                    candidate_start = s.end  # Update start time to the end of the conflicting task
                    break

            # If no conflict was found, return the candidate start time
            if not conflict:
                return candidate_start
    def assign_task(self , task : Task , start : datetime):
        end = start + timedelta(minutes = task.duration_minutes)
        self.schedule.append(ScheduledEntry(task = task, start = start, end = end))

        self.schedule.sort(key = lambda x : x.start)
        return start,end
    
    def describe_schedule(self) -> str:
        lines = []
        for s in self.schedule:
            lines.append(f"{s.task.id}({s.task.name}): {s.start.strftime('%H:%M')} - {s.end.strftime('%H:%M')}")
        return "\n".join(lines) if lines else "empty"
    
    def plan_execution_with_llm(self, task : Task, rag_context: str = "") -> Dict[str, Any]:
        prompt_text = Machine_Execution_Prompt.format(
            machine_name=self.name,
            task_id=task.id,
            task_name=task.name,
            duration_minutes=task.duration_minutes,
            current_schedule=self.describe_schedule()
        )
        
        # Add RAG context if available
        if rag_context:
            prompt_text += f"\n\nRelevant Knowledge:\n{rag_context}"

        # Generate a response using the Hugging Face pipeline
        response = self.llm(prompt_text, max_new_tokens=50, num_return_sequences=1, truncation=True)
        text = response[0]["generated_text"]

        # Attempt to parse JSON from the response
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            m = re.search(r"(\{.*\})", text, re.S)
            if m:
                try:
                    return json.loads(m.group(1))
                except json.JSONDecodeError:
                    pass

        # Fallback: return a simple default plan
        return {"plan_start_offset_minutes": 0, "plan_duration_minutes": task.duration_minutes, "preconditions": []}

    def register_with_server(self, server_url: str = "http://localhost:8000"):
        """Register this machine with the A2A server"""
        self.a2a_client = A2AClient(server_url)
        
        # Create agent card with skills based on capabilities
        skills = [Skill(id=cap, name=cap.title()) for cap in self.capabilities]
        skills.append(Skill(id="status_report", name="Status Report"))
        
        self.agent_card = AgentCard(
            name=self.id,
            description=f"Machine agent: {self.name}",
            url=f"http://localhost:8000",  
            skills=skills
        )
        
        self.a2a_client.register(self.agent_card)
        print(f"[{self.id}] Registered with A2A server")

    def poll_inbox(self) -> list:
        """Check inbox for pending tasks"""
        if not self.a2a_client:
            return []
        
        pending = self.a2a_client.get_pending_tasks(self.id)
        return pending

    def process_task(self, a2a_task: A2ATask, rag_context: str = ""):
        """Process a task from the A2A inbox"""
        # Update status
        self.status = MachineStatus.BUSY
        self.a2a_client.update_task(a2a_task.id, TaskStatus.IN_PROGRESS)
        
        # Extract task details from message
        content = a2a_task.messages[0].content
        
        # Create internal Task object
        internal_task = Task(
            id=content.get("job_id", a2a_task.id),
            name=content.get("job_type", "unknown"),
            duration_minutes=content.get("duration_minutes", 30),
            required_capability=content.get("job_type", "")
        )
        
        # Use LLM to plan execution
        plan = self.plan_execution_with_llm(internal_task, rag_context)
        
        # Assign to schedule
        start, end = self.assign_task(internal_task, datetime.now())
        
        # Complete the A2A task
        self.a2a_client.update_task(
            a2a_task.id, 
            TaskStatus.COMPLETED,
            {
                "result": "completed",
                "scheduled_start": start.isoformat(),
                "scheduled_end": end.isoformat(),
                "plan": plan
            }
        )
        
        self.status = MachineStatus.IDLE
        print(f"[{self.id}] Completed task {internal_task.id}")
    
    def step(self):
        """Main step function - poll inbox and process pending tasks"""
        pending_tasks = self.poll_inbox()
        for task in pending_tasks:
            self.process_task(task)
        return len(pending_tasks)


@dataclass
class SchedulerAgent:
    """
    Scheduler Agent - Core decision-maker
    - Receives job arrivals from JobAgent
    - Discovers available machines via A2A
    - Assigns jobs to machines using LLM + RAG
    - Handles conflict resolution
    """
    id: str
    name: str = "MasterScheduler"
    
    # A2A Communication
    a2a_client: A2AClient = field(default=None)
    agent_card: AgentCard = field(default=None)
    
    # RAG System
    vector_store: Any = field(default=None)
    
    # LLM for decision making
    llm: Any = field(default_factory=lambda: pipeline("text-generation", model="gpt2", device=-1))
    
    # Tracking
    jobs_scheduled: List[Dict] = field(default_factory=list)
    jobs_completed: List[Dict] = field(default_factory=list)
    
    def register_with_server(self, server_url: str = "http://localhost:8000"):
        """Register scheduler with A2A server"""
        self.a2a_client = A2AClient(server_url)
        
        self.agent_card = AgentCard(
            name=self.id,
            description="Master scheduler for production line",
            url=server_url,
            skills=[
                Skill(id="schedule_task", name="Schedule Task", 
                      description="Assign tasks to machines"),
                Skill(id="resolve_conflict", name="Resolve Conflict",
                      description="Resolve scheduling conflicts"),
                Skill(id="get_status", name="Get Status",
                      description="Get system status"),
            ]
        )
        
        self.a2a_client.register(self.agent_card)
        print(f"[{self.id}] Scheduler registered with A2A server")
    
    def set_vector_store(self, vector_store):
        """Set the RAG vector store for knowledge retrieval"""
        self.vector_store = vector_store
        print(f"[{self.id}] RAG vector store configured")
    
    def get_rag_context(self, query: str) -> str:
        """Retrieve relevant context from vector store"""
        if not self.vector_store:
            return ""
        
        results = self.vector_store.search(query, top_k=3)
        if not results:
            return ""
        
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(f"{i}. {result['text']} (relevance: {result['score']:.2f})")
        
        return "\n".join(context_parts)
    
    def discover_machines(self, capability: str = None) -> List[AgentCard]:
        """Discover available machines, optionally filtered by capability"""
        if not self.a2a_client:
            return []
        
        if capability:
            return self.a2a_client.find_by_skill(capability)
        else:
            # Get all agents and filter out non-machines (like self)
            all_agents = self.a2a_client.discover_agents()
            return [a for a in all_agents if a.name != self.id]
    
    def poll_inbox(self) -> list:
        """Check inbox for job requests"""
        if not self.a2a_client:
            return []
        return self.a2a_client.get_pending_tasks(self.id)
    
    def pick_best_machine(self, job: Dict) -> Optional[str]:
        """
        Use LLM + RAG to pick the best machine for a job.
        Returns machine agent name or None if no suitable machine found.
        """
        required_capability = job.get("job_type", job.get("required_capability", ""))
        
        # Discover machines with required capability
        available_machines = self.discover_machines(required_capability)
        
        if not available_machines:
            print(f"[{self.id}] No machines found with capability: {required_capability}")
            return None
        
        # Get RAG context for scheduling decision
        rag_query = f"scheduling {required_capability} task duration {job.get('duration_minutes', 30)} minutes"
        rag_context = self.get_rag_context(rag_query)
        
        if rag_context:
            print(f"[{self.id}] RAG Context:\n{rag_context}")
        
        # For now, simple selection (first available)
        # TODO: Use LLM for smarter selection based on RAG context
        machine_names = [m.name for m in available_machines]
        
        # Build prompt for LLM
        prompt = f"""You are a production scheduler. Pick the best machine for this job.
Job: {job.get('job_type', 'unknown')} - Duration: {job.get('duration_minutes', 30)} minutes
Available machines: {machine_names}

Relevant Knowledge:
{rag_context if rag_context else 'No additional context available.'}

Return only the machine name, nothing else."""

        try:
            response = self.llm(prompt, max_new_tokens=20, num_return_sequences=1, truncation=True)
            text = response[0]["generated_text"]
            
            # Try to extract machine name from response
            for name in machine_names:
                if name in text:
                    return name
        except Exception as e:
            print(f"[{self.id}] LLM error: {e}")
        
        # Fallback: return first available machine
        return machine_names[0] if machine_names else None
    
    def assign_job_to_machine(self, job: Dict, machine_name: str) -> Optional[A2ATask]:
        """Send job to a machine via A2A"""
        if not self.a2a_client:
            return None
        
        task = self.a2a_client.send_task(
            receiver=machine_name,
            content={
                "action": "execute_job",
                "job_id": job.get("job_id", f"J{len(self.jobs_scheduled)+1:04d}"),
                "job_type": job.get("job_type", "unknown"),
                "duration_minutes": job.get("duration_minutes", 30),
                "priority": job.get("priority", "normal"),
            },
            sender=self.id
        )
        
        # Track scheduled job
        self.jobs_scheduled.append({
            "job": job,
            "machine": machine_name,
            "task_id": task.id,
            "scheduled_at": datetime.now().isoformat()
        })
        
        print(f"[{self.id}] Assigned job {job.get('job_id', 'unknown')} to {machine_name}")
        return task
    
    def schedule_job(self, job: Dict) -> bool:
        """
        Main scheduling method - picks machine and assigns job.
        Returns True if successful, False otherwise.
        """
        print(f"[{self.id}] Scheduling job: {job.get('job_id', 'unknown')} ({job.get('job_type', 'unknown')})")
        
        # Pick best machine
        machine_name = self.pick_best_machine(job)
        
        if not machine_name:
            print(f"[{self.id}] Failed to find suitable machine for job")
            return False
        
        # Assign to machine
        task = self.assign_job_to_machine(job, machine_name)
        return task is not None
    
    def process_inbox_job(self, a2a_task: A2ATask):
        """Process a job request from inbox (sent by JobAgent)"""
        content = a2a_task.messages[0].content
        
        # Mark as in progress
        self.a2a_client.update_task(a2a_task.id, TaskStatus.IN_PROGRESS)
        
        # Schedule the job
        success = self.schedule_job(content)
        
        # Complete the task
        if success:
            self.a2a_client.update_task(
                a2a_task.id,
                TaskStatus.COMPLETED,
                {"result": "scheduled", "job_id": content.get("job_id")}
            )
        else:
            self.a2a_client.update_task(
                a2a_task.id,
                TaskStatus.FAILED,
                {"error": "No suitable machine found"}
            )
    
    def step(self):
        """Main step function - process incoming job requests"""
        pending = self.poll_inbox()
        for task in pending:
            self.process_inbox_job(task)
        return len(pending)
    
    def get_system_status(self) -> Dict:
        """Get overall system status"""
        machines = self.discover_machines()
        return {
            "scheduler_id": self.id,
            "jobs_scheduled": len(self.jobs_scheduled),
            "jobs_completed": len(self.jobs_completed),
            "available_machines": len(machines),
            "machine_names": [m.name for m in machines]
        }


@dataclass  
class JobAgent:
    """
    Job Agent (Job Generator)
    - Generates new jobs (normal and rush orders)
    - Sends job creation messages to Scheduler via A2A
    - Simulates job priority, due dates, etc.
    """
    id: str
    name: str = "JobGenerator"
    scheduler_id: str = "Scheduler"  # Name of scheduler agent to send jobs to
    
    # A2A Communication
    a2a_client: A2AClient = field(default=None)
    agent_card: AgentCard = field(default=None)
    
    # Job generation config
    job_types: List[str] = field(default_factory=lambda: ["cutting", "welding", "painting"])
    job_counter: int = 0
    
    # Tracking
    jobs_generated: List[Dict] = field(default_factory=list)
    
    def register_with_server(self, server_url: str = "http://localhost:8000"):
        """Register job agent with A2A server"""
        self.a2a_client = A2AClient(server_url)
        
        self.agent_card = AgentCard(
            name=self.id,
            description="Job generator for production line",
            url=server_url,
            skills=[
                Skill(id="generate_job", name="Generate Job",
                      description="Generate new production jobs"),
                Skill(id="generate_rush_order", name="Generate Rush Order",
                      description="Generate high-priority rush orders"),
            ]
        )
        
        self.a2a_client.register(self.agent_card)
        print(f"[{self.id}] JobAgent registered with A2A server")
    
    def generate_job(self, job_type: str = None, rush: bool = False, 
                     duration_minutes: int = None) -> Dict:
        """Generate a new job"""
        import random
        
        self.job_counter += 1
        job_id = f"J{self.job_counter:04d}"
        
        # Random job type if not specified
        if job_type is None:
            job_type = random.choice(self.job_types)
        
        # Random duration if not specified
        if duration_minutes is None:
            duration_minutes = random.randint(15, 60) if not rush else random.randint(10, 30)
        
        # Priority and due date based on rush status
        if rush:
            priority = "high"
            due_date = random.randint(5, 15)  # Shorter due dates for rush
        else:
            priority = "normal"
            due_date = random.randint(20, 60)
        
        job = {
            "job_id": job_id,
            "job_type": job_type,
            "duration_minutes": duration_minutes,
            "priority": priority,
            "due_date": due_date,
            "rush": rush,
            "created_at": datetime.now().isoformat()
        }
        
        self.jobs_generated.append(job)
        print(f"[{self.id}] Generated {'RUSH ' if rush else ''}job {job_id}: {job_type} ({duration_minutes} min)")
        
        return job
    
    def send_job_to_scheduler(self, job: Dict) -> Optional[A2ATask]:
        """Send job to scheduler via A2A"""
        if not self.a2a_client:
            print(f"[{self.id}] Not connected to A2A server!")
            return None
        
        task = self.a2a_client.send_task(
            receiver=self.scheduler_id,
            content=job,
            sender=self.id
        )
        
        print(f"[{self.id}] Sent job {job['job_id']} to {self.scheduler_id}")
        return task
    
    def generate_and_send(self, job_type: str = None, rush: bool = False) -> Optional[A2ATask]:
        """Generate a job and immediately send to scheduler"""
        job = self.generate_job(job_type=job_type, rush=rush)
        return self.send_job_to_scheduler(job)
    
    def simulate_random_job(self, rush_probability: float = 0.2) -> Optional[A2ATask]:
        """Simulate generating a random job with chance of rush order"""
        import random
        rush = random.random() < rush_probability
        return self.generate_and_send(rush=rush)
    
    def step(self, generate_probability: float = 0.3, rush_probability: float = 0.2) -> int:
        """
        Main step function - randomly generate jobs.
        Returns number of jobs generated.
        """
        import random
        
        if random.random() < generate_probability:
            self.simulate_random_job(rush_probability)
            return 1
        return 0
    
    def get_stats(self) -> Dict:
        """Get job generation statistics"""
        rush_count = sum(1 for j in self.jobs_generated if j.get("rush", False))
        return {
            "agent_id": self.id,
            "total_jobs": len(self.jobs_generated),
            "rush_jobs": rush_count,
            "normal_jobs": len(self.jobs_generated) - rush_count,
            "job_types": list(set(j["job_type"] for j in self.jobs_generated))
        }