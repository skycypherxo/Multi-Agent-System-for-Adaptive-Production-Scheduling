"""
A2A (Agent-to-Agent) Protocol Models
Based on Google's A2A specification for agent communication
https://github.com/google/A2A

Core concepts:
- AgentCard: Self-describing manifest of what an agent can do
- Task: A unit of work delegated to an agent
- Artifact: Output produced by completing a task
"""

from pydantic import BaseModel, Field
from typing import Any, Optional, List
from datetime import datetime
from enum import Enum
import uuid


# ============================================================
# AGENT CARD - "Who am I and what can I do?"
# ============================================================

class Skill(BaseModel):
    """
    A capability/skill that an agent possesses.
    
    Example:
        Skill(
            id="cutting",
            name="Metal Cutting",
            description="Precision cutting of metal sheets",
            input_schema={"material": "string", "dimensions": "object"}
        )
    """
    id: str
    name: str
    description: Optional[str] = None
    input_schema: Optional[dict] = None   # JSON Schema for expected inputs
    output_schema: Optional[dict] = None  # JSON Schema for outputs


class AgentCapabilities(BaseModel):
    """Technical capabilities of an agent"""
    streaming: bool = False              # Can stream responses?
    push_notifications: bool = False     # Can send push notifications?
    state_management: bool = True        # Maintains internal state?


class AgentCard(BaseModel):
    """
    A2A Agent Card - self-describing agent manifest.
    
    Each agent publishes this so others can discover:
    - What the agent does (skills)
    - How to reach it (url)
    - What it supports (capabilities)
    
    Example:
        AgentCard(
            name="CutterMachine-01",
            description="CNC cutting machine for metal processing",
            url="http://localhost:8001",
            skills=[Skill(id="cutting", name="Metal Cutting")]
        )
    """
    name: str
    description: Optional[str] = None
    url: str                                    # Agent's HTTP endpoint
    version: str = "1.0.0"
    capabilities: AgentCapabilities = Field(default_factory=AgentCapabilities)
    skills: List[Skill] = []
    
    def has_skill(self, skill_id: str) -> bool:
        """Check if agent has a specific skill"""
        return any(s.id == skill_id for s in self.skills)


# ============================================================
# TASK - "What work needs to be done?"
# ============================================================

class TaskStatus(str, Enum):
    """Status of a task in its lifecycle"""
    PENDING = "pending"           # Task received, not started
    IN_PROGRESS = "in-progress"   # Agent is working on it
    COMPLETED = "completed"       # Successfully finished
    FAILED = "failed"             # Failed with error
    CANCELLED = "cancelled"       # Cancelled by requester


class Message(BaseModel):
    """
    A message in a task conversation.
    
    role: "user" = requester (e.g., Scheduler asking Machine to do something)
          "agent" = executor (e.g., Machine responding with status)
    """
    role: str  # "user" or "agent"
    content: Any
    timestamp: datetime = Field(default_factory=datetime.now)


class Artifact(BaseModel):
    """
    Output produced by completing a task.
    
    Example: After a cutting task completes, the artifact might be:
        Artifact(
            name="cutting_result",
            type="application/json",
            data={"pieces_cut": 5, "quality_score": 0.95}
        )
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str = "application/json"  # MIME type
    name: Optional[str] = None
    data: Any = None


class Task(BaseModel):
    """
    A2A Task - a unit of work delegated to an agent.
    
    Lifecycle:
    1. Requester sends task with initial message -> PENDING
    2. Agent starts work -> IN_PROGRESS  
    3. Agent completes -> COMPLETED (with artifacts) or FAILED
    
    Example:
        task = Task(
            status=TaskStatus.PENDING,
            messages=[Message(role="user", content={"action": "cut", "material": "steel"})]
        )
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: TaskStatus = TaskStatus.PENDING
    messages: List[Message] = []
    artifacts: List[Artifact] = []
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    def add_message(self, role: str, content: Any) -> "Task":
        """Add a message to the task conversation"""
        self.messages.append(Message(role=role, content=content))
        self.updated_at = datetime.now()
        return self
    
    def add_artifact(self, name: str, data: Any, mime_type: str = "application/json") -> "Task":
        """Add an artifact (output) to the task"""
        self.artifacts.append(Artifact(name=name, data=data, type=mime_type))
        self.updated_at = datetime.now()
        return self
    
    def complete(self, result_data: Any = None) -> "Task":
        """Mark task as completed, optionally with result artifact"""
        self.status = TaskStatus.COMPLETED
        if result_data:
            self.add_artifact("result", result_data)
        self.updated_at = datetime.now()
        return self
    
    def fail(self, error: str) -> "Task":
        """Mark task as failed with error message"""
        self.status = TaskStatus.FAILED
        self.add_message("agent", {"error": error})
        self.updated_at = datetime.now()
        return self


# ============================================================
# API REQUEST/RESPONSE MODELS
# ============================================================

class SendTaskRequest(BaseModel):
    """Request to send a task to an agent"""
    task_id: Optional[str] = None  # If continuing existing task
    message: Message


class SendTaskResponse(BaseModel):
    """Response from sending a task"""
    task: Task


class GetTaskRequest(BaseModel):
    """Request to get task status"""
    task_id: str


class CancelTaskRequest(BaseModel):
    """Request to cancel a task"""
    task_id: str
    reason: Optional[str] = None
