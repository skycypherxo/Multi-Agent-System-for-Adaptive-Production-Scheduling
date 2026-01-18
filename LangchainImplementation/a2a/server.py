"""
A2A Server - Central hub for agent communication

Endpoints:
    GET  /                          - Server info
    POST /agents/register           - Register an agent
    GET  /agents                    - List all agents
    GET  /agents/{agent_id}         - Get agent card
    GET  /agents/{agent_id}/.well-known/agent.json  - A2A standard discovery
    
    POST /tasks/send                - Send task to an agent
    GET  /tasks/{task_id}           - Get task status
    POST /tasks/{task_id}/cancel    - Cancel a task
    
    GET  /agents/{agent_id}/tasks   - Get all tasks for an agent (inbox)
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional
from datetime import datetime
import uvicorn

from .models import (
    AgentCard,
    Task,
    TaskStatus,
    Message,
    SendTaskRequest,
    SendTaskResponse,
    GetTaskRequest,
    CancelTaskRequest,
)


app = FastAPI(
    title="A2A Communication Server",
    description="Agent-to-Agent protocol server for production scheduling system",
    version="1.0.0"
)

# Enable CORS for dashboard/frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# IN-MEMORY STORAGE (replace with Redis/DB in production)
# ============================================================

# Registered agents: agent_name -> AgentCard
agent_registry: Dict[str, AgentCard] = {}

# All tasks: task_id -> Task
task_store: Dict[str, Task] = {}

# Agent inboxes: agent_name -> List[task_id]
agent_inboxes: Dict[str, List[str]] = {}


# ============================================================
# SERVER INFO
# ============================================================

@app.get("/")
def root():
    """Server info and stats"""
    return {
        "name": "A2A Communication Server",
        "version": "1.0.0",
        "agents_registered": len(agent_registry),
        "total_tasks": len(task_store),
        "endpoints": {
            "register_agent": "POST /agents/register",
            "list_agents": "GET /agents",
            "send_task": "POST /tasks/send",
            "get_task": "GET /tasks/{task_id}",
        }
    }


# ============================================================
# AGENT REGISTRY ENDPOINTS
# ============================================================

@app.post("/agents/register")
def register_agent(agent_card: AgentCard):
    """
    Register an agent with the server.
    Agent publishes its AgentCard so others can discover it.
    """
    agent_registry[agent_card.name] = agent_card
    agent_inboxes[agent_card.name] = []
    
    print(f"[A2A Server] Agent registered: {agent_card.name}")
    print(f"             Skills: {[s.id for s in agent_card.skills]}")
    
    return {
        "status": "registered",
        "agent": agent_card.name,
        "message": f"Agent '{agent_card.name}' registered with {len(agent_card.skills)} skills"
    }


@app.get("/agents")
def list_agents() -> List[AgentCard]:
    """List all registered agents"""
    return list(agent_registry.values())


@app.get("/agents/{agent_name}")
def get_agent(agent_name: str) -> AgentCard:
    """Get agent card by name"""
    if agent_name not in agent_registry:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
    return agent_registry[agent_name]


@app.get("/agents/{agent_name}/.well-known/agent.json")
def get_agent_card_wellknown(agent_name: str) -> AgentCard:
    """
    A2A standard discovery endpoint.
    Agents can be discovered at /.well-known/agent.json
    """
    return get_agent(agent_name)


@app.get("/agents/by-skill/{skill_id}")
def find_agents_by_skill(skill_id: str) -> List[AgentCard]:
    """Find all agents that have a specific skill"""
    return [
        agent for agent in agent_registry.values()
        if agent.has_skill(skill_id)
    ]


# ============================================================
# TASK ENDPOINTS
# ============================================================

@app.post("/tasks/send")
def send_task(receiver: str, request: SendTaskRequest) -> SendTaskResponse:
    """
    Send a task to an agent.
    
    - receiver: Name of the agent to send to
    - request: Contains the message (and optionally task_id to continue)
    """
    # Check agent exists
    if receiver not in agent_registry:
        raise HTTPException(status_code=404, detail=f"Agent '{receiver}' not found")
    
    # Create new task or continue existing
    if request.task_id and request.task_id in task_store:
        task = task_store[request.task_id]
        task.messages.append(request.message)
        task.updated_at = datetime.now()
    else:
        task = Task(
            messages=[request.message]
        )
        task_store[task.id] = task
        agent_inboxes[receiver].append(task.id)
    
    print(f"[A2A Server] Task {task.id} sent to {receiver}")
    print(f"             Content: {request.message.content}")
    
    return SendTaskResponse(task=task)


@app.get("/tasks/{task_id}")
def get_task(task_id: str) -> Task:
    """Get task by ID"""
    if task_id not in task_store:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    return task_store[task_id]


@app.post("/tasks/{task_id}/cancel")
def cancel_task(task_id: str, request: CancelTaskRequest = None):
    """Cancel a task"""
    if task_id not in task_store:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    
    task = task_store[task_id]
    task.status = TaskStatus.CANCELLED
    task.updated_at = datetime.now()
    
    reason = request.reason if request else "Cancelled by requester"
    task.add_message("system", {"cancelled": True, "reason": reason})
    
    return {"status": "cancelled", "task_id": task_id}


@app.post("/tasks/{task_id}/update")
def update_task_status(task_id: str, status: TaskStatus, message: Optional[Message] = None):
    """Update task status (called by agent when processing)"""
    if task_id not in task_store:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    
    task = task_store[task_id]
    task.status = status
    task.updated_at = datetime.now()
    
    if message:
        task.messages.append(message)
    
    return {"status": "updated", "task": task}


# ============================================================
# AGENT INBOX ENDPOINTS
# ============================================================

@app.get("/agents/{agent_name}/tasks")
def get_agent_inbox(agent_name: str, status: Optional[TaskStatus] = None) -> List[Task]:
    """
    Get all tasks assigned to an agent (agent's inbox).
    Optionally filter by status.
    """
    if agent_name not in agent_registry:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
    
    task_ids = agent_inboxes.get(agent_name, [])
    tasks = [task_store[tid] for tid in task_ids if tid in task_store]
    
    if status:
        tasks = [t for t in tasks if t.status == status]
    
    return tasks


@app.get("/agents/{agent_name}/tasks/pending")
def get_pending_tasks(agent_name: str) -> List[Task]:
    """Get only pending tasks for an agent"""
    return get_agent_inbox(agent_name, status=TaskStatus.PENDING)


# ============================================================
# RUN SERVER
# ============================================================

def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the A2A server"""
    print(f"\n{'='*60}")
    print("  A2A COMMUNICATION SERVER")
    print(f"  Running on http://{host}:{port}")
    print(f"{'='*60}\n")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
