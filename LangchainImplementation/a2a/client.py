"""
A2A Client - Helper class for agents to communicate with the A2A server

Usage:
    client = A2AClient("http://localhost:8000")
    
    # Register agent
    client.register(my_agent_card)
    
    # Discover agents
    agents = client.discover_agents()
    cutters = client.find_by_skill("cutting")
    
    # Send task to another agent
    task = client.send_task("Scheduler", message_content)
    
    # Check inbox for tasks assigned to me
    my_tasks = client.get_my_tasks("MachineAgent-01")
"""

import httpx
from typing import List, Optional, Any
from .models import AgentCard, Task, TaskStatus, Message, SendTaskRequest


class A2AClient:
    """Client for communicating with A2A server"""
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url.rstrip("/")
        self._client = httpx.Client(timeout=30.0)
    
    def _url(self, path: str) -> str:
        return f"{self.server_url}{path}"
    
    # ============================================================
    # AGENT REGISTRATION & DISCOVERY
    # ============================================================
    
    def register(self, agent_card: AgentCard) -> dict:
        """Register an agent with the server"""
        response = self._client.post(
            self._url("/agents/register"),
            json=agent_card.model_dump(mode='json')
        )
        response.raise_for_status()
        return response.json()
    
    def discover_agents(self) -> List[AgentCard]:
        """Get all registered agents"""
        response = self._client.get(self._url("/agents"))
        response.raise_for_status()
        return [AgentCard(**a) for a in response.json()]
    
    def get_agent(self, agent_name: str) -> AgentCard:
        """Get a specific agent's card"""
        response = self._client.get(self._url(f"/agents/{agent_name}"))
        response.raise_for_status()
        return AgentCard(**response.json())
    
    def find_by_skill(self, skill_id: str) -> List[AgentCard]:
        """Find agents with a specific skill"""
        response = self._client.get(self._url(f"/agents/by-skill/{skill_id}"))
        response.raise_for_status()
        return [AgentCard(**a) for a in response.json()]
    
    # ============================================================
    # TASK OPERATIONS
    # ============================================================
    
    def send_task(
        self, 
        receiver: str, 
        content: Any, 
        sender: str = "user",
        task_id: Optional[str] = None
    ) -> Task:
        """
        Send a task to an agent.
        
        Args:
            receiver: Name of agent to send to
            content: The task content/payload
            sender: Who is sending (for message role)
            task_id: Optional - continue existing task
        """
        message = Message(role=sender, content=content)
        request = SendTaskRequest(task_id=task_id, message=message)
        
        response = self._client.post(
            self._url(f"/tasks/send?receiver={receiver}"),
            json=request.model_dump(mode='json')
        )
        response.raise_for_status()
        return Task(**response.json()["task"])
    
    def get_task(self, task_id: str) -> Task:
        """Get task by ID"""
        response = self._client.get(self._url(f"/tasks/{task_id}"))
        response.raise_for_status()
        return Task(**response.json())
    
    def cancel_task(self, task_id: str, reason: str = None) -> dict:
        """Cancel a task"""
        response = self._client.post(
            self._url(f"/tasks/{task_id}/cancel"),
            json={"task_id": task_id, "reason": reason}
        )
        response.raise_for_status()
        return response.json()
    
    def update_task(self, task_id: str, status: TaskStatus, message_content: Any = None) -> dict:
        """Update task status"""
        params = {"status": status.value}
        json_body = None
        
        if message_content:
            json_body = {"role": "agent", "content": message_content}
        
        response = self._client.post(
            self._url(f"/tasks/{task_id}/update"),
            params=params,
            json=json_body
        )
        response.raise_for_status()
        return response.json()
    
    # ============================================================
    # INBOX OPERATIONS
    # ============================================================
    
    def get_my_tasks(self, agent_name: str, status: Optional[TaskStatus] = None) -> List[Task]:
        """Get tasks assigned to an agent"""
        url = self._url(f"/agents/{agent_name}/tasks")
        if status:
            url += f"?status={status.value}"
        
        response = self._client.get(url)
        response.raise_for_status()
        return [Task(**t) for t in response.json()]
    
    def get_pending_tasks(self, agent_name: str) -> List[Task]:
        """Get only pending tasks for an agent"""
        response = self._client.get(self._url(f"/agents/{agent_name}/tasks/pending"))
        response.raise_for_status()
        return [Task(**t) for t in response.json()]
    
    # ============================================================
    # UTILITIES
    # ============================================================
    
    def health_check(self) -> dict:
        """Check if server is running"""
        response = self._client.get(self._url("/"))
        response.raise_for_status()
        return response.json()
    
    def close(self):
        """Close the HTTP client"""
        self._client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
