import random
import time
from enum import Enum

class MachineStatus(Enum):
    IDLE = "idle"
    BUSY = "busy"
    BROKEN = "broken"
    MAINTENANCE = "maintenance"
    OPERATIONAL = "operational"

class MachineAgent:
    """
    Machine Agent
    - Represents an individual machine (A, B, C)
    - Tracks current job, status (operational, broken, maintenance)
    - Sends status updates to Scheduler and Maintenance Agent
    - Executes assigned jobs
    """
    
    def __init__(self, machine_id, scheduler_agent=None, maintenance_agent=None):
        self.machine_id = machine_id
        self.status = MachineStatus.IDLE
        self.current_job = None
        self.job_start_time = None
        self.job_progress = 0.0
        self.inbox = []
        
        # Agent references
        self.scheduler_agent = scheduler_agent
        self.maintenance_agent = maintenance_agent
        
        # Machine characteristics
        self.processing_speed = random.uniform(0.8, 1.2)  # Speed multiplier
        self.reliability = random.uniform(0.85, 0.98)  # Probability of not breaking down
        self.maintenance_due_hours = random.randint(100, 200)  # Hours until maintenance needed
        self.current_operating_hours = 0
        
        # Statistics
        self.jobs_completed = 0
        self.total_processing_time = 0
        self.breakdown_count = 0
        self.last_maintenance = time.time()
        
        print(f"[{self.machine_id}] Machine initialized - Speed: {self.processing_speed:.2f}, Reliability: {self.reliability:.2f}")

    def process_messages(self):
        """Process incoming messages"""
        for sender, msg in self.inbox:
            msg_type = msg.get('type', 'unknown')
            
            if msg_type == 'assign_job':
                self._handle_job_assignment(msg['job'])
                
            elif msg_type == 'maintenance_schedule':
                self._handle_maintenance_schedule(msg)
                
            elif msg_type == 'emergency_stop':
                self._handle_emergency_stop(msg)
                
            elif msg_type == 'status_request':
                self._send_status_update()

        self.inbox.clear()

    def _handle_job_assignment(self, job):
        """Handle job assignment from scheduler"""
        if self.status == MachineStatus.IDLE:
            self.current_job = job
            self.job_start_time = time.time()
            self.job_progress = 0.0
            self.status = MachineStatus.BUSY
            
            print(f"[{self.machine_id}] Started job {job['job_id']} (estimated time: {job['processing_time']})")
            self._send_status_update()
        else:
            print(f"[{self.machine_id}] Cannot accept job {job['job_id']} - currently {self.status.value}")

    def _handle_maintenance_schedule(self, msg):
        """Handle scheduled maintenance"""
        maintenance_time = msg.get('scheduled_time', time.time())
        if time.time() >= maintenance_time:
            self._start_maintenance()

    def _handle_emergency_stop(self, msg):
        """Handle emergency stop command"""
        reason = msg.get('reason', 'Emergency stop')
        print(f"[{self.machine_id}] Emergency stop: {reason}")
        self._stop_current_job()
        self.status = MachineStatus.BROKEN
        self._send_status_update()

    def _start_maintenance(self):
        """Start maintenance process"""
        if self.current_job:
            self._stop_current_job()
            
        self.status = MachineStatus.MAINTENANCE
        self.last_maintenance = time.time()
        self.current_operating_hours = 0  # Reset after maintenance
        
        print(f"[{self.machine_id}] Starting maintenance")
        self._send_status_update()
        self._notify_maintenance_agent("maintenance_started")

    def _stop_current_job(self):
        """Stop current job (due to breakdown or maintenance)"""
        if self.current_job:
            job_id = self.current_job['job_id']
            print(f"[{self.machine_id}] Stopping job {job_id} due to {self.status.value}")
            
            # Notify scheduler about interrupted job
            if self.scheduler_agent:
                msg = {
                    "type": "job_interrupted",
                    "job_id": job_id,
                    "reason": self.status.value,
                    "progress": self.job_progress
                }
                self.scheduler_agent.inbox.append((self.machine_id, msg))
            
            self.current_job = None
            self.job_start_time = None
            self.job_progress = 0.0

    def execute_job(self):
        """Execute the current job"""
        if not self.current_job or self.status != MachineStatus.BUSY:
            return

        # Calculate progress
        elapsed_time = time.time() - self.job_start_time
        adjusted_processing_time = self.current_job['processing_time'] / self.processing_speed
        self.job_progress = min(elapsed_time / adjusted_processing_time, 1.0)

        # Check if job is completed
        if self.job_progress >= 1.0:
            self._complete_job()
        
        # Update operating hours
        self.current_operating_hours += 0.1  # Assume each step is 0.1 hours

    def _complete_job(self):
        """Complete the current job"""
        if self.current_job:
            job_id = self.current_job['job_id']
            completion_time = time.time() - self.job_start_time
            
            # Update statistics
            self.jobs_completed += 1
            self.total_processing_time += completion_time
            
            # Notify scheduler
            if self.scheduler_agent:
                msg = {
                    "type": "job_done",
                    "job_id": job_id,
                    "completion_time": completion_time,
                    "machine_efficiency": self.processing_speed
                }
                self.scheduler_agent.inbox.append((self.machine_id, msg))
            
            print(f"[{self.machine_id}] Completed job {job_id} in {completion_time:.2f} time units")
            
            # Reset job state
            self.current_job = None
            self.job_start_time = None
            self.job_progress = 0.0
            self.status = MachineStatus.IDLE
            
            self._send_status_update()

    def check_breakdown(self):
        """Check if machine breaks down"""
        if self.status == MachineStatus.BUSY and random.random() > self.reliability:
            self._trigger_breakdown()

    def _trigger_breakdown(self):
        """Trigger a machine breakdown"""
        self.breakdown_count += 1
        self.status = MachineStatus.BROKEN
        
        if self.current_job:
            self._stop_current_job()
        
        print(f"[{self.machine_id}] BREAKDOWN! Total breakdowns: {self.breakdown_count}")
        self._send_status_update()
        self._notify_maintenance_agent("breakdown")

    def check_maintenance_due(self):
        """Check if maintenance is due"""
        if (self.current_operating_hours >= self.maintenance_due_hours and 
            self.status in [MachineStatus.IDLE]):
            self._notify_maintenance_agent("maintenance_due")

    def _send_status_update(self):
        """Send status update to scheduler and maintenance agent"""
        status_msg = {
            "type": "status_update",
            "status": self.status.value,
            "current_job": self.current_job['job_id'] if self.current_job else None,
            "job_progress": self.job_progress,
            "operating_hours": self.current_operating_hours,
            "jobs_completed": self.jobs_completed,
            "breakdown_count": self.breakdown_count
        }
        
        # Send to scheduler
        if self.scheduler_agent:
            self.scheduler_agent.inbox.append((self.machine_id, status_msg))
        
        # Send to maintenance agent
        if self.maintenance_agent:
            self.maintenance_agent.inbox.append((self.machine_id, status_msg))

    def _notify_maintenance_agent(self, alert_type):
        """Send specific alerts to maintenance agent"""
        if self.maintenance_agent:
            msg = {
                "type": "maintenance_alert",
                "alert_type": alert_type,
                "machine_id": self.machine_id,
                "operating_hours": self.current_operating_hours,
                "last_maintenance": self.last_maintenance,
                "breakdown_count": self.breakdown_count
            }
            self.maintenance_agent.inbox.append((self.machine_id, msg))

    def perform_repair(self):
        """Perform repair after breakdown"""
        if self.status == MachineStatus.BROKEN:
            repair_time = random.uniform(2, 8)  # Random repair time
            print(f"[{self.machine_id}] Repair completed after {repair_time:.2f} time units")
            
            self.status = MachineStatus.IDLE
            self._send_status_update()
            return True
        return False

    def complete_maintenance(self):
        """Complete scheduled maintenance"""
        if self.status == MachineStatus.MAINTENANCE:
            maintenance_time = random.uniform(3, 6)  # Random maintenance time
            print(f"[{self.machine_id}] Maintenance completed after {maintenance_time:.2f} time units")
            
            # Maintenance improves reliability slightly
            self.reliability = min(self.reliability + 0.02, 0.99)
            self.status = MachineStatus.IDLE
            self._send_status_update()
            return True
        return False

    def get_performance_metrics(self):
        """Get machine performance metrics"""
        avg_processing_time = (self.total_processing_time / self.jobs_completed 
                             if self.jobs_completed > 0 else 0)
        
        return {
            "machine_id": self.machine_id,
            "status": self.status.value,
            "jobs_completed": self.jobs_completed,
            "breakdown_count": self.breakdown_count,
            "operating_hours": self.current_operating_hours,
            "average_processing_time": avg_processing_time,
            "reliability": self.reliability,
            "processing_speed": self.processing_speed
        }

    def step(self):
        """Main step function"""
        self.process_messages()
        
        if self.status == MachineStatus.BUSY:
            self.execute_job()
            self.check_breakdown()
        
        self.check_maintenance_due()
        
        # Periodic status updates
        if random.random() < 0.1:  # 10% chance each step
            self._send_status_update()
