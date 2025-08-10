import heapq
import time

class SchedulerAgent:
    """
    Scheduler Agent - Core decision-maker
    - Receives job arrivals, machine status updates, production line alerts
    - Assigns jobs to machines
    - Adjusts schedule dynamically based on real-time info
    - Interacts with generative AI + RAG for adaptive scheduling logic
    """
    
    def __init__(self, name):
        self.name = name
        self.inbox = []  # Messages from Job, Machine, ProductionLine, etc.
        self.job_queue = []  # Priority queue: list of (priority_value, job_dict)
        self.priority_map = {'high': 0, 'normal': 1, 'low': 2}
        self.machines = {}  # machine_id: {'status': status, 'current_job': job_id, 'agent': machine_agent}
        self.completed_jobs = []
        self.production_lines = {}  # line_id: production_line_agent
        self.maintenance_agent = None
        
    def register_machine(self, machine_id, machine_agent=None):
        """Register a machine with the scheduler"""
        self.machines[machine_id] = {
            'status': 'idle',
            'current_job': None,
            'agent': machine_agent,
            'last_update': time.time()
        }

    def register_production_line(self, line_id, production_line_agent):
        """Register a production line with the scheduler"""
        self.production_lines[line_id] = production_line_agent
        
    def register_maintenance_agent(self, maintenance_agent):
        """Register the maintenance agent"""
        self.maintenance_agent = maintenance_agent

    def process_messages(self):
        """Process all incoming messages from various agents"""
        for sender, msg in self.inbox:
            msg_type = msg.get('type', 'unknown')
            
            if msg_type == 'new_job':
                self._handle_new_job(msg['job'])
                
            elif msg_type == 'job_done':
                self._handle_job_completion(sender, msg)
                
            elif msg_type == 'status_update':
                self._handle_status_update(sender, msg)
                
            elif msg_type == 'production_alert':
                self._handle_production_alert(sender, msg)
                
            elif msg_type == 'maintenance_alert':
                self._handle_maintenance_alert(sender, msg)
                
            elif msg_type == 'rescheduling_request':
                self._handle_rescheduling_request(sender, msg)

        self.inbox.clear()

    def _handle_new_job(self, job):
        """Handle new job arrival"""
        priority_val = self.priority_map.get(job['priority'], 1)
        # Use negative due_date for earliest due date first (within same priority)
        heapq.heappush(self.job_queue, (priority_val, -job['due_date'], job))
        print(f"[{self.name}] Added job {job['job_id']} with priority {job['priority']} (due: {job['due_date']}) to queue")

    def _handle_job_completion(self, machine_id, msg):
        """Handle job completion from machine"""
        if machine_id in self.machines:
            self.machines[machine_id]['status'] = 'idle'
            self.machines[machine_id]['current_job'] = None
            self.completed_jobs.append({
                'job_id': msg['job_id'],
                'machine_id': machine_id,
                'completion_time': time.time()
            })
            print(f"[{self.name}] Machine {machine_id} completed job {msg['job_id']}")

    def _handle_status_update(self, machine_id, msg):
        """Handle machine status updates"""
        if machine_id in self.machines:
            old_status = self.machines[machine_id]['status']
            new_status = msg['status']
            self.machines[machine_id]['status'] = new_status
            self.machines[machine_id]['last_update'] = time.time()
            
            print(f"[{self.name}] Machine {machine_id} status: {old_status} -> {new_status}")
            
            # If machine broke down, need to reschedule its job
            if new_status in ['broken', 'maintenance'] and self.machines[machine_id]['current_job']:
                job_id = self.machines[machine_id]['current_job']
                self._reschedule_job(job_id, f"Machine {machine_id} {new_status}")

    def _handle_production_alert(self, sender, msg):
        """Handle alerts from production lines"""
        alert_type = msg.get('alert_type', 'unknown')
        print(f"[{self.name}] Production alert from {sender}: {alert_type}")
        
        if alert_type == 'bottleneck':
            self._handle_bottleneck_alert(sender, msg)
        elif alert_type == 'line_down':
            self._handle_line_down_alert(sender, msg)

    def _handle_maintenance_alert(self, sender, msg):
        """Handle maintenance alerts"""
        alert_type = msg.get('alert_type', 'unknown')
        print(f"[{self.name}] Maintenance alert: {alert_type}")
        
        if alert_type == 'scheduled_maintenance':
            affected_machines = msg.get('machines', [])
            for machine_id in affected_machines:
                if machine_id in self.machines:
                    self._reschedule_job(self.machines[machine_id]['current_job'], 
                                       f"Scheduled maintenance on {machine_id}")

    def _handle_rescheduling_request(self, sender, msg):
        """Handle rescheduling requests"""
        reason = msg.get('reason', 'Unknown')
        print(f"[{self.name}] Rescheduling request from {sender}: {reason}")
        self._dynamic_reschedule()

    def _reschedule_job(self, job_id, reason):
        """Reschedule a job due to machine issues"""
        if job_id:
            print(f"[{self.name}] Rescheduling job {job_id} due to: {reason}")
            # In a real implementation, you'd put the job back in the queue
            # For now, just log the event

    def _dynamic_reschedule(self):
        """Perform dynamic rescheduling based on current conditions"""
        print(f"[{self.name}] Performing dynamic rescheduling...")
        # AI/RAG logic would go here for adaptive scheduling
        self._optimize_schedule()

    def _optimize_schedule(self):
        """Optimize the current schedule using AI/ML techniques"""
        # Placeholder for AI-powered scheduling optimization
        # This could integrate with generative AI and RAG for intelligent scheduling
        available_machines = [mid for mid, info in self.machines.items() 
                            if info['status'] == 'idle']
        queue_size = len(self.job_queue)
        
        print(f"[{self.name}] Schedule optimization: {len(available_machines)} idle machines, {queue_size} jobs queued")

    def _handle_bottleneck_alert(self, sender, msg):
        """Handle bottleneck alerts from production lines"""
        bottleneck_machine = msg.get('bottleneck_machine')
        print(f"[{self.name}] Bottleneck detected at {bottleneck_machine} in line {sender}")
        # Could implement load balancing logic here

    def _handle_line_down_alert(self, sender, msg):
        """Handle production line down alerts"""
        affected_machines = msg.get('affected_machines', [])
        print(f"[{self.name}] Production line {sender} down, affecting machines: {affected_machines}")

    def assign_jobs(self):
        """Assign jobs to available machines"""
        for machine_id, machine_info in self.machines.items():
            if machine_info['status'] == 'idle' and self.job_queue:
                # Get highest priority job
                priority_val, neg_due_date, job = heapq.heappop(self.job_queue)
                
                # Update machine status
                self.machines[machine_id]['status'] = 'busy'
                self.machines[machine_id]['current_job'] = job['job_id']
                
                # Send assignment message to machine agent
                if machine_info['agent']:
                    assignment_msg = {
                        "type": "assign_job",
                        "job": job,
                        "assigned_by": self.name
                    }
                    machine_info['agent'].inbox.append((self.name, assignment_msg))
                
                print(f"[{self.name}] Assigned job {job['job_id']} (priority: {job['priority']}) to {machine_id}")

    def get_system_status(self):
        """Get current system status"""
        status = {
            'jobs_in_queue': len(self.job_queue),
            'completed_jobs': len(self.completed_jobs),
            'machine_status': {mid: info['status'] for mid, info in self.machines.items()},
            'busy_machines': len([m for m in self.machines.values() if m['status'] == 'busy']),
            'idle_machines': len([m for m in self.machines.values() if m['status'] == 'idle']),
            'broken_machines': len([m for m in self.machines.values() if m['status'] == 'broken'])
        }
        return status

    def step(self):
        """Main step function"""
        self.process_messages()
        self.assign_jobs()
        
        # Periodic optimization
        if time.time() % 10 < 1:  # Every ~10 seconds
            self._optimize_schedule()
        self.assign_jobs()
