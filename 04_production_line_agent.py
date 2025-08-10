import time
import statistics

class ProductionLineAgent:
    """
    ProductionLine Agent
    - Manages a production line consisting of multiple machines
    - Monitors line progress and bottlenecks
    - Sends alerts to Scheduler for rescheduling needs
    - Coordinates jobs across machines in the line
    """
    
    def __init__(self, line_id, machine_agents, scheduler_agent=None):
        self.line_id = line_id
        self.machine_agents = machine_agents  # List of MachineAgent objects
        self.scheduler_agent = scheduler_agent
        self.inbox = []
        
        # Line configuration
        self.machines = {machine.machine_id: machine for machine in machine_agents}
        self.machine_sequence = [machine.machine_id for machine in machine_agents]  # Production sequence
        
        # Line monitoring
        self.line_jobs = {}  # job_id: {current_stage, start_time, progress_stages}
        self.bottlenecks = []
        self.line_efficiency = 1.0
        self.throughput_history = []
        
        # Statistics
        self.jobs_completed = 0
        self.total_line_time = 0
        self.bottleneck_incidents = 0
        
        print(f"[{self.line_id}] Production line initialized with machines: {self.machine_sequence}")

    def process_messages(self):
        """Process incoming messages"""
        for sender, msg in self.inbox:
            msg_type = msg.get('type', 'unknown')
            
            if msg_type == 'job_started':
                self._handle_job_started(msg)
            elif msg_type == 'job_completed':
                self._handle_job_completed(msg)
            elif msg_type == 'machine_status_change':
                self._handle_machine_status_change(sender, msg)
            elif msg_type == 'throughput_request':
                self._send_throughput_report()

        self.inbox.clear()

    def _handle_job_started(self, msg):
        """Handle job started on the line"""
        job_id = msg['job_id']
        machine_id = msg['machine_id']
        
        if job_id not in self.line_jobs:
            self.line_jobs[job_id] = {
                'current_stage': 0,
                'start_time': time.time(),
                'progress_stages': [machine_id],
                'stage_times': [time.time()]
            }
        
        print(f"[{self.line_id}] Job {job_id} started on machine {machine_id}")

    def _handle_job_completed(self, msg):
        """Handle job completion on a machine in the line"""
        job_id = msg['job_id']
        machine_id = msg['machine_id']
        
        if job_id in self.line_jobs:
            job_info = self.line_jobs[job_id]
            job_info['progress_stages'].append(f"{machine_id}_completed")
            job_info['stage_times'].append(time.time())
            
            # Check if this was the last machine in the sequence
            current_machine_index = self.machine_sequence.index(machine_id)
            if current_machine_index == len(self.machine_sequence) - 1:
                # Job completed entire line
                self._complete_line_job(job_id)
            else:
                # Move to next machine in sequence
                self._advance_job_to_next_stage(job_id, current_machine_index)

    def _complete_line_job(self, job_id):
        """Complete a job that has finished the entire production line"""
        if job_id in self.line_jobs:
            job_info = self.line_jobs[job_id]
            total_time = time.time() - job_info['start_time']
            
            self.jobs_completed += 1
            self.total_line_time += total_time
            self.throughput_history.append(total_time)
            
            # Keep only last 20 throughput records
            if len(self.throughput_history) > 20:
                self.throughput_history.pop(0)
            
            print(f"[{self.line_id}] Job {job_id} completed entire production line in {total_time:.2f} time units")
            
            # Notify scheduler
            if self.scheduler_agent:
                msg = {
                    "type": "line_job_completed",
                    "job_id": job_id,
                    "line_id": self.line_id,
                    "total_time": total_time,
                    "stages": job_info['progress_stages']
                }
                self.scheduler_agent.inbox.append((self.line_id, msg))
            
            del self.line_jobs[job_id]

    def _advance_job_to_next_stage(self, job_id, current_machine_index):
        """Advance job to next machine in the production sequence"""
        if current_machine_index + 1 < len(self.machine_sequence):
            next_machine_id = self.machine_sequence[current_machine_index + 1]
            next_machine = self.machines[next_machine_id]
            
            # Check if next machine is available
            if next_machine.status.value == 'idle':
                # In a real system, you'd coordinate with the scheduler
                # to assign the job to the next machine
                print(f"[{self.line_id}] Job {job_id} ready for next stage: {next_machine_id}")
            else:
                print(f"[{self.line_id}] Job {job_id} waiting for {next_machine_id} (status: {next_machine.status.value})")

    def _handle_machine_status_change(self, machine_id, msg):
        """Handle machine status changes that affect the line"""
        new_status = msg.get('status')
        
        if new_status in ['broken', 'maintenance']:
            self._handle_machine_down(machine_id, new_status)
        elif new_status == 'idle':
            self._check_bottleneck_resolution(machine_id)

    def monitor_line_progress(self):
        """Monitor overall line progress and identify bottlenecks"""
        machine_statuses = {mid: machine.status.value for mid, machine in self.machines.items()}
        busy_machines = [mid for mid, status in machine_statuses.items() if status == 'busy']
        idle_machines = [mid for mid, status in machine_statuses.items() if status == 'idle']
        broken_machines = [mid for mid, status in machine_statuses.items() if status in ['broken', 'maintenance']]
        
        # Detect bottlenecks
        self._detect_bottlenecks(machine_statuses)
        
        # Calculate line efficiency
        self._calculate_line_efficiency(machine_statuses)
        
        # Monitor job flow
        self._monitor_job_flow()
        
        return {
            'busy_machines': len(busy_machines),
            'idle_machines': len(idle_machines),
            'broken_machines': len(broken_machines),
            'active_jobs': len(self.line_jobs),
            'line_efficiency': self.line_efficiency
        }

    def _detect_bottlenecks(self, machine_statuses):
        """Detect bottlenecks in the production line"""
        # Simple bottleneck detection: if a machine is consistently busy while others are idle
        busy_machines = [mid for mid, status in machine_statuses.items() if status == 'busy']
        idle_machines = [mid for mid, status in machine_statuses.items() if status == 'idle']
        
        # Check for queue buildup (simplified)
        for i, machine_id in enumerate(self.machine_sequence[:-1]):
            current_machine = self.machines[machine_id]
            next_machine_id = self.machine_sequence[i + 1]
            next_machine = self.machines[next_machine_id]
            
            # Bottleneck condition: current machine busy, next machine down/busy for extended time
            if (current_machine.status.value == 'busy' and 
                next_machine.status.value in ['broken', 'maintenance', 'busy']):
                
                if machine_id not in self.bottlenecks:
                    self.bottlenecks.append(machine_id)
                    self.bottleneck_incidents += 1
                    self._send_bottleneck_alert(machine_id, next_machine_id)

    def _send_bottleneck_alert(self, bottleneck_machine, blocking_machine):
        """Send bottleneck alert to scheduler"""
        if self.scheduler_agent:
            msg = {
                "type": "production_alert",
                "alert_type": "bottleneck",
                "line_id": self.line_id,
                "bottleneck_machine": bottleneck_machine,
                "blocking_machine": blocking_machine,
                "severity": "medium"
            }
            self.scheduler_agent.inbox.append((self.line_id, msg))
            
        print(f"[{self.line_id}] BOTTLENECK ALERT: {bottleneck_machine} blocked by {blocking_machine}")

    def _check_bottleneck_resolution(self, machine_id):
        """Check if a bottleneck has been resolved"""
        if machine_id in self.bottlenecks:
            self.bottlenecks.remove(machine_id)
            print(f"[{self.line_id}] Bottleneck resolved at {machine_id}")

    def _handle_machine_down(self, machine_id, reason):
        """Handle machine going down"""
        # Notify scheduler about line disruption
        if self.scheduler_agent:
            msg = {
                "type": "production_alert",
                "alert_type": "machine_down",
                "line_id": self.line_id,
                "affected_machine": machine_id,
                "reason": reason,
                "severity": "high"
            }
            self.scheduler_agent.inbox.append((self.line_id, msg))
        
        print(f"[{self.line_id}] Machine {machine_id} down: {reason}")
        
        # Check if entire line needs to stop
        machine_index = self.machine_sequence.index(machine_id)
        if machine_index == 0:  # First machine down affects entire line
            self._send_line_down_alert(machine_id, reason)

    def _send_line_down_alert(self, failed_machine, reason):
        """Send line down alert to scheduler"""
        if self.scheduler_agent:
            msg = {
                "type": "production_alert",
                "alert_type": "line_down",
                "line_id": self.line_id,
                "failed_machine": failed_machine,
                "reason": reason,
                "affected_machines": self.machine_sequence,
                "severity": "critical"
            }
            self.scheduler_agent.inbox.append((self.line_id, msg))
        
        print(f"[{self.line_id}] LINE DOWN: {failed_machine} failure affects entire line")

    def _calculate_line_efficiency(self, machine_statuses):
        """Calculate overall line efficiency"""
        total_machines = len(self.machines)
        operational_machines = sum(1 for status in machine_statuses.values() 
                                 if status in ['idle', 'busy'])
        
        self.line_efficiency = operational_machines / total_machines if total_machines > 0 else 0

    def _monitor_job_flow(self):
        """Monitor job flow through the line"""
        if len(self.line_jobs) > len(self.machine_sequence):
            # Too many jobs in the line - potential congestion
            if self.scheduler_agent:
                msg = {
                    "type": "production_alert",
                    "alert_type": "congestion",
                    "line_id": self.line_id,
                    "active_jobs": len(self.line_jobs),
                    "capacity": len(self.machine_sequence),
                    "severity": "medium"
                }
                self.scheduler_agent.inbox.append((self.line_id, msg))

    def _send_throughput_report(self):
        """Send throughput report to scheduler"""
        if self.scheduler_agent and self.throughput_history:
            avg_throughput = statistics.mean(self.throughput_history)
            msg = {
                "type": "throughput_report",
                "line_id": self.line_id,
                "average_throughput": avg_throughput,
                "jobs_completed": self.jobs_completed,
                "line_efficiency": self.line_efficiency,
                "bottleneck_incidents": self.bottleneck_incidents
            }
            self.scheduler_agent.inbox.append((self.line_id, msg))

    def coordinate_jobs(self):
        """Coordinate jobs across machines in the line"""
        # Check for jobs waiting to advance to next stage
        for job_id, job_info in self.line_jobs.items():
            current_stage = job_info.get('current_stage', 0)
            if current_stage < len(self.machine_sequence) - 1:
                next_machine_id = self.machine_sequence[current_stage + 1]
                next_machine = self.machines[next_machine_id]
                
                # If next machine is idle, could coordinate job transfer
                if next_machine.status.value == 'idle':
                    # In practice, this would involve scheduler coordination
                    pass

    def get_line_metrics(self):
        """Get production line metrics"""
        avg_line_time = (self.total_line_time / self.jobs_completed 
                        if self.jobs_completed > 0 else 0)
        
        return {
            "line_id": self.line_id,
            "jobs_completed": self.jobs_completed,
            "average_line_time": avg_line_time,
            "line_efficiency": self.line_efficiency,
            "bottleneck_incidents": self.bottleneck_incidents,
            "active_jobs": len(self.line_jobs),
            "machine_count": len(self.machines),
            "throughput_history": self.throughput_history[-5:] if self.throughput_history else []
        }

    def send_rescheduling_request(self, reason):
        """Send rescheduling request to scheduler"""
        if self.scheduler_agent:
            msg = {
                "type": "rescheduling_request",
                "reason": reason,
                "line_id": self.line_id,
                "priority": "high"
            }
            self.scheduler_agent.inbox.append((self.line_id, msg))
            
        print(f"[{self.line_id}] Rescheduling request sent: {reason}")

    def step(self):
        """Main step function"""
        self.process_messages()
        
        # Monitor line progress
        line_status = self.monitor_line_progress()
        
        # Coordinate jobs
        self.coordinate_jobs()
        
        # Periodic reporting
        if self.jobs_completed > 0 and self.jobs_completed % 5 == 0:
            self._send_throughput_report()
        
        return line_status
