import random
import time

class JobAgent:
    """
    Job Agent (Job Generator)
    - Generates new jobs (normal and rush orders)
    - Sends job creation messages to Scheduler
    - Can simulate job priority, due dates, etc.
    """
    
    def __init__(self, name, scheduler, event_generator):
        self.name = name
        self.scheduler = scheduler
        self.event_generator = event_generator
        self.current_event = None
        self.job_counter = 0
        
    def generate_job(self, job_type="normal"):
        """Generate a new job with specified type (normal or rush)"""
        self.job_counter += 1
        job_id = f"J{self.job_counter:04d}"
        
        if job_type == "rush":
            priority = "high"
            due_date = random.randint(5, 15)  # Shorter due dates for rush orders
            processing_time = random.randint(2, 4)
        else:
            priority = "normal"
            due_date = random.randint(20, 40)
            processing_time = random.randint(3, 8)
            
        job = {
            "job_id": job_id,
            "priority": priority,
            "due_date": due_date,
            "processing_time": processing_time,
            "arrival_time": time.time(),
            "job_type": job_type
        }
        
        return job

    def getData(self):
        """Pull one event from event generator"""
        self.current_event = self.event_generator.get_event()

    def analyseJob(self):
        """Convert event data into job dict"""
        if not self.current_event:
            return None
            
        job = {
            "job_id": self.current_event["job_id"],
            "priority": self.current_event["priority"],
            "due_date": self.current_event["due_date"],
            "processing_time": self.current_event["processing_time"],
            "arrival_time": time.time(),
            "job_type": "rush" if self.current_event["priority"] == "high" else "normal"
        }
        return job

    def sendMessage(self, job):
        """Send job creation message to Scheduler"""
        if job:
            message = {
                "type": "new_job",
                "job": job,
                "sender": self.name
            }
            self.scheduler.inbox.append((self.name, message))
            print(f"[{self.name}] Generated and sent {job['job_type']} job {job['job_id']} with priority {job['priority']}")

    def simulate_job_generation(self, rush_probability=0.2):
        """Simulate generating jobs with a chance for rush orders"""
        if random.random() < rush_probability:
            job = self.generate_job("rush")
        else:
            job = self.generate_job("normal")
        self.sendMessage(job)
        return job

    def step(self):
        """Main step function - process events from event generator"""
        self.getData()
        job = self.analyseJob()
        if job:
            self.sendMessage(job)
