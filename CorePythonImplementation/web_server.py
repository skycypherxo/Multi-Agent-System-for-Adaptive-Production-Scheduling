from flask import Flask, render_template_string, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
import json
import time
import threading
import random
from datetime import datetime

# Import our agents using the import helper
from agent_imports import (
    EventGenerator, JobAgent, SchedulerAgent, 
    MachineAgent, ProductionLineAgent, MaintenanceAlertAgent
)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'production_system_secret'
socketio = SocketIO(app, cors_allowed_origins="*")

class ProductionSystemServer:
    def __init__(self):
        self.running = False
        self.start_time = time.time()
        
        # Initialize agents
        self.event_generator = EventGenerator()
        self.scheduler = SchedulerAgent("MainScheduler")
        self.maintenance_agent = MaintenanceAlertAgent("MaintenanceAgent")
        
        # Create machines
        self.machines = {}
        for machine_id in ['Machine-A', 'Machine-B', 'Machine-C']:
            machine = MachineAgent(machine_id, self.scheduler, self.maintenance_agent)
            self.machines[machine_id] = machine
            self.scheduler.register_machine(machine_id, machine)
        
        # Create production line
        machine_list = list(self.machines.values())
        self.production_line = ProductionLineAgent("ProductionLine-1", machine_list, self.scheduler)
        self.scheduler.register_production_line("ProductionLine-1", self.production_line)
        
        # Create job agent
        self.job_agent = JobAgent("JobGenerator", self.scheduler, self.event_generator)
        
        # Register maintenance agent with scheduler
        self.scheduler.register_maintenance_agent(self.maintenance_agent)
        
        # System statistics
        self.stats = {
            'total_jobs': 0,
            'completed_jobs': 0,
            'rush_jobs': 0,
            'active_machines': 0,
            'system_efficiency': 100,
            'job_queue': [],
            'messages': []
        }
        
    def add_message(self, agent_name, message):
        """Add a message to the log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.stats['messages'].append({
            'timestamp': timestamp,
            'agent': agent_name,
            'message': message
        })
        
        # Keep only last 100 messages
        if len(self.stats['messages']) > 100:
            self.stats['messages'] = self.stats['messages'][-100:]
        
        # Emit to all connected clients
        socketio.emit('new_message', {
            'timestamp': timestamp,
            'agent': agent_name,
            'message': message
        })
    
    def update_stats(self):
        """Update system statistics"""
        # Update job queue
        self.stats['job_queue'] = []
        for priority_val, neg_due_date, job in self.scheduler.job_queue:
            self.stats['job_queue'].append({
                'id': job['job_id'],
                'priority': job['priority'],
                'due_date': job['due_date']
            })
        
        # Update machine stats
        active_machines = 0
        machine_stats = {}
        
        for machine_id, machine in self.machines.items():
            status = machine.status.value
            if status == 'busy':
                active_machines += 1
            
            machine_stats[machine_id] = {
                'status': status,
                'jobs_completed': machine.jobs_completed,
                'efficiency': int(machine.reliability * 100),
                'current_job': machine.current_job['job_id'] if machine.current_job else None,
                'breakdown_count': machine.breakdown_count
            }
        
        self.stats['active_machines'] = active_machines
        self.stats['completed_jobs'] = sum(m.jobs_completed for m in self.machines.values())
        self.stats['system_efficiency'] = int(
            (self.stats['completed_jobs'] / max(1, self.stats['total_jobs'])) * 100
        )
        
        # Get production line stats
        production_stats = {
            'throughput': self.production_line.total_throughput,
            'bottlenecks': len(self.production_line.bottlenecks),
            'line_efficiency': int(self.production_line.line_efficiency * 100)
        }
        
        # Get maintenance stats
        maintenance_stats = {
            'repairs_completed': len(self.maintenance_agent.maintenance_history),
            'alerts_sent': len(self.maintenance_agent.alert_history),
            'scheduled_maintenance': len(self.maintenance_agent.scheduled_maintenance)
        }
        
        # Emit updated stats to all clients
        socketio.emit('stats_update', {
            'stats': self.stats,
            'machines': machine_stats,
            'production': production_stats,
            'maintenance': maintenance_stats,
            'uptime': int(time.time() - self.start_time)
        })
    
    def simulation_step(self):
        """Run one simulation step"""
        try:
            # Generate events randomly
            if random.random() < 0.3:
                self.event_generator.generate_event()
                self.stats['total_jobs'] += 1
                if self.event_generator.event_queue and self.event_generator.event_queue[-1]['priority'] == 'high':
                    self.stats['rush_jobs'] += 1
            
            # Step all agents
            self.job_agent.step()
            self.scheduler.step()
            
            for machine in self.machines.values():
                machine.step()
            
            self.production_line.step()
            self.maintenance_agent.step()
            
            # Update statistics
            self.update_stats()
            
            # Random events
            if random.random() < 0.02:  # 2% chance
                self.trigger_random_event()
                
        except Exception as e:
            self.add_message("System", f"Error in simulation step: {str(e)}")
    
    def trigger_random_event(self):
        """Trigger random system events"""
        event_type = random.choice(['breakdown', 'maintenance', 'rush_job', 'efficiency_boost'])
        
        if event_type == 'breakdown':
            machine_id = random.choice(list(self.machines.keys()))
            machine = self.machines[machine_id]
            if machine.status.value in ['busy', 'idle']:
                machine._trigger_breakdown()
                self.add_message("System", f"Random breakdown occurred at {machine_id}")
        
        elif event_type == 'maintenance':
            machine_id = random.choice(list(self.machines.keys()))
            machine = self.machines[machine_id]
            if machine.status.value == 'idle':
                machine._start_maintenance()
                self.add_message("Maintenance", f"Preventive maintenance started for {machine_id}")
        
        elif event_type == 'rush_job':
            rush_job = self.job_agent.generate_job("rush")
            self.job_agent.sendMessage(rush_job)
            self.add_message("JobAgent", f"Emergency rush job {rush_job['job_id']} generated!")
        
        elif event_type == 'efficiency_boost':
            machine_id = random.choice(list(self.machines.keys()))
            machine = self.machines[machine_id]
            machine.processing_speed *= 1.1
            self.add_message("System", f"Efficiency boost applied to {machine_id}")

    def start_simulation(self):
        """Start the simulation"""
        self.running = True
        self.add_message("System", "Production simulation started")
        
        def simulation_loop():
            while self.running:
                self.simulation_step()
                time.sleep(2)  # 2 second intervals
        
        simulation_thread = threading.Thread(target=simulation_loop)
        simulation_thread.daemon = True
        simulation_thread.start()
    
    def stop_simulation(self):
        """Stop the simulation"""
        self.running = False
        self.add_message("System", "Production simulation stopped")
    
    def reset_system(self):
        """Reset the entire system"""
        self.stop_simulation()
        
        # Reset all agents
        self.__init__()
        
        self.add_message("System", "System reset completed")

# Global system instance
production_system = ProductionSystemServer()

@app.route('/')
def dashboard():
    """Serve the main dashboard"""
    return send_from_directory('.', 'interactive_dashboard.html')

@app.route('/api/status')
def get_status():
    """Get current system status"""
    return jsonify({
        'running': production_system.running,
        'stats': production_system.stats,
        'uptime': int(time.time() - production_system.start_time)
    })

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    production_system.add_message("System", "New client connected to dashboard")
    production_system.update_stats()

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

@socketio.on('start_simulation')
def handle_start_simulation():
    """Handle start simulation request"""
    production_system.start_simulation()

@socketio.on('stop_simulation')
def handle_stop_simulation():
    """Handle stop simulation request"""
    production_system.stop_simulation()

@socketio.on('reset_system')
def handle_reset_system():
    """Handle system reset request"""
    production_system.reset_system()

@socketio.on('generate_rush_job')
def handle_generate_rush_job():
    """Handle generate rush job request"""
    rush_job = production_system.job_agent.generate_job("rush")
    production_system.job_agent.sendMessage(rush_job)
    production_system.stats['total_jobs'] += 1
    production_system.stats['rush_jobs'] += 1
    production_system.add_message("JobAgent", f"Manual rush job {rush_job['job_id']} generated")

@socketio.on('trigger_maintenance')
def handle_trigger_maintenance():
    """Handle trigger maintenance request"""
    machine_id = random.choice(list(production_system.machines.keys()))
    machine = production_system.machines[machine_id]
    if machine.status.value in ['idle']:
        machine._start_maintenance()
        production_system.add_message("Maintenance", f"Manual maintenance triggered for {machine_id}")
    else:
        production_system.add_message("Maintenance", f"Cannot start maintenance - {machine_id} is {machine.status.value}")

@socketio.on('simulate_breakdown')
def handle_simulate_breakdown():
    """Handle simulate breakdown request"""
    machine_id = random.choice(list(production_system.machines.keys()))
    machine = production_system.machines[machine_id]
    machine._trigger_breakdown()
    production_system.add_message("System", f"Manual breakdown simulated for {machine_id}")

if __name__ == '__main__':
    print("🏭 Production System Dashboard Server")
    print("📊 Dashboard URL: http://localhost:5000")
    print("🔧 Starting server...")
    
    # Add some initial messages
    production_system.add_message("System", "Production system initialized")
    production_system.add_message("System", "All agents are ready")
    production_system.add_message("System", "Open http://localhost:5000 to view dashboard")
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
