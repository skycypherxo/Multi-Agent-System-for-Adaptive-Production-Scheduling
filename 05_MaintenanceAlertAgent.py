import time
import random
from datetime import datetime, timedelta
from enum import Enum

class MaintenanceType(Enum):
    PREVENTIVE = "preventive"
    CORRECTIVE = "corrective"
    EMERGENCY = "emergency"
    SCHEDULED = "scheduled"

class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class MaintenanceAlertAgent:
    """
    Maintenance + Alert Agent
    - Tracks machine breakdowns and schedules repairs
    - Sends maintenance schedules to Machines and Scheduler
    - Broadcasts alerts about urgent issues (machine failures, safety concerns)
    - Manages preventive maintenance schedules
    """
    
    def __init__(self, name, scheduler_agent=None):
        self.name = name
        self.scheduler_agent = scheduler_agent
        self.inbox = []
        
        # Machine tracking
        self.machine_agents = {}  # machine_id: machine_agent
        self.machine_health = {}  # machine_id: health_metrics
        
        # Maintenance scheduling
        self.maintenance_schedule = {}  # machine_id: [maintenance_records]
        self.repair_queue = []  # List of machines needing repair
        self.preventive_maintenance_intervals = {}  # machine_id: hours_interval
        
        # Alert system
        self.active_alerts = []
        self.alert_history = []
        self.emergency_protocols = {
            'fire': self._fire_emergency_protocol,
            'safety': self._safety_emergency_protocol,
            'critical_breakdown': self._critical_breakdown_protocol
        }
        
        # Statistics
        self.total_repairs = 0
        self.total_maintenance_hours = 0
        self.breakdown_prevented = 0
        self.emergency_incidents = 0
        
        print(f"[{self.name}] Maintenance and Alert Agent initialized")

    def register_machine(self, machine_id, machine_agent):
        """Register a machine for maintenance tracking"""
        self.machine_agents[machine_id] = machine_agent
        self.machine_health[machine_id] = {
            'reliability': machine_agent.reliability,
            'operating_hours': 0,
            'last_maintenance': time.time(),
            'breakdown_count': 0,
            'maintenance_due': False
        }
        
        # Set default preventive maintenance interval
        self.preventive_maintenance_intervals[machine_id] = random.randint(150, 250)
        self.maintenance_schedule[machine_id] = []
        
        print(f"[{self.name}] Registered machine {machine_id} for maintenance tracking")

    def process_messages(self):
        """Process incoming messages from machines and other agents"""
        for sender, msg in self.inbox:
            msg_type = msg.get('type', 'unknown')
            
            if msg_type == 'status_update':
                self._handle_machine_status_update(sender, msg)
            elif msg_type == 'maintenance_alert':
                self._handle_maintenance_alert(sender, msg)
            elif msg_type == 'emergency_alert':
                self._handle_emergency_alert(sender, msg)
            elif msg_type == 'maintenance_request':
                self._handle_maintenance_request(sender, msg)
            elif msg_type == 'safety_concern':
                self._handle_safety_concern(sender, msg)

        self.inbox.clear()

    def _handle_machine_status_update(self, machine_id, msg):
        """Handle machine status updates"""
        if machine_id in self.machine_health:
            health_record = self.machine_health[machine_id]
            
            # Update machine health metrics
            health_record['operating_hours'] = msg.get('operating_hours', health_record['operating_hours'])
            health_record['breakdown_count'] = msg.get('breakdown_count', health_record['breakdown_count'])
            
            # Check if maintenance is due
            self._check_preventive_maintenance_due(machine_id)
            
            # Update reliability based on breakdowns
            if msg.get('breakdown_count', 0) > health_record.get('last_known_breakdowns', 0):
                health_record['last_known_breakdowns'] = msg.get('breakdown_count', 0)
                self._update_machine_reliability(machine_id)

    def _handle_maintenance_alert(self, machine_id, msg):
        """Handle maintenance alerts from machines"""
        alert_type = msg.get('alert_type', 'unknown')
        
        if alert_type == 'breakdown':
            self._handle_breakdown(machine_id, msg)
        elif alert_type == 'maintenance_due':
            self._schedule_preventive_maintenance(machine_id)
        elif alert_type == 'maintenance_started':
            self._track_maintenance_start(machine_id)

    def _handle_breakdown(self, machine_id, msg):
        """Handle machine breakdown"""
        breakdown_time = time.time()
        severity = self._assess_breakdown_severity(machine_id, msg)
        
        # Add to repair queue
        repair_record = {
            'machine_id': machine_id,
            'breakdown_time': breakdown_time,
            'severity': severity,
            'estimated_repair_time': self._estimate_repair_time(severity),
            'type': MaintenanceType.CORRECTIVE
        }
        
        self.repair_queue.append(repair_record)
        self.repair_queue.sort(key=lambda x: (x['severity'].value, x['breakdown_time']))
        
        # Create alert
        alert = self._create_alert(
            alert_type='machine_breakdown',
            machine_id=machine_id,
            severity=severity,
            message=f"Machine {machine_id} breakdown detected"
        )
        
        self._broadcast_alert(alert)
        
        # Schedule immediate repair
        self._schedule_repair(repair_record)
        
        print(f"[{self.name}] Breakdown detected on {machine_id} - Severity: {severity.value}")

    def _assess_breakdown_severity(self, machine_id, msg):
        """Assess the severity of a breakdown"""
        operating_hours = msg.get('operating_hours', 0)
        breakdown_count = msg.get('breakdown_count', 1)
        
        # Simple severity assessment
        if breakdown_count > 5 or operating_hours > 500:
            return AlertSeverity.CRITICAL
        elif breakdown_count > 3 or operating_hours > 300:
            return AlertSeverity.HIGH
        elif breakdown_count > 1:
            return AlertSeverity.MEDIUM
        else:
            return AlertSeverity.LOW

    def _estimate_repair_time(self, severity):
        """Estimate repair time based on severity"""
        base_times = {
            AlertSeverity.LOW: (1, 3),
            AlertSeverity.MEDIUM: (2, 6),
            AlertSeverity.HIGH: (4, 10),
            AlertSeverity.CRITICAL: (8, 16)
        }
        
        min_time, max_time = base_times[severity]
        return random.uniform(min_time, max_time)

    def _schedule_repair(self, repair_record):
        """Schedule a repair"""
        machine_id = repair_record['machine_id']
        
        # Notify scheduler about upcoming repair
        if self.scheduler_agent:
            msg = {
                'type': 'maintenance_alert',
                'alert_type': 'repair_scheduled',
                'machine_id': machine_id,
                'estimated_duration': repair_record['estimated_repair_time'],
                'severity': repair_record['severity'].value
            }
            self.scheduler_agent.inbox.append((self.name, msg))
        
        print(f"[{self.name}] Repair scheduled for {machine_id} - EST: {repair_record['estimated_repair_time']:.1f} hours")

    def _schedule_preventive_maintenance(self, machine_id):
        """Schedule preventive maintenance"""
        if machine_id in self.machine_agents:
            # Find optimal time slot (when machine is idle)
            maintenance_record = {
                'machine_id': machine_id,
                'type': MaintenanceType.PREVENTIVE,
                'scheduled_time': time.time() + random.uniform(1, 8),  # Schedule within next 8 time units
                'estimated_duration': random.uniform(3, 6),
                'priority': 'medium'
            }
            
            self.maintenance_schedule[machine_id].append(maintenance_record)
            
            # Notify scheduler
            if self.scheduler_agent:
                msg = {
                    'type': 'maintenance_alert',
                    'alert_type': 'scheduled_maintenance',
                    'machines': [machine_id],
                    'scheduled_time': maintenance_record['scheduled_time'],
                    'duration': maintenance_record['estimated_duration']
                }
                self.scheduler_agent.inbox.append((self.name, msg))
            
            print(f"[{self.name}] Preventive maintenance scheduled for {machine_id}")

    def _check_preventive_maintenance_due(self, machine_id):
        """Check if preventive maintenance is due"""
        if machine_id not in self.machine_health:
            return
            
        health_record = self.machine_health[machine_id]
        interval = self.preventive_maintenance_intervals.get(machine_id, 200)
        
        hours_since_maintenance = health_record['operating_hours']
        
        if hours_since_maintenance >= interval and not health_record.get('maintenance_due', False):
            health_record['maintenance_due'] = True
            self._schedule_preventive_maintenance(machine_id)

    def _update_machine_reliability(self, machine_id):
        """Update machine reliability based on breakdown history"""
        if machine_id in self.machine_health and machine_id in self.machine_agents:
            health_record = self.machine_health[machine_id]
            machine_agent = self.machine_agents[machine_id]
            
            # Decrease reliability with more breakdowns
            breakdown_count = health_record.get('breakdown_count', 0)
            reliability_penalty = breakdown_count * 0.02
            
            new_reliability = max(machine_agent.reliability - reliability_penalty, 0.5)
            machine_agent.reliability = new_reliability
            health_record['reliability'] = new_reliability

    def _handle_emergency_alert(self, sender, msg):
        """Handle emergency alerts"""
        emergency_type = msg.get('emergency_type', 'unknown')
        severity = AlertSeverity.CRITICAL
        
        alert = self._create_alert(
            alert_type='emergency',
            machine_id=sender,
            severity=severity,
            message=f"Emergency: {emergency_type}",
            emergency_type=emergency_type
        )
        
        self._broadcast_alert(alert)
        
        # Execute emergency protocol
        if emergency_type in self.emergency_protocols:
            self.emergency_protocols[emergency_type](sender, msg)
        
        self.emergency_incidents += 1

    def _handle_safety_concern(self, sender, msg):
        """Handle safety concerns"""
        concern = msg.get('concern', 'Unknown safety issue')
        
        alert = self._create_alert(
            alert_type='safety_concern',
            machine_id=sender,
            severity=AlertSeverity.HIGH,
            message=f"Safety concern: {concern}"
        )
        
        self._broadcast_alert(alert)

    def _create_alert(self, alert_type, machine_id, severity, message, **kwargs):
        """Create a new alert"""
        alert = {
            'alert_id': len(self.alert_history) + 1,
            'type': alert_type,
            'machine_id': machine_id,
            'severity': severity,
            'message': message,
            'timestamp': time.time(),
            'status': 'active',
            **kwargs
        }
        
        return alert

    def _broadcast_alert(self, alert):
        """Broadcast alert to all relevant agents"""
        self.active_alerts.append(alert)
        self.alert_history.append(alert.copy())
        
        # Notify scheduler
        if self.scheduler_agent:
            scheduler_msg = {
                'type': 'maintenance_alert',
                'alert_type': alert['type'],
                'machine_id': alert['machine_id'],
                'severity': alert['severity'].value,
                'message': alert['message']
            }
            self.scheduler_agent.inbox.append((self.name, scheduler_msg))
        
        # Notify machine agent if applicable
        machine_id = alert.get('machine_id')
        if machine_id and machine_id in self.machine_agents:
            machine_msg = {
                'type': 'maintenance_alert',
                'alert': alert
            }
            self.machine_agents[machine_id].inbox.append((self.name, machine_msg))
        
        print(f"[{self.name}] ALERT BROADCAST: {alert['severity'].value.upper()} - {alert['message']}")

    def _fire_emergency_protocol(self, machine_id, msg):
        """Execute fire emergency protocol"""
        print(f"[{self.name}] FIRE EMERGENCY PROTOCOL ACTIVATED")
        
        # Shutdown all machines in vicinity
        for mid, machine_agent in self.machine_agents.items():
            emergency_msg = {
                'type': 'emergency_stop',
                'reason': 'fire_emergency',
                'immediate': True
            }
            machine_agent.inbox.append((self.name, emergency_msg))

    def _safety_emergency_protocol(self, machine_id, msg):
        """Execute safety emergency protocol"""
        print(f"[{self.name}] SAFETY EMERGENCY PROTOCOL ACTIVATED")
        
        # Stop affected machine and notify safety team
        if machine_id in self.machine_agents:
            emergency_msg = {
                'type': 'emergency_stop',
                'reason': 'safety_emergency'
            }
            self.machine_agents[machine_id].inbox.append((self.name, emergency_msg))

    def _critical_breakdown_protocol(self, machine_id, msg):
        """Execute critical breakdown protocol"""
        print(f"[{self.name}] CRITICAL BREAKDOWN PROTOCOL ACTIVATED")
        
        # Prioritize repair and notify scheduler for rescheduling
        if self.scheduler_agent:
            msg = {
                'type': 'rescheduling_request',
                'reason': f'Critical breakdown on {machine_id}',
                'priority': 'critical',
                'affected_machines': [machine_id]
            }
            self.scheduler_agent.inbox.append((self.name, msg))

    def perform_scheduled_maintenance(self):
        """Execute scheduled maintenance tasks"""
        current_time = time.time()
        
        for machine_id, schedule in self.maintenance_schedule.items():
            for maintenance_record in schedule.copy():
                if (current_time >= maintenance_record['scheduled_time'] and
                    maintenance_record.get('status') != 'completed'):
                    
                    self._execute_maintenance(machine_id, maintenance_record)

    def _execute_maintenance(self, machine_id, maintenance_record):
        """Execute a maintenance task"""
        if machine_id in self.machine_agents:
            machine_agent = self.machine_agents[machine_id]
            
            # Send maintenance command to machine
            maintenance_msg = {
                'type': 'maintenance_schedule',
                'maintenance_type': maintenance_record['type'].value,
                'scheduled_time': maintenance_record['scheduled_time'],
                'duration': maintenance_record['estimated_duration']
            }
            machine_agent.inbox.append((self.name, maintenance_msg))
            
            # Update records
            maintenance_record['status'] = 'in_progress'
            maintenance_record['actual_start_time'] = time.time()
            
            # Update machine health
            if machine_id in self.machine_health:
                self.machine_health[machine_id]['last_maintenance'] = time.time()
                self.machine_health[machine_id]['maintenance_due'] = False
            
            self.total_maintenance_hours += maintenance_record['estimated_duration']
            
            print(f"[{self.name}] Executing {maintenance_record['type'].value} maintenance on {machine_id}")

    def process_repair_queue(self):
        """Process the repair queue"""
        if self.repair_queue:
            # Take highest priority repair
            repair_record = self.repair_queue.pop(0)
            machine_id = repair_record['machine_id']
            
            if machine_id in self.machine_agents:
                machine_agent = self.machine_agents[machine_id]
                
                # Attempt repair
                if machine_agent.perform_repair():
                    self.total_repairs += 1
                    print(f"[{self.name}] Repair completed on {machine_id}")
                    
                    # Clear related alerts
                    self._resolve_alerts(machine_id, 'machine_breakdown')

    def _resolve_alerts(self, machine_id, alert_type):
        """Resolve alerts for a specific machine and type"""
        for alert in self.active_alerts:
            if (alert['machine_id'] == machine_id and 
                alert['type'] == alert_type and 
                alert['status'] == 'active'):
                alert['status'] = 'resolved'
                alert['resolved_time'] = time.time()
        
        # Remove resolved alerts from active list
        self.active_alerts = [alert for alert in self.active_alerts if alert['status'] == 'active']

    def get_maintenance_metrics(self):
        """Get maintenance and alert metrics"""
        active_alerts_by_severity = {severity.value: 0 for severity in AlertSeverity}
        for alert in self.active_alerts:
            active_alerts_by_severity[alert['severity'].value] += 1
        
        return {
            'total_repairs': self.total_repairs,
            'total_maintenance_hours': self.total_maintenance_hours,
            'breakdown_prevented': self.breakdown_prevented,
            'emergency_incidents': self.emergency_incidents,
            'active_alerts': len(self.active_alerts),
            'alerts_by_severity': active_alerts_by_severity,
            'machines_under_maintenance': sum(1 for schedule in self.maintenance_schedule.values() 
                                           if any(r.get('status') == 'in_progress' for r in schedule)),
            'repair_queue_length': len(self.repair_queue)
        }

    def step(self):
        """Main step function"""
        self.process_messages()
        self.perform_scheduled_maintenance()
        self.process_repair_queue()
        
        # Periodic health checks
        for machine_id in self.machine_health:
            self._check_preventive_maintenance_due(machine_id)
        
        # Clean up old resolved alerts
        if len(self.alert_history) > 100:  # Keep last 100 alerts
            self.alert_history = self.alert_history[-100:]
