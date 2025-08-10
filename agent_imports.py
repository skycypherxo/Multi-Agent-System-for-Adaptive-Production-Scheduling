# Import modules with numeric prefixes
from event_generator import EventGenerator
import importlib.util
import sys
import os

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Import 01_job_agent.py as job_agent
spec = importlib.util.spec_from_file_location("job_agent", os.path.join(current_dir, "01_job_agent.py"))
job_agent = importlib.util.module_from_spec(spec)
sys.modules["job_agent"] = job_agent
spec.loader.exec_module(job_agent)

# Import 02_scheduler_agent.py as scheduler_agent
spec = importlib.util.spec_from_file_location("scheduler_agent", os.path.join(current_dir, "02_scheduler_agent.py"))
scheduler_agent = importlib.util.module_from_spec(spec)
sys.modules["scheduler_agent"] = scheduler_agent
spec.loader.exec_module(scheduler_agent)

# Import 03_machine_agent.py as machine_agent
spec = importlib.util.spec_from_file_location("machine_agent", os.path.join(current_dir, "03_machine_agent.py"))
machine_agent = importlib.util.module_from_spec(spec)
sys.modules["machine_agent"] = machine_agent
spec.loader.exec_module(machine_agent)

# Import 04_production_line_agent.py as production_line_agent
spec = importlib.util.spec_from_file_location("production_line_agent", os.path.join(current_dir, "04_production_line_agent.py"))
production_line_agent = importlib.util.module_from_spec(spec)
sys.modules["production_line_agent"] = production_line_agent
spec.loader.exec_module(production_line_agent)

# Import 05_MaintenanceAlertAgent.py as maintenance_alert_agent
spec = importlib.util.spec_from_file_location("maintenance_alert_agent", os.path.join(current_dir, "05_MaintenanceAlertAgent.py"))
maintenance_alert_agent = importlib.util.module_from_spec(spec)
sys.modules["maintenance_alert_agent"] = maintenance_alert_agent
spec.loader.exec_module(maintenance_alert_agent)

# Export the classes
EventGenerator = EventGenerator
JobAgent = job_agent.JobAgent
SchedulerAgent = scheduler_agent.SchedulerAgent
MachineAgent = machine_agent.MachineAgent
ProductionLineAgent = production_line_agent.ProductionLineAgent
MaintenanceAlertAgent = maintenance_alert_agent.MaintenanceAlertAgent
