# Re-export for easier imports
import importlib.util
import sys
import os

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Import 00_event_generator.py
spec = importlib.util.spec_from_file_location("event_generator_module", os.path.join(current_dir, "00_event_generator.py"))
event_generator_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(event_generator_module)

EventGenerator = event_generator_module.EventGenerator
