from dataclasses import dataclass
from typing import Optional 
from datetime import datetime , timedelta

@dataclass
class Task:
    id : str
    name : str
    duration_minutes : int 
    earliest_start : Optional[datetime] = None
    metadata : dict = None
    required_capability : str = ""  # Added required_capability attribute