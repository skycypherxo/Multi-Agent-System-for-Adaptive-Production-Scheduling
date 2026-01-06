from langchain_community.llms import GPT4All
from langchain.schema import HumanMessage , SystemMessage

import json, re 
from typing import List , Dict , Any , Optional
from dataclasses import dataclass, field
from datetime import datetime , timedelta 
from transformers import pipeline


from tasks import Task
from prompts import Machine_Execution_Prompt


def _now():
    return datetime.now()

def _overlaps(a_start , a_end , b_start , b_end):
    return not (a_end <= b_start or b_end <= a_start )

@dataclass
class ScheduledEntry:
    task : Task
    start : datetime
    end : datetime


@dataclass
class MachineAgent:
    id : str
    name : str 
    capabilities : List[str]
    schedule : List[ScheduledEntry] = field(default_factory=list)
    llm : Any = field(default_factory=lambda: pipeline("text-generation", model="gpt2", device=-1))

    def is_free_between(self, start:datetime , end : datetime) -> bool:
        #loop thru the entire schedule and check if it overlaps or not. If it does return False otherwise return True ezz
        for s in self.schedule:
            if _overlaps(start , end , s.start ,s.end):
                return False
        return True
    
    def next_free(self , earliest: datetime , duration_minutes : int) -> datetime:
        candidate_start = earliest

        while True:
            candidate_end = candidate_start + timedelta(minutes=duration_minutes)
            conflict = False  # Initially, it's false

            for s in self.schedule:
                if _overlaps(candidate_start, candidate_end, s.start, s.end):
                    conflict = True
                    candidate_start = s.end  # Update start time to the end of the conflicting task
                    break

            # If no conflict was found, return the candidate start time
            if not conflict:
                return candidate_start
    def assign_task(self , task : Task , start : datetime):
        end = start + timedelta(minutes = task.duration_minutes)
        self.schedule.append(ScheduledEntry(task = task, start = start, end = end))

        self.schedule.sort(key = lambda x : x.start)
        return start,end
    
    def describe_schedule(self) -> str:
        lines = []
        for s in self.schedule:
            lines.append(f"{s.task.id}({s.task.name}): {s.start.strftime('%H:%M')} - {s.end.strftime('%H:%M')}")
        return "\n".join(lines) if lines else "empty"
    
    def plan_execution_with_llm(self, task : Task, rag_context: str = "") -> Dict[str, Any]:
        prompt_text = Machine_Execution_Prompt.format(
            machine_name=self.name,
            task_id=task.id,
            task_name=task.name,
            duration_minutes=task.duration_minutes,
            current_schedule=self.describe_schedule()
        )
        
        # Add RAG context if available
        if rag_context:
            prompt_text += f"\n\nRelevant Knowledge:\n{rag_context}"

        # Generate a response using the Hugging Face pipeline
        response = self.llm(prompt_text, max_new_tokens=50, num_return_sequences=1, truncation=True)
        text = response[0]["generated_text"]

        # Attempt to parse JSON from the response
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            m = re.search(r"(\{.*\})", text, re.S)
            if m:
                try:
                    return json.loads(m.group(1))
                except json.JSONDecodeError:
                    pass

        # Fallback: return a simple default plan
        return {"plan_start_offset_minutes": 0, "plan_duration_minutes": task.duration_minutes, "preconditions": []}



