#this is the logic for the machine code and in case of a conflict 
from langchain.prompts import PromptTemplate

Machine_Execution_Prompt = PromptTemplate(
    input_variables=["machine_name" , "task_id" , "task_name" , "duration_minutes" , "current_schedule"],
    template = (
        "You are a Machine Agent {machine_name} in a factory. "
        "You have a new task to perform with its id as {task_id} that should take {duration_minutes} minutes."
        "Your current schedule is: \n {current_schedule} listed as tasks from start/end\n"
        "Produce a JSON plan(and ONLY the JSON) with these fields : \n"
        "   -plan_start_offset_minutes : integer (minutes from now to when you plan to start)\n"
        "   -plan_duration_minutes : integer (may be same as duration or adjusted)\n"
        "   -preconditions : short list of precondition strings. \n  \n"
        "Be concise and return valid JSON only."
    )
)

Assignment_Prompt = PromptTemplate(
    input_variables = ["task_id", "task_name", "duration_minutes", "machines_overview"],
    template = (
        "You are a production scheduler - The Master Agent. You must pick out the best machines to run task {task_id}. "
        "The name of the task is {task_name} and its duration in minutes is {duration_minutes}. "
        "Machines and their current schedule/abilities is: {machines_overview}. Each machine can handle only one task at a time.\n"
        "Return a strictly JSON ONLY with the following fields:\n"
        "   - machine_id: chosen machine id. Should be an integer\n"
        "   - reasoning: single-sentence justification (plain text)\n"
        "Return valid JSON only."
    )
)


Conflict_Resolution_prompt = PromptTemplate(
    input_variables = ["conflict_description", "options_overview"],
    template = (
        "Two or more tasks conflict on the factory schedule. Description:\n"
        "{conflict_description}\n\n"
        "Options/metadata:\n"
        "{options_overview}\n\n"
        "As the master scheduler, return a JSON decision (only JSON specifically) with fields:\n"
        "  - action: one of ['delay', 'reassign']\n"
        "  - task_to_adjust: the task id to adjust\n"
        "  - delay_minutes: integer (if action == 'delay', otherwise 0)\n"
        "  - reassign_to_machine_id: string (if action == 'reassign', otherwise null)\n\n"
        "Prefer reassigning to an available machine with the right capability; if not possible, suggest a short delay."
    )
)