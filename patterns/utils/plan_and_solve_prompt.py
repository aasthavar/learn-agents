from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage

### -------------------------------------------- PLANNER -------------------------------------------- ###
planner_prompt_system_content = """Let's first understand the problem and devise a plan to solve the problem.
Please output the plan starting with the header 'Plan:' and then followed by a numbered list of steps.
Please make the plan the minimum number of steps required to accurately complete the task. 
If the task is a question, the final step should almost always be 'Given the above steps taken, please respond to the users original question'.
At the end of your plan, say '<END_OF_PLAN>'"""

planner_prompt_human_template = "{input}"

planner_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=planner_prompt_system_content),
        HumanMessagePromptTemplate.from_template(planner_prompt_human_template),
    ]
)

### -------------------------------------------- EXECUTOR -------------------------------------------- ###

executor_prompt_human_template = """{objective}

Previous steps: {previous_steps}

Current objective: {current_step}

{agent_scratchpad}"""

executor_prompt_input_variables = ["objective", "previous_steps", "current_step", "agent_scratchpad"]