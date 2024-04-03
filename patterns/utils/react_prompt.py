from typing import Sequence
from langchain.tools import BaseTool
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate

AGENT_PREFIX = """You are an AI assistant. You always stays on topic of the human input and does not diverge from it."""

AGENT_FORMAT_INSTRUCTIONS = """You have access to the following tools:
{tool_descriptions}

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question"""

AGENT_QUESTION_PROMPT = """Remember to respond with your knowledge when the question does not correspond to any tool.
The previous conversation is below, where H refers to the human and A refers to the assistant:
{chat_history}

Always append "Final Answer:" when returning the final answer.
Question: {question}"""

AGENT_SUFFIX = """Thought: {agent_scratchpad}"""

def create_agent_prompt(
        tools: Sequence[BaseTool],
        prefix: str = AGENT_PREFIX,
        # context: str = AGENT_CONTEXT,
        format_instructions: str = AGENT_FORMAT_INSTRUCTIONS,
        question_prompt: str = AGENT_QUESTION_PROMPT,
        suffix: str = AGENT_SUFFIX,
    ) -> PromptTemplate:
    
        human_prompt = PromptTemplate(
            input_variables=[
                "chat_history", 
                "question",
                "agent_scratchpad"
            ],
            partial_variables={
                "tool_names": ", ".join([tool.name for tool in tools]),
                "tool_descriptions": "\n".join(
                    [f"    {tool.name}: {tool.description}" for tool in tools]
                )
            },
            template='\n'.join(
                [
                    prefix,
                    format_instructions,
                    # context,
                    question_prompt,
                    suffix
                ]
            )
        )
        human_message_prompt = HumanMessagePromptTemplate(prompt=human_prompt)
        return ChatPromptTemplate.from_messages([human_message_prompt])
