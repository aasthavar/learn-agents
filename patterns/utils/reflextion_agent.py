import boto3
from langchain_community.chat_models import BedrockChat
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import get_buffer_string
from .reflextion_prompt import *
from .tools import tools
from typing import List


def get_main_agent_scratchpad(intermediate_steps, final_answer):
    scratchpad = format_log_to_str(intermediate_steps)
    scratchpad += f"I now have final answer.\nFinal Answer: {final_answer}"
    return scratchpad

def format_reflections(reflections: List[str],
                        header: str = self_reflect_header) -> str:
    if reflections == []:
        return ''
    else:
        return header + 'Reflections:\n- ' + '\n- '.join([r.strip() for r in reflections])

def get_main_agent():
    prompt = create_agent_prompt(tools=tools)
    llm = BedrockChat(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        client=boto3.client("bedrock-runtime"),
        model_kwargs={
            "max_tokens": 4096, 
            "temperature": 0.5,
        },
        streaming=False,
        region_name="us-west-2",
    )
    output_parser = ReActSingleInputOutputParser()

    # TODO: write a format_reflections() - commented in one of the cells above
    agent = (
        {
            "reflections": lambda x: x["reflections"],
            "chat_history": lambda x: get_buffer_string(x["chat_history"]),
            # "chat_history": lambda x: x["chat_history"]
            "question": lambda x: x["question"],
            "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
        }
        | prompt
        | llm.bind(stop=["\nObservation"])
        | output_parser
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True, 
        input_key="question",
        output_key="output",
        ai_prefix="A",
        human_prefix="H",
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True
    )

    return agent_executor


def get_evaluator_agent():
    from langchain.tools import WikipediaQueryRun
    from langchain_community.utilities import WikipediaAPIWrapper
    wikipedia = WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(top_k_results=1), # todo: change it back to 5 -> done to test reflexion
        handle_tool_error=True,
    )
    # tools = [wikipedia]
    
    from langchain.agents import Tool
    from langchain_community.utilities import GoogleSerperAPIWrapper
    google_serper = GoogleSerperAPIWrapper(k=3)
    eval_tools = [
        wikipedia,
        Tool(
            name="search-google",
            func=google_serper.run,
            description="A search engine. Useful for when you need to answer questions about current events. Input should be a search query.",
        ),
    ]
    
    prompt = create_evaluator_prompt(tools=eval_tools)

    llm = BedrockChat(
        # model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        client=boto3.client("bedrock-runtime"),
        model_kwargs={
            "max_tokens": 4096, 
            "temperature": 0.0,
        },
        streaming=False,
        region_name="us-west-2",
    )

    output_parser = ReActSingleInputOutputParser()

    agent = (
        {
            "answer": lambda x: x["answer"],
            "question": lambda x: x["question"],
            "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
        }
        | prompt
        | llm.bind(stop=["\nObservation"])
        | output_parser
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=eval_tools,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True
    )

    return agent_executor


def get_reflexion_chain():
    prompt = create_self_reflect_prompt(tools=tools)

    llm = BedrockChat(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        client=boto3.client("bedrock-runtime"),
        model_kwargs={
            "max_tokens": 4096, 
            "temperature": 0.0,
        },
        streaming=False,
        region_name="us-west-2",
    )

    output_parser = StrOutputParser()

    chain = (
        {
            "question": lambda x: x["question"],
            "scratchpad": lambda x: x["scratchpad"],
        }
        | prompt
        | llm
        | output_parser
    )
    return chain
