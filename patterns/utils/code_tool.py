import boto3
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_models import BedrockChat
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate

code_gen_template = """Write some python code to plot a graph for the given fields in the below query and the data. Also in the code written to generate the graph, save the figure as output.png using plt.save()

{question}

Return only python code in Markdown format, e.g.:

```python
....
```"""
code_gen_prompt = ChatPromptTemplate.from_template(code_gen_template)

llm = BedrockChat(
    model_id="anthropic.claude-v2:1", 
    # model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    client=boto3.client("bedrock-runtime"),
    model_kwargs={
        "max_tokens_to_sample": 512, 
        "temperature": 0.0
    },
)

def _sanitize_output(text: str):
    _, after = text.split("```python")
    return after.split("```")[0]

def code_generate_and_execute(query: str) -> str:
    """Use this function to generator infographics/graphs/charts"""

    chain = ( 
        {"question" : RunnablePassthrough()}
        | code_gen_prompt 
        # | ChatAnthropic(model="claude-3-sonnet-20240229") 
        | llm
        | StrOutputParser() 
        | _sanitize_output 
        
    )
    #| PythonREPL().run

    response = chain.invoke(query)
    print(response)
    exec(response)

    return response