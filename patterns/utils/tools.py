from langchain.agents import Tool
from langchain.tools import WikipediaQueryRun
from langchain_community.tools.wikidata.tool import WikidataAPIWrapper, WikidataQueryRun
from langchain_community.tools.reddit_search.tool import RedditSearchRun
from langchain_community.utilities.reddit_search import RedditSearchAPIWrapper
from langchain_community.tools.reddit_search.tool import RedditSearchSchema
from langchain_community.tools.google_scholar import GoogleScholarQueryRun
from langchain_community.utilities.google_scholar import GoogleScholarAPIWrapper
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain_community.tools.human.tool import HumanInputRun
from langchain_community.utilities import WikipediaAPIWrapper, GoogleSerperAPIWrapper
from .code_tool import *
# config stuff
from dotenv import load_dotenv
from pathlib import Path
dotenv_path = Path("/home/ubuntu/config.env")
load_dotenv(dotenv_path=dotenv_path)

# define tools #TODO: add image related, aws services related tools 
wikipedia = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(top_k_results=1), # todo: change it back to 5 -> done to test reflexion
    handle_tool_error=True,
)
wikidata = WikidataQueryRun(
    api_wrapper=WikidataAPIWrapper(),
    handle_tool_error=True,
) 
reddit = RedditSearchRun( # TODO: anonymize below stuff
    api_wrapper=RedditSearchAPIWrapper(
        reddit_client_id="3YuSASt3zg1HV2LkMh22WQ",
        reddit_client_secret="ehIrscW0N_FBTzrFA6hJT5fNwPvwHw",
        reddit_user_agent="learn-agents" # any string
    )
)
google_serper = GoogleSerperAPIWrapper()
google_scholar = GoogleScholarQueryRun(api_wrapper=GoogleScholarAPIWrapper())
pubmed = PubmedQueryRun()
arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper())
wolfram = WolframAlphaAPIWrapper()
human = HumanInputRun()

tools = [
    Tool(
        name="code-generator-executor",
        func=lambda question: code_tool.code_generate_and_execute(question),
        description=(
            "Use this to generate infographics, graphs and charts."
            " Input should be the question itself."
        )
    ),
    Tool(
        name="search-wikipedia",
        func=wikipedia.run,
        description=(
            "Tool that searches the Wikipedia API. "
            "Useful for when you need to answer general questions about "
            "people, places, companies, facts, historical events, or other subjects. "
            "Input should be a search query."
        )
    ),
    # Tool(
    #     name="search-wikidata",
    #     func=wikidata.run,
    #     description=(
    #         "Tool that searches the Wikidata API. "
    #         "Useful for when you need to answer general questions about "
    #         "people, places, companies, facts, historical events, or other subjects. "
    #         "Input should be the exact name of the item you want information about "
    #         "or a Wikidata QID."
    #     ),
    # ),
    # Tool(
    #     name="search-reddit",
    #     func=lambda question: reddit.run(tool_input=RedditSearchSchema(
    #         query=question, sort="new", time_filter="week", subreddit="all", limit="3"
    #     ).dict()),
    #     # func=reddit.run,
    #     description=(
    #         "A tool that searches for posts on Reddit."
    #         "Useful when you need to know post information on a subreddit."
    #     ),
    # ),
    # Tool(
    #     name="search-google",
    #     func=google_serper.run,
    #     description="A search engine. Useful for when you need to answer questions about current events. Input should be a search query.",
    # ),
    Tool(
        name="search-google-scholar",
        func=google_scholar.run,
        description=(
            "Tool for Google Scholar Search. "
            "Useful for when you need to get information about"
            "research papers from Google Scholar"
            "Input should be a search query."
        ),
    ),
    # Tool(
    #     name="search-pubmed",
    #     func=pubmed.run,
    #     description=(
    #         "A wrapper around PubMed. "
    #         "Useful for when you need to answer questions about medicine, health, "
    #         "and biomedical topics "
    #         "from biomedical literature, MEDLINE, life science journals, and online books. "
    #         "Input should be a search query."
    #     ),
    # ),
    Tool(
        name="search-arxiv",
        func=arxiv.run,
        description=(
            "A wrapper around Arxiv.org "
            "Useful for when you need to answer questions about Physics, Mathematics, "
            "Computer Science, Quantitative Biology, Quantitative Finance, Statistics, "
            "Electrical Engineering, and Economics "
            "from scientific articles on arxiv.org. "
            "Input should be a search query."
        ),
    ),
    Tool(
        name="wolfram-aplha",
        func=wolfram.run,
        description=(
            "Tool answers factual queries by computing answers from externally sourced data"
            " ask math problems."
        ),
    ),
    # Tool(
    #     name="human-in-the-loop",
    #     func=human.run,
    #     description=(
    #         "You can ask a human for guidance when you think you "
    #         "got stuck or you are not sure what to do next. "
    #         "The input should be a question for the human."
    #     ),
    # ),

]