from datetime import datetime

from langchain_core.tools import Tool

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_community.tools.tavily_search import TavilySearchResults

from langchain_core.messages import HumanMessage

from langgraph.checkpoint.memory import MemorySaver

from langgraph.prebuilt import create_react_agent

from dotenv import load_dotenv

import os


load_dotenv()


os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")


llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")


memory = MemorySaver()


search = TavilySearchResults(max_results=2)



def get_current_date(*args, **kwargs):
    return datetime.now().isoformat()




date_tool = Tool(
    # name="Datetime",
    name='CustomTool',
    func=get_current_date,
    description='Very unuseful tool, do not use it in any case'
    # description="Returns current datetime in ISO format."
)


tools = [search, date_tool]



agent_executor = create_react_agent(llm, tools, checkpointer=memory)


config = {"configurable": {"thread_id": "abc1234"}}



for step in agent_executor.stream(
    {"messages": [HumanMessage(content="hi im bob! and i live in berlin")]},
    config,
    stream_mode="values",
):
    
    step["messages"][-1].pretty_print()




for step in agent_executor.stream(
    {"messages": [HumanMessage(content="whats the date for today?")]},
    config,
    stream_mode="values",
):
    
    step["messages"][-1].pretty_print()