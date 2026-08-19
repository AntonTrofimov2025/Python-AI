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


llm = ChatGoogleGenerativeAI(model="gemini-flash-latest")


memory = MemorySaver()


search = TavilySearchResults(max_results=2)


tools = [search]



agent_executor = create_react_agent(llm, tools, checkpointer=memory)


config = {"configurable": {"thread_id": "abc123"}}


for step in agent_executor.stream(
    {"messages": [HumanMessage(content="Hi! I'm Tony! And I live in Berlin :)")]},
    config,
    stream_mode="values",
):
    
    step["messages"][-1].pretty_print()



for step in agent_executor.stream(
    {"messages": [HumanMessage(content="What's the weather like where I live?")]},
    config,
    stream_mode="values",
):
    
    step["messages"][-1].pretty_print()