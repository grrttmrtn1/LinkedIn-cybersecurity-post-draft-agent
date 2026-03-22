import deepagents
from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_google_genai import *
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import datetime
from langchain_core.tools import tool

today = datetime.date.today()
current_year = str(datetime.datetime.now().year)


load_dotenv()

GoogleModel = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=1.0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

@tool
def internet_search(query: str):
    """
    Search the internet for CURRENT trending cybersecurity, AI, or technology news. 
    ALWAYS include the current year in your search query to avoid old results.
    """
    # Force the query to be about NOW
    if current_year not in query:
        query += f" {current_year}"
        
    search_tool = DuckDuckGoSearchResults(output_format="json")
    return search_tool.run(query)

research_instructions = f"""Today is {today}. 
You are a Senior Cybersecurity Researcher. 

CRITICAL INSTRUCTION: Do not rely on your internal memory for "trending" news. 
1. You MUST use the `internet_search` tool to find events that happened in the current month.
2. Look for specific incidents this year (e.g., the Stryker network disruption or the Tycoon 2FA takedown).
3. If the search results are empty or outdated, refine your search to "cybersecurity or tech news in the current month".
4. Draft a LinkedIn post only AFTER you have confirmed data from the current year.
5. Make sure any hashtags include a final #AIAgentGenerated hashtag.
"""

agent = create_deep_agent(
        model=init_chat_model(
        "gemini-2.5-flash", 
        model_provider="google_genai"
    ),
    tools=[internet_search],
    system_prompt=research_instructions,
)

result = agent.invoke({"messages": [{"role": "user", "content": "Draft a linkedin post based on trending cybersecurity, AI, or technology news from the last month. Create this with engagement in mind."}]})

# Print the agent's response
print(result.get('messages')[-1].content[0].get('text'))

