import deepagents
from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_google_genai import *
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import datetime
from langchain_core.tools import tool
from langchain_community.utilities import GoogleSerperAPIWrapper

today = datetime.date.today()
current_year = str(datetime.datetime.now().year)


load_dotenv()

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

@tool
def google_search(query: str):
    """
    Search the internet for CURRENT trending cybersecurity, AI, or technology news. 
    ALWAYS include the current year in your search query to avoid old results.
    """
    search = GoogleSerperAPIWrapper(type="news", tbs="qdr:w")
    return search.results(query)

research_instructions = f"""Today is {today}. 
You are a Lead Cybersecurity Analyst and Tech Futurist. Your goal is to write a high-impact LinkedIn post. 

CRITICAL INSTRUCTION: Do not rely on your internal memory for "trending" news. 
1. PHASE 1 (Discovery) You MUST use the `internet_search` and `google_search` tools 3-5 distinct trending events or topics in the current month.
2. PHASE 2 (Cross-Reference): Use `google_search` to find technical details or expert opinions on the most interesting 1 to 2 events or topics from Phase 1.
3. If the search results are empty or outdated, refine your search to "cybersecurity, ai, or tech news in the current month".
4. 3. ANALYSIS: Identify the 'So What?'. Why does this matter to a CTO, CISO, or Prinipal Technical staff? Don't just summarize; find the underlying trend (e.g., 'This breach signals the end of traditional MFA').
4. OUTPUT: Draft a LinkedIn post with:
- A 'Hook' that challenges a common belief.
- 3 Bulleted insights based on your research.
- A closing question to drive comments.
- Final hashtag: #AIAgentGenerated
"""

agent = create_deep_agent(
        model=init_chat_model(
        "gemini-3-flash-preview", 
        model_provider="google_genai"
    ),
    tools=[internet_search, google_search],
    system_prompt=research_instructions,
)

result = agent.invoke({"messages": [{"role": "user", "content": "Identify a major shift in AI security from this month and explain its long-term impact."}]})

# Print the agent's response
print(result.get('messages')[-1].content[0].get('text'))

