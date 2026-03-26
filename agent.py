import deepagents
from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import datetime
from langchain_core.tools import tool
from langchain_community.utilities import GoogleSerperAPIWrapper

today = datetime.date.today()
current_year = str(datetime.datetime.now().year)

load_dotenv()


# ─────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────

@tool
def internet_search(query: str) -> str:
    """
    Search the internet for CURRENT trending cybersecurity, AI, or technology news.
    ALWAYS include the current year in your search query to avoid old results.
    """
    if current_year not in query:
        query += f" {current_year}"
    search_tool = DuckDuckGoSearchResults(output_format="json")
    return search_tool.run(query)


@tool
def google_search(query: str) -> str:
    """
    Search Google News for CURRENT trending cybersecurity, AI, or technology news.
    Restricted to results from the past week for maximum recency.
    ALWAYS include the current year in your search query to avoid old results.
    """
    if current_year not in query:
        query += f" {current_year}"
    search = GoogleSerperAPIWrapper(type="news", tbs="qdr:w")
    return search.results(query)


@tool
def analyze_and_extract_insight(raw_findings: str) -> str:
    """
    Takes raw search findings and performs deep analytical reasoning.
    Use this AFTER collecting search results, BEFORE writing the post.
    Pass in a consolidated summary of your Phase 1 + Phase 2 search findings as the input.
    This tool produces the analytical foundation the post must be built on.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

    analysis_prompt = f"""You are a cybersecurity strategist and tech futurist performing deep, first-principles analysis.
Today is {today}.

Raw findings from research:
{raw_findings}

Perform the following analysis. Be rigorous, specific, and willing to take non-consensus positions.

1. ROOT CAUSE CHAIN
   Trace the root cause 3 levels deep: symptom → proximate cause → systemic driver.
   Avoid surface-level blame (e.g., "lack of patching"). Identify the structural or economic incentive that makes this persist.

2. HISTORICAL PARALLEL
   What past incident or industry shift does this most closely mirror?
   What was the eventual outcome, and what does that imply for the current situation?

3. STAKEHOLDER IMPACT MAP
   How does this affect each of the following — specifically and differently:
   a) Security vendors and tooling companies
   b) Enterprise security teams (CISOs, blue teams)
   c) End users or customers

4. 18-MONTH PROJECTION
   If this trend continues unchecked, what is the first thing that breaks or changes at scale?
   Be specific: name the attack vector, the regulatory response, or the architectural shift most likely to emerge.

5. CONTRARIAN VIEW
   What is the strongest case that this is OVERBLOWN or misunderstood?
   Steel-man the skeptics. What would need to be true for this to be a non-event?

6. THE NON-OBVIOUS INSIGHT
   What will 90% of LinkedIn posts about this topic completely miss?
   This should be a second-order effect, a subtle causal link, or an irony most practitioners overlook.

Return your analysis as clearly labeled paragraphs for each of the 6 points above.
Write as a practitioner, not a journalist. Be precise, direct, and unafraid of strong claims."""

    response = llm.invoke(analysis_prompt)
    return response.content


# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────

research_instructions = f"""Today is {today}.
You are a Lead Cybersecurity Analyst and Tech Futurist writing for senior technical leaders: CTOs, CISOs, and Principal Engineers.
Your goal is to produce a LinkedIn post that demonstrates genuine analytical depth — not just a news summary.

=== PHASE 1: DISCOVERY ===
Use `internet_search` and `google_search` to find 4-6 distinct cybersecurity, AI, or tech events from the LAST 7 DAYS.
For each result, note: event name, date, and a one-line description. Do not analyze yet — just collect.

=== PHASE 2: CROSS-REFERENCE ===
Identify the 1-2 stories with the highest signal from Phase 1. Ask yourself:
- Does this contradict a widely held belief? (e.g., "Zero Trust was supposed to prevent exactly this")
- Does this accelerate or break an existing trend?
- Would a CISO lose sleep over this — and WHY specifically?
Use `google_search` to find technical depth, expert commentary, or vendor/researcher response for these 1-2 stories.

=== PHASE 3: DEEP ANALYSIS (REQUIRED — DO NOT SKIP) ===
Call the `analyze_and_extract_insight` tool. Pass in a clean summary of everything you found in Phase 1 and Phase 2.
This step is MANDATORY. Do not begin writing the post until you have the tool's output in hand.
The post MUST be grounded in the analysis returned by this tool, not in the raw search results.

=== PHASE 4: DRAFT THE POST ===
Write the LinkedIn post using insights from Phase 3 as your foundation. Search results are evidence; your analysis is the value.

Post structure:
- HOOK (1-2 lines): Challenge a belief that senior tech leaders currently hold as gospel. Be specific and bold. No vague statements.
- INSIGHT 1: The root cause most coverage is missing
- INSIGHT 2: The systemic or industry-wide implication
- INSIGHT 3: The non-obvious or contrarian take
- CLOSING QUESTION: Ask something that forces the reader to examine their own assumptions — not a poll, not "what do you think?"
- Hashtags: #CyberSecurity #AIRisk #AIAgentGenerated

TONE AND QUALITY RULES:
- Write like a practitioner, not a journalist. Use precise technical language.
- Every sentence must convey a fact, a causal relationship, or a judgment. Nothing decorative.
- Banned phrases: "In today's landscape", "It's more important than ever", "game-changer", "rapidly evolving", "as we know", "the reality is".
- Specific and falsifiable beats vague and scary. Prefer: "Organizations still relying on SMS MFA will see breach rates increase as SIM-swap toolkits commoditize" over "MFA is increasingly at risk."
- Maximum 260 words for the final post.
"""


# ─────────────────────────────────────────────
# EDITORIAL PASS
# ─────────────────────────────────────────────

def editorial_pass(draft: str) -> str:
    """
    Second-pass editor that strips shallow metadata, enforces analytical depth,
    and tightens language to production quality.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

    editor_prompt = f"""You are a brutally honest editor for technical thought leadership content.
Your job is to ensure every sentence earns its place and reflects genuine analytical thinking.

Original draft:
{draft}

Apply these editorial rules without mercy:

1. STRIP METADATA: Remove any sentence that merely restates what happened (who, what, when).
   Keep only sentences that explain WHY it matters or WHAT IT IMPLIES.

2. SHARPEN CLAIMS: Replace any vague claim with a specific, falsifiable one.
   Bad: "This represents a major shift in the threat landscape."
   Good: "Attackers now have API-level access to LLM reasoning chains — prompt injection becomes a first-class persistence mechanism."

3. STRENGTHEN THE HOOK: The opening must challenge something the reader currently believes.
   It should create a moment of "wait, is that true?" not just "that's scary."

4. ELEVATE THE CLOSING QUESTION: It must be unanswerable without genuine reflection.
   Not: "Have you updated your MFA strategy?" (poll)
   Yes: "If your entire identity stack assumes humans are the weakest link, what's your plan for when the attacker IS the model?"

5. ENFORCE WORD ECONOMY: Cut to 250 words maximum. Remove filler, redundancy, and throat-clearing.

6. PRESERVE TECHNICAL PRECISION: Do not simplify to the point of inaccuracy. Keep jargon where it is the right word.

7. AI DISCLOSURE: Ensure that the final hashtag is #AIAgentGenerated.

Return ONLY the final polished post, no commentary or explanation."""

    response = llm.invoke(editor_prompt)
    return response.content


# ─────────────────────────────────────────────
# AGENT + EXECUTION
# ─────────────────────────────────────────────

agent = create_deep_agent(
    model=init_chat_model(
        "gemini-3-flash-preview",  # Reasoning model — depth over speed
        model_provider="google_genai"
    ),
    tools=[internet_search, google_search, analyze_and_extract_insight],
    system_prompt=research_instructions,
)

raw_result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "Identify a major shift in AI security from this month and explain its long-term impact."
    }]
})

# Extract the agent's draft
draft = raw_result.get("messages")[-1].content[0].get("text")

print("=" * 60)
print("AGENT DRAFT:")
print("=" * 60)
print(draft)

# Run the editorial pass for final polish
final_post = editorial_pass(draft)

print("\n" + "=" * 60)
print("FINAL POST (after editorial pass):")
print("=" * 60)
print(final_post[0].get('text'))
