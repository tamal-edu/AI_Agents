##### iter-6

# import os
# import torch
# from transformers import AutoTokenizer, AutoModel
# import streamlit as st
# from dotenv import load_dotenv

# # ----------------------------------------------------------------
# # Set the Streamlit page configuration as the very first command.
# # ----------------------------------------------------------------
# st.set_page_config(page_title="Agentic GenAI App", layout="centered")

# # ----------------------------------------------------------------
# # 1. Environment Setup
# # ----------------------------------------------------------------
# load_dotenv()
# os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")
# # Uncomment if you also use OpenAI:
# # os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

# # ----------------------------------------------------------------
# # 2. AGNO and Related Imports
# # ----------------------------------------------------------------
# from agno.agent import Agent
# from agno.models.groq import Groq
# from agno.embedder.base import Embedder
# from agno.tools.duckduckgo import DuckDuckGoTools
# from agno.tools.yfinance import YFinanceTools
# from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
# from agno.vectordb.lancedb import LanceDb, SearchType

# # ----------------------------------------------------------------
# # 3. Custom HuggingFace Embedder
# # ----------------------------------------------------------------
# class HuggingFaceEmbedder(Embedder):
#     def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
#         self.model = AutoModel.from_pretrained(model_name)
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.dimensions = self.model.config.hidden_size

#     def get_embedding(self, text: str) -> list[float]:
#         inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
#         with torch.no_grad():
#             outputs = self.model(**inputs)
#         # Mean pooling over token embeddings
#         return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

#     def get_embedding_and_usage(self, text: str):
#         embedding = self.get_embedding(text)
#         usage = len(text)  # Simple usage metric based on text length
#         return embedding, usage

# # ----------------------------------------------------------------
# # 4. Knowledge Base with Caching
# # ----------------------------------------------------------------
# @st.cache_resource(show_spinner=True)
# def load_kb():
#     kb = PDFUrlKnowledgeBase(
#         urls=["https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
#         vector_db=LanceDb(
#             uri="tmp/lancedb",
#             table_name="recipes",
#             search_type=SearchType.hybrid,
#             embedder=HuggingFaceEmbedder(),
#         ),
#     )
#     kb.load()
#     return kb

# knowledge_base = load_kb()

# # ----------------------------------------------------------------
# # 5. Agent Setup
# # ----------------------------------------------------------------

# # 5A. Web Agent using DuckDuckGoTools
# web_agent = Agent(
#     name="Web Agent",
#     role="Search the web for information",
#     model=Groq(id="qwen-2.5-32b"),
#     tools=[DuckDuckGoTools()],
#     instructions="Always include the sources.",
#     show_tool_calls=True,
#     markdown=True,
# )

# # 5B. Finance Agent using YFinanceTools
# finance_agent = Agent(
#     name="Finance Agent",
#     role="Get financial data",
#     model=Groq(id="qwen-2.5-32b"),
#     tools=[YFinanceTools(
#         stock_price=True,
#         analyst_recommendations=True,
#         stock_fundamentals=True,
#         company_info=True
#     )],
#     instructions="Use tables to display data and always include sources.",
#     show_tool_calls=True,
#     markdown=True,
# )

# # 5C. Chef Agent using PDF Knowledge Base (Thai recipes)
# chef_agent = Agent(
#     name="Chef Agent",
#     description="You are a Thai cuisine expert!",
#     model=Groq(id="qwen-2.5-32b"),
#     instructions=[
#         "Search your knowledge base for Thai recipes.",
#         "If the question is better suited for the web, search the web to fill in gaps.",
#         "Prefer the information in your knowledge base over the web results."
#     ],
#     knowledge=knowledge_base,
#     tools=[DuckDuckGoTools()],
#     show_tool_calls=True,
#     markdown=True,
# )

# # 5D. Team Agent combining Web and Finance Agents
# agent_team = Agent(
#     team=[web_agent, finance_agent],
#     model=Groq(id="qwen-2.5-32b"),
#     instructions=["Always include sources", "Use tables to display data"],
#     show_tool_calls=True,
#     markdown=True,
# )

# # ----------------------------------------------------------------
# # 6. Memory-Aware Agent Response Function with Duplicate Check
# # ----------------------------------------------------------------
# def get_agent_response_with_memory(agent_name: str, prompt: str, stream: bool = False) -> str:
#     """
#     Maintains conversation history (memory) per agent in Streamlit's session state.
#     If the same query was asked before, returns the previous answer.
#     Otherwise, appends the new query to the conversation history and sends the combined prompt to the agent.
#     """
#     # Initialize conversation history in session state if needed.
#     if "conversation_history" not in st.session_state:
#         st.session_state["conversation_history"] = {}
    
#     # Create history for the agent if not exists.
#     if agent_name not in st.session_state["conversation_history"]:
#         st.session_state["conversation_history"][agent_name] = []
#     history = st.session_state["conversation_history"][agent_name]

#     # Check if the same question was asked before.
#     for i in range(0, len(history), 2):
#         if history[i].strip().lower() == ("user: " + prompt.strip().lower()):
#             # If found, return the previously recorded agent response.
#             if i + 1 < len(history):
#                 return history[i+1].replace("Agent: ", "").strip()

#     # Combine previous conversation with the new query.
#     conversation_context = "\n".join(history)
#     combined_prompt = (conversation_context + "\nUser: " + prompt) if conversation_context else "User: " + prompt

#     # Select the appropriate agent.
#     if agent_name == "Web Agent":
#         response_obj = web_agent.run(combined_prompt, stream=stream)
#     elif agent_name == "Finance Agent":
#         response_obj = finance_agent.run(combined_prompt, stream=stream)
#     elif agent_name == "Chef Agent":
#         response_obj = chef_agent.run(combined_prompt, stream=stream)
#     elif agent_name == "Team Agent":
#         response_obj = agent_team.run(combined_prompt, stream=stream)
#     else:
#         return "Unknown agent selection."

#     # Extract the final answer.
#     if hasattr(response_obj, "content") and response_obj.content:
#         response_text = response_obj.content.strip()
#     else:
#         response_text = str(response_obj)

#     # Update the conversation history.
#     history.append("User: " + prompt)
#     history.append("Agent: " + response_text)
#     st.session_state["conversation_history"][agent_name] = history

#     return response_text

# # ----------------------------------------------------------------
# # 7. Sidebar Chat History
# # ----------------------------------------------------------------
# def sidebar_chat_history(agent_name: str):
#     st.sidebar.header("Chat History")
#     if agent_name in st.session_state.get("conversation_history", {}):
#         history = st.session_state["conversation_history"][agent_name]
#         # Display conversation history in reverse chronological order.
#         for msg in reversed(history):
#             st.sidebar.markdown(msg)
#     else:
#         st.sidebar.write("No conversation history yet.")

#     if st.sidebar.button("Clear History"):
#         st.session_state["conversation_history"][agent_name] = []
#         st.sidebar.success("History cleared.")

# # ----------------------------------------------------------------
# # 8. Streamlit App Interface
# # ----------------------------------------------------------------
# def main():
#     st.title("Agentic GenAI App")
#     st.markdown(
#         "This is a simple agentic generative AI application built with **Groq**, **Agno**, **Streamlit**, "
#         "and powerful open source LLMs. The sidebar displays your conversation history (memory)."
#     )

#     # Select the agent from a dropdown.
#     agent_option = st.selectbox(
#         "Select Agent",
#         options=["Web Agent", "Finance Agent", "Chef Agent", "Team Agent"]
#     )

#     user_query = st.text_input("Enter your query:")

#     if st.button("Submit Query"):
#         if user_query.strip() == "":
#             st.warning("Please enter a valid query.")
#         else:
#             st.markdown("**Agent Response:**")
#             response_text = get_agent_response_with_memory(agent_option, user_query, stream=False)
#             st.markdown(response_text)

#     # Show chat history in the sidebar.
#     sidebar_chat_history(agent_option)

# if __name__ == "__main__":
#     main()


#### iter-7 dashboard with multiple agents (future enhancement)

# import os
# import time
# import torch
# from transformers import AutoTokenizer, AutoModel
# import streamlit as st
# from dotenv import load_dotenv

# # ----------------------------------------------------------------
# # Set the Streamlit page configuration as the very first command.
# # ----------------------------------------------------------------
# st.set_page_config(page_title="Agentic GenAI App", layout="centered")

# # ----------------------------------------------------------------
# # 1. Environment Setup
# # ----------------------------------------------------------------
# load_dotenv()
# os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")
# # Uncomment if you also use OpenAI:
# # os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

# # ----------------------------------------------------------------
# # 2. AGNO and Related Imports
# # ----------------------------------------------------------------
# from agno.agent import Agent
# from agno.models.groq import Groq
# from agno.embedder.base import Embedder
# from agno.tools.duckduckgo import DuckDuckGoTools
# from agno.tools.yfinance import YFinanceTools
# from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
# from agno.vectordb.lancedb import LanceDb, SearchType

# # ----------------------------------------------------------------
# # 3. Custom HuggingFace Embedder
# # ----------------------------------------------------------------
# class HuggingFaceEmbedder(Embedder):
#     def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
#         self.model = AutoModel.from_pretrained(model_name)
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.dimensions = self.model.config.hidden_size

#     def get_embedding(self, text: str) -> list[float]:
#         inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
#         with torch.no_grad():
#             outputs = self.model(**inputs)
#         # Mean pooling over token embeddings
#         return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

#     def get_embedding_and_usage(self, text: str):
#         embedding = self.get_embedding(text)
#         usage = len(text)  # Simple usage metric based on text length
#         return embedding, usage

# # ----------------------------------------------------------------
# # 4. Knowledge Base with Caching
# # ----------------------------------------------------------------
# @st.cache_resource(show_spinner=True)
# def load_kb():
#     kb = PDFUrlKnowledgeBase(
#         urls=["https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
#         vector_db=LanceDb(
#             uri="tmp/lancedb",
#             table_name="recipes",
#             search_type=SearchType.hybrid,
#             embedder=HuggingFaceEmbedder(),
#         ),
#     )
#     kb.load()
#     return kb

# knowledge_base = load_kb()

# # ----------------------------------------------------------------
# # 5. Agent Setup
# # ----------------------------------------------------------------

# # 5A. Web Agent using DuckDuckGoTools
# web_agent = Agent(
#     name="Web Agent",
#     role="Search the web for information",
#     model=Groq(id="qwen-2.5-32b"),
#     tools=[DuckDuckGoTools()],
#     instructions="Always include the sources.",
#     show_tool_calls=True,
#     markdown=True,
# )

# # 5B. Finance Agent using YFinanceTools
# finance_agent = Agent(
#     name="Finance Agent",
#     role="Get financial data",
#     model=Groq(id="qwen-2.5-32b"),
#     tools=[YFinanceTools(
#         stock_price=True,
#         analyst_recommendations=True,
#         stock_fundamentals=True,
#         company_info=True
#     )],
#     instructions="Coordinate with other agents if you donot unserstand something and to find the closest symbol/ticker of a certain entity from the internet if you are unable to know. Also Use tables to display data and always include sources.",
#     show_tool_calls=True,
#     markdown=True,
# )

# # 5C. Chef Agent using PDF Knowledge Base (Thai recipes)
# chef_agent = Agent(
#     name="Chef Agent",
#     description="You are a Thai cuisine expert!",
#     model=Groq(id="qwen-2.5-32b"),
#     instructions=[
#         "Search your knowledge base for Thai recipes.",
#         "If the question is better suited for the web, search the web to fill in gaps.",
#         "Prefer the information in your knowledge base over the web results."
#     ],
#     knowledge=knowledge_base,
#     tools=[DuckDuckGoTools()],
#     show_tool_calls=True,
#     markdown=True,
# )

# # 5D. Team Agent combining Web and Finance Agents
# agent_team = Agent(
#     team=[web_agent, finance_agent],
#     model=Groq(id="qwen-2.5-32b"),
#     instructions=["Always include sources", "Use tables to display data"],
#     show_tool_calls=True,
#     markdown=True,
# )

# # ----------------------------------------------------------------
# # 6. Memory-Aware Agent Response Function with Duplicate Check and Logging
# # ----------------------------------------------------------------
# def get_agent_response_with_memory(agent_name: str, prompt: str, stream: bool = False) -> str:
#     """
#     Maintains conversation history (memory) per agent in session_state.
#     If the same query was asked before, returns the previous answer.
#     Otherwise, it appends the new query to the conversation history,
#     logs the query and timestamp, and sends the combined prompt to the agent.
#     """
#     # Initialize conversation history and logs in session_state if needed.
#     if "conversation_history" not in st.session_state:
#         st.session_state["conversation_history"] = {}
#     if "agent_logs" not in st.session_state:
#         st.session_state["agent_logs"] = {}

#     # Create history and logs for the agent if not exists.
#     if agent_name not in st.session_state["conversation_history"]:
#         st.session_state["conversation_history"][agent_name] = []
#     if agent_name not in st.session_state["agent_logs"]:
#         st.session_state["agent_logs"][agent_name] = []
    
#     history = st.session_state["conversation_history"][agent_name]
#     logs = st.session_state["agent_logs"][agent_name]

#     # Check if the same question was asked before.
#     for i in range(0, len(history), 2):
#         if history[i].strip().lower() == ("user: " + prompt.strip().lower()):
#             return history[i+1].replace("Agent: ", "").strip()

#     # Combine conversation history with new query.
#     conversation_context = "\n".join(history)
#     combined_prompt = (conversation_context + "\nUser: " + prompt) if conversation_context else "User: " + prompt

#     # Timestamp the query.
#     timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
#     # Select the appropriate agent.
#     if agent_name == "Web Agent":
#         response_obj = web_agent.run(combined_prompt, stream=stream)
#     elif agent_name == "Finance Agent":
#         response_obj = finance_agent.run(combined_prompt, stream=stream)
#     elif agent_name == "Chef Agent":
#         response_obj = chef_agent.run(combined_prompt, stream=stream)
#     elif agent_name == "Team Agent":
#         response_obj = agent_team.run(combined_prompt, stream=stream)
#     else:
#         return "Unknown agent selection."

#     # Extract the final answer.
#     if hasattr(response_obj, "content") and response_obj.content:
#         response_text = response_obj.content.strip()
#     else:
#         response_text = str(response_obj)

#     # Update the conversation history.
#     history.append("User: " + prompt)
#     history.append("Agent: " + response_text)
#     st.session_state["conversation_history"][agent_name] = history

#     # Log the interaction.
#     logs.append(f"{timestamp} - Query: {prompt} | Response Length: {len(response_text)}")
#     st.session_state["agent_logs"][agent_name] = logs

#     return response_text

# # ----------------------------------------------------------------
# # 7. Sidebar with Chat History and Dashboard
# # ----------------------------------------------------------------
# def sidebar_panel(agent_name: str):
#     panel = st.sidebar.radio("Select Panel", options=["Chat History", "Dashboard"])
#     if panel == "Chat History":
#         st.sidebar.header("Chat History")
#         if agent_name in st.session_state.get("conversation_history", {}):
#             history = st.session_state["conversation_history"][agent_name]
#             # Display conversation history in reverse chronological order.
#             for msg in reversed(history):
#                 st.sidebar.markdown(msg)
#         else:
#             st.sidebar.write("No conversation history yet.")

#         if st.sidebar.button("Clear History", key=f"clear_{agent_name}"):
#             st.session_state["conversation_history"][agent_name] = []
#             st.sidebar.success("History cleared.")
#     else:  # Dashboard
#         st.sidebar.header("Logs/Telemetry")
#         # Show metrics: total queries and log entries.
#         total_queries = 0
#         logs = st.session_state.get("agent_logs", {}).get(agent_name, [])
#         if agent_name in st.session_state.get("conversation_history", {}):
#             total_queries = len(st.session_state["conversation_history"][agent_name]) // 2
#         st.sidebar.markdown(f"**Total Queries:** {total_queries}")
#         st.sidebar.markdown("**Recent Logs:**")
#         for log in logs[-5:][::-1]:
#             st.sidebar.markdown(f"- {log}")
#         if st.sidebar.button("Clear Logs", key=f"clear_logs_{agent_name}"):
#             st.session_state["agent_logs"][agent_name] = []
#             st.sidebar.success("Logs cleared.")

# # ----------------------------------------------------------------
# # 8. Streamlit App Interface
# # ----------------------------------------------------------------
# def main():
#     st.title("Agentic GenAI App")
#     st.markdown(
#         "This app is an agentic generative AI system built with **Groq**, **Agno**, and **Streamlit**. "
#         "It includes conversation memory, duplicate query lookup, and a sidebar dashboard inspired by Phidata's ideas."
#     )

#     # Agent selection.
#     agent_option = st.selectbox(
#         "Select Agent",
#         options=["Web Agent", "Finance Agent", "Chef Agent", "Team Agent"]
#     )

#     # Display the sidebar panel.
#     sidebar_panel(agent_option)

#     user_query = st.text_input("Enter your query:")

#     if st.button("Submit Query"):
#         if user_query.strip() == "":
#             st.warning("Please enter a valid query.")
#         else:
#             st.markdown("**Agent Response:**")
#             response_text = get_agent_response_with_memory(agent_option, user_query, stream=False)
#             st.markdown(response_text)

# if __name__ == "__main__":
#     main()

#### iter-9

import os
import time
import torch
from transformers import AutoTokenizer, AutoModel
import streamlit as st
from dotenv import load_dotenv

# ----------------------------------------------------------------
# Set the Streamlit page configuration as the very first command.
# ----------------------------------------------------------------
st.set_page_config(page_title="Agentic GenAI App", layout="centered")

# ----------------------------------------------------------------
# 1. Credential Management: .env vs TOML (st.secrets)
# ----------------------------------------------------------------
# Load .env file for local development
load_dotenv()

def get_credential(key: str) -> str:
    """
    Retrieve the credential value for the given key.
    Checks st.secrets (TOML method) first; if not found, falls back to os.getenv (.env method).
    """
    # st.secrets is always available in Streamlit Community Cloud. In local development it may be empty.
    if hasattr(st, "secrets") and key in st.secrets and st.secrets[key]:
        return st.secrets[key]
    return os.getenv(key, "")

# Set credentials in os.environ for downstream libraries.
os.environ["GROQ_API_KEY"] = get_credential("GROQ_API_KEY")
# Uncomment if you also use OpenAI:
# os.environ["OPENAI_API_KEY"] = get_credential("OPENAI_API_KEY")

# ----------------------------------------------------------------
# 2. AGNO and Related Imports
# ----------------------------------------------------------------
from agno.agent import Agent
from agno.models.groq import Groq
from agno.embedder.base import Embedder
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.vectordb.lancedb import LanceDb, SearchType

# ----------------------------------------------------------------
# 3. Custom HuggingFace Embedder
# ----------------------------------------------------------------
class HuggingFaceEmbedder(Embedder):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.dimensions = self.model.config.hidden_size

    def get_embedding(self, text: str) -> list[float]:
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Mean pooling over token embeddings
        return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

    def get_embedding_and_usage(self, text: str):
        embedding = self.get_embedding(text)
        usage = len(text)  # Simple usage metric based on text length
        return embedding, usage

# ----------------------------------------------------------------
# 4. Knowledge Base with Caching
# ----------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_kb():
    kb = PDFUrlKnowledgeBase(
        urls=["https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
        vector_db=LanceDb(
            uri="tmp/lancedb",
            table_name="recipes",
            search_type=SearchType.hybrid,
            embedder=HuggingFaceEmbedder(),
        ),
    )
    kb.load()
    return kb

knowledge_base = load_kb()

# ----------------------------------------------------------------
# 5. Agent Setup
# ----------------------------------------------------------------

# 5A. Web Agent using DuckDuckGoTools
web_agent = Agent(
    name="Web Agent",
    role="Search the web for information",
    model=Groq(id="qwen-qwq-32b"),
    tools=[DuckDuckGoTools()],
    instructions="Always include the sources.",
    show_tool_calls=True,
    markdown=True,
)

# 5B. Finance Agent using YFinanceTools
finance_agent = Agent(
    name="Finance Agent",
    role="Get financial data",
    model=Groq(id="qwen-qwq-32b"),
    tools=[YFinanceTools(
        stock_price=True,
        analyst_recommendations=True,
        stock_fundamentals=True,
        company_info=True
    )],
    instructions=("Coordinate with other agents if you do not understand something, and find the closest symbol/ticker "
                  "for an entity from the web when needed. Also, use tables to display data and always include sources."),
    show_tool_calls=True,
    markdown=True,
)

# 5C. Chef Agent using PDF Knowledge Base (Thai recipes)
chef_agent = Agent(
    name="Chef Agent",
    description="You are a Thai cuisine expert!",
    model=Groq(id="qwen-qwq-32b"),
    instructions=[
        "Search your knowledge base for Thai recipes.",
        "If the question is better suited for the web, search the web to fill in gaps.",
        "Prefer the information in your knowledge base over the web results.",
        "Provide links for each source."
    ],
    knowledge=knowledge_base,
    tools=[DuckDuckGoTools()],
    show_tool_calls=True,
    markdown=True,
)

# 5D. Team Agent combining all available agents (Web, Finance, Chef)
agent_team = Agent(
    team=[web_agent, finance_agent, chef_agent],
    model=Groq(id="qwen-qwq-32b"),
    instructions=[
        "Always include sources and links.",
        "Use tables to display data.",
        "If one agent has an error or lacks sufficient information, coordinate with the others and provide the best solution."
    ],
    show_tool_calls=True,
    markdown=True,
)

# ----------------------------------------------------------------
# 6. Memory-Aware Agent Response Function with Duplicate Check and Logging
# ----------------------------------------------------------------
def get_agent_response_with_memory(agent_name: str, prompt: str, stream: bool = False) -> str:
    """
    Maintains conversation history (memory) per agent in session_state.
    If the same query was asked before, returns the previous answer.
    Otherwise, appends the new query to the conversation history,
    logs the query and timestamp, and sends the combined prompt to the agent.
    """
    # Initialize conversation history and logs in session_state if needed.
    if "conversation_history" not in st.session_state:
        st.session_state["conversation_history"] = {}
    if "agent_logs" not in st.session_state:
        st.session_state["agent_logs"] = {}

    # Create history and logs for the agent if not exists.
    if agent_name not in st.session_state["conversation_history"]:
        st.session_state["conversation_history"][agent_name] = []
    if agent_name not in st.session_state["agent_logs"]:
        st.session_state["agent_logs"][agent_name] = []
    
    history = st.session_state["conversation_history"][agent_name]
    logs = st.session_state["agent_logs"][agent_name]

    # Check if the same question was asked before.
    for i in range(0, len(history), 2):
        if history[i].strip().lower() == ("user: " + prompt.strip().lower()):
            return history[i+1].replace("Agent: ", "").strip()

    # Combine conversation history with new query.
    conversation_context = "\n".join(history)
    combined_prompt = (conversation_context + "\nUser: " + prompt) if conversation_context else "User: " + prompt

    # Timestamp the query.
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Select the appropriate agent.
    if agent_name == "Web Agent":
        response_obj = web_agent.run(combined_prompt, stream=stream)
    elif agent_name == "Finance Agent":
        response_obj = finance_agent.run(combined_prompt, stream=stream)
    elif agent_name == "Chef Agent":
        response_obj = chef_agent.run(combined_prompt, stream=stream)
    elif agent_name == "Team Agent":
        response_obj = agent_team.run(combined_prompt, stream=stream)
    else:
        return "Unknown agent selection."

    # Extract the final answer.
    if hasattr(response_obj, "content") and response_obj.content:
        response_text = response_obj.content.strip()
    else:
        response_text = str(response_obj)

    # Update the conversation history.
    history.append("User: " + prompt)
    history.append("Agent: " + response_text)
    st.session_state["conversation_history"][agent_name] = history

    # Log the interaction.
    logs.append(f"{timestamp} - Query: {prompt} | Response Length: {len(response_text)}")
    st.session_state["agent_logs"][agent_name] = logs

    return response_text

# ----------------------------------------------------------------
# 7. Sidebar with Chat History and Logs
# ----------------------------------------------------------------
def sidebar_panel(agent_name: str):
    panel = st.sidebar.radio("Select Panel", options=["Chat History", "Telemetry"])
    if panel == "Chat History":
        st.sidebar.header("Chat History")
        if agent_name in st.session_state.get("conversation_history", {}):
            history = st.session_state["conversation_history"][agent_name]
            # Display conversation history in reverse chronological order.
            for msg in reversed(history):
                st.sidebar.markdown(msg)
        else:
            st.sidebar.write("No conversation history yet.")

        if st.sidebar.button("Clear History", key=f"clear_{agent_name}"):
            st.session_state["conversation_history"][agent_name] = []
            st.sidebar.success("History cleared.")
    else:  # Dashboard
        st.sidebar.header("Logs")
        # Show metrics: total queries and log entries.
        total_queries = 0
        logs = st.session_state.get("agent_logs", {}).get(agent_name, [])
        if agent_name in st.session_state.get("conversation_history", {}):
            total_queries = len(st.session_state["conversation_history"][agent_name]) // 2
        st.sidebar.markdown(f"**Total Queries:** {total_queries}")
        st.sidebar.markdown("**Recent Logs:**")
        for log in logs[-5:][::-1]:
            st.sidebar.markdown(f"- {log}")
        if st.sidebar.button("Clear Logs", key=f"clear_logs_{agent_name}"):
            st.session_state["agent_logs"][agent_name] = []
            st.sidebar.success("Logs cleared.")

# ----------------------------------------------------------------
# 8. Streamlit App Interface
# ----------------------------------------------------------------
def main():
    st.title("Agentic GenAI App")
    st.markdown(
        "This app is an agentic generative AI system built with **Groq**, **Agno**, **HuggingFace** and **Streamlit**. "
        "It includes conversation memory, duplicate query lookup, and Telemetry inspired by Phidata's(**Agnos**) ideas. "
        "The **Team Agent** now coordinates all subordinate agents.\n\n"
    )

    # Agent selection.
    agent_option = st.selectbox(
        "Select Agent",
        options=["Web Agent", "Finance Agent", "Chef Agent", "Team Agent"]
    )

    # Display the sidebar panel.
    sidebar_panel(agent_option)

    user_query = st.text_input("Enter your query:")

    if st.button("Submit Query"):
        if user_query.strip() == "":
            st.warning("Please enter a valid query.")
        else:
            st.markdown("**Agent Response:**")
            response_text = get_agent_response_with_memory(agent_option, user_query, stream=False)
            st.markdown(response_text)

if __name__ == "__main__":
    main()

