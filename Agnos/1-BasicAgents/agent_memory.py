from agno.agent import Agent
#from agno.models.openai import OpenAIChat
#from agno.embedder.openai import OpenAIEmbedder

from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.vectordb.lancedb import LanceDb, SearchType
import os
from dotenv import load_dotenv
load_dotenv()

#os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")


###
#Custom Huggingface Embedder
from agno.embedder.base import Embedder
from transformers import AutoTokenizer, AutoModel
import torch

class HuggingFaceEmbedder(Embedder):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.dimensions = self.model.config.hidden_size

    def get_embedding(self, text: str) -> list[float]:
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

    def get_embedding_and_usage(self, text: str):
        embedding = self.get_embedding(text)
        usage = len(text)  # Simple usage metric: length of the text
        return embedding, usage

###

agent = Agent(
    model=Groq(id="qwen-2.5-32b"),
    description="You are a Thai cuisine expert!",
    instructions=[
        "Search your knowledge base for Thai recipes.",
        "If the question is better suited for the web, search the web to fill in gaps.",
        "Prefer the information in your knowledge base over the web results."
    ],
    knowledge=PDFUrlKnowledgeBase(
        urls=["https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
        vector_db=LanceDb(
            uri="tmp/lancedb",
            table_name="recipes",
            search_type=SearchType.hybrid,
            #embedder=OpenAIEmbedder(id="text-embedding-3-small"),
            embedder=HuggingFaceEmbedder()  # Explicitly use your custom embedder
        ),
    ),
    tools=[DuckDuckGoTools()],
    show_tool_calls=True,
    markdown=True
)

# Comment out after the knowledge base is loaded
if agent.knowledge is not None:
    agent.knowledge.load()

agent.print_response("How do i make an indian dish?", stream=True)

agent.print_response(input("Enter you rag query"), stream=True)
# agent.print_response("How do I make chicken and galangal in coconut milk soup", stream=True)
# agent.print_response("What is the history of Thai curry?", stream=True)