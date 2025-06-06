import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from llama_index.core import SimpleDirectoryReader, KnowledgeGraphIndex
from llama_index.llms.langchain import LangChainLLM
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()
api_key = os.environ["OPENAI_API_KEY"]
open_router_api_base = os.environ["OPEN_ROUTER_API_BASE"]

# Load and split data
article = SimpleDirectoryReader(input_files=["../data/2505.23724v1.txt"]).load_data()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = [chunk for chunk in article for chunk in splitter.split_text(chunk.text)]

# Embed chunks
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(chunks, show_progress_bar=True)

print(embeddings.shape)

# Uncomment to use the LLM
llm = ChatOpenAI(
    base_url=open_router_api_base,
    model_name="meta-llama/llama-4-scout:free",
    openai_api_key=api_key,
)
wrapped_llm = LangChainLLM(llm=llm)
