import gradio as gr

# from model import LLM
from llama_index.llms.ollama import Ollama
from vector_store import FaissStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import ServiceContext
from llama_index.core import set_global_service_context
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dir_path",
    type=str,
    required=True,
    help="Path to the directory",
    default="",
)

args = parser.parse_args()
dir_path = args.dir_path

llm = Ollama(model="llama2", request_timeout=30.0)
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2")
service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
set_global_service_context(service_context)

faiss_store = FaissStore(dir_path)
query_engine = faiss_store.get_query_engine()


def chatbot(query, history):
    res = query_engine.query(query)
    return str(res)


interface = gr.ChatInterface(
    fn=chatbot, title="resRag", description="ask about resumes"
)

interface.launch(server_name="localhost", server_port=9090)
