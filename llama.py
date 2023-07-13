#importovanje
from llama_index import VectorStoreIndex, SimpleDirectoryReader,ServiceContext, LLMPredictor
from llama_index import StorageContext, load_index_from_storage
from langchain import HuggingFaceHub
from llama_index.callbacks import CallbackManager, LlamaDebugHandler
from llama_index import GPTVectorStoreIndex, LangchainEmbedding
from llama_index.llms import HuggingFaceLLM
from transformers import AutoTokenizer, TFAutoModelForQuestionAnswering
from langchain.embeddings import HuggingFaceEmbeddings
#kreiranje indeksa na osnovu ucitanih dokumenata

access_token = "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"
repo_id = "google/flan-t5-base"
llm_predictor = LLMPredictor(llm = HuggingFaceHub(
    repo_id=repo_id, 
    model_kwargs= {"temperature": 0.5, "max_length": 64},
    huggingfacehub_api_token= access_token
))
embed_model = LangchainEmbedding(HuggingFaceEmbeddings())
service_context = ServiceContext.from_defaults(llm_predictor= llm_predictor, embed_model=embed_model)

documents = SimpleDirectoryReader('docs_database').load_data()
index = VectorStoreIndex.from_documents(documents, service_context=service_context)
index.set_index_id("sova_vector_index")

#perzistiramo ceo indeks
index.storage_context.persist()
query_engine = index.as_query_engine()

response = query_engine.query("How are CNNs also know?")
print(response)