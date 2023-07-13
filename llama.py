#importovanje
from llama_index import VectorStoreIndex, SimpleDirectoryReader,ServiceContext, LLMPredictor
from llama_index import StorageContext, load_index_from_storage
from langchain import HuggingFaceHub
from llama_index import LangchainEmbedding
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import PromptTemplate
from llama_index import set_global_service_context
#odabir modela 
access_token = "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"
repo_id = "google/flan-t5-base"
llm_predictor = LLMPredictor(llm = HuggingFaceHub(
    repo_id=repo_id, 
    model_kwargs= {"temperature": 0.5, "max_length": 64},
    huggingfacehub_api_token= access_token
))
embed_model = LangchainEmbedding(HuggingFaceEmbeddings())
service_context = ServiceContext.from_defaults(llm_predictor= llm_predictor, embed_model=embed_model)
set_global_service_context(service_context)
#kreiranje indeksa pri prvom pozivu i updatovanje kasnije
def load_index(dir_path):
    documents = SimpleDirectoryReader(dir_path, filename_as_id=True).load_data()
    try:
        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        index = load_index_from_storage(storage_context)
        print("Index loaded from storage")
    except FileNotFoundError:
        #pravimo indeks od nule samo pri prvom pozivu
        index = VectorStoreIndex.from_documents(documents, service_context=service_context)
        index.set_index_id("sova_vector_index")
        index.storage_context.persist()
        print("New index created")

    refreshed_docs = index.refresh_ref_docs(documents, update_kwargs={"delete_kwargs": {'delete_from_docstore': True}})
    print('Number of newly inserted/refreshed docs: ', sum(refreshed_docs))
    index.storage_context.persist()
    return index 

index = load_index('./docs_database')
query_engine = index.as_query_engine()

#pravljenje template-a za odgovor
response = query_engine.query("What are word embeddings used for")
template = """Answer: {answer}. \nIf the question cannot be answered using the information provided answer with "Sorry, but I can't provide an answer".
"""
model_prompt = PromptTemplate(input_variables = ['answer'], template=template)

#prompt = PromptTemplate.from_template(template)
print(model_prompt.format(answer = response))