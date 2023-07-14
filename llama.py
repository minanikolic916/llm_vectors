#importovanje
from llama_index import VectorStoreIndex, SimpleDirectoryReader,ServiceContext, LLMPredictor
from llama_index import StorageContext, load_index_from_storage
from langchain import HuggingFaceHub
from llama_index import LangchainEmbedding
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import PromptTemplate
from llama_index import set_global_service_context
import streamlit as st

#streamlit integracija
st.title("Sova demo app")
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    st.write("File loaded")

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

chat_prompt = st.chat_input("Ask a question")
#pravljenje template-a za odgovor
if chat_prompt is not None:
    response = query_engine.query(str(chat_prompt))
    template = """{answer}. \n".
    """
    model_prompt = PromptTemplate(input_variables = ['answer'], template=template)
    with st.chat_message("user"):
        st.write(f"Human: {chat_prompt}")
    with st.chat_message("assistant"):
        if response:
            st.write(f"Assistant: {model_prompt.format(answer = response)}")

#prompt = PromptTemplate.from_template(template)
#print(model_prompt.format(answer = response))




    