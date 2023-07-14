#importovanje
from llama_index import VectorStoreIndex, SimpleDirectoryReader,ServiceContext, LLMPredictor
from llama_index import StorageContext, load_index_from_storage
from langchain import HuggingFaceHub
from llama_index import LangchainEmbedding
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import PromptTemplate
from llama_index import set_global_service_context
import streamlit as st

#file loader 
st.title("Sova demo app")
uploaded_file = st.file_uploader("Choose a file or multiple files")
if uploaded_file is not None:
    st.write("File loaded")

#temp slider 
temperature_value = st.slider('Please select the model temperature:', 0.0, 1.0, 0.5)
st.write('Current temperature:', temperature_value)

#odabir modela 
access_token = "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"
repo_id = "google/flan-t5-base"
llm_predictor = LLMPredictor(llm = HuggingFaceHub(
    repo_id=repo_id, 
    model_kwargs= {"temperature": temperature_value, "max_length": 64},
    huggingfacehub_api_token= access_token
))
embed_model = LangchainEmbedding(HuggingFaceEmbeddings())
service_context = ServiceContext.from_defaults(llm_predictor= llm_predictor, embed_model=embed_model)
set_global_service_context(service_context)
#za nalazenje source-a odakle je response 
file_metadata = lambda x : {"filename": x}
def find_source(response):
    max_score = 0
    source = response.source_nodes[0]
    for node in response.source_nodes:
        if node.score > max_score:
            max_score = node.score
            source = node
        if source.score> 0.3:
            return source.node.metadata.get('filename')
        else:
            return "General Knowledge. Cannot verify source."
        
#kreiranje indeksa pri prvom pozivu i updatovanje kasnije
def load_index(dir_path):
    documents = SimpleDirectoryReader(dir_path, filename_as_id=True, file_metadata=file_metadata).load_data()
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
    file_source = find_source(response)
    template = """{answer}
    \nFrom source: {file_source}
    """
    model_prompt = PromptTemplate(input_variables = ['answer', 'file_source'], template=template)
    with st.chat_message("user"):
        st.write(f"Human: {chat_prompt}")
    with st.chat_message("assistant"):
        if response:
            st.write(f"Assistant: {model_prompt.format(answer = response, file_source = file_source)}")




    