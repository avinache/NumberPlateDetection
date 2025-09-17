#import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from ctransformers import AutoModelForCausalLM
import streamlit as st
# used to load model
from langchain.llms import ctransformers
#from langchain_community.llms import ctransformers
# Load data from directory
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
# Embedding
from langchain.embeddings import HuggingFaceEmbeddings
# Vector DB
from langchain.vectorstores import FAISS
# Split text / Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Prompting / Prompt
#from langchain import PromptTemplate
from langchain.prompts import PromptTemplate
#from langchain import prompts
# QUestion and answer
#from langchain.chains import RetrievalQA
#from langchain.chains.retrieval_qa.base import RetrievalQA


from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain



import warnings
warnings.filterwarnings("ignore")
import os
import shutil
#from langchain.prompts import PromptTemplate
from ctransformers import AutoModelForCausalLM

llama_model_path = "C:/Users/mavin/Downloads/llama-2-7b-chat.ggmlv3.q8_0.bin"

Input_dir = "C:/Users/mavin/Documents/GUVI/VS_Studio/env/Scripts/output"

if os.path.exists(Input_dir):
    shutil.rmtree(Input_dir)

os.makedirs(Input_dir, exist_ok=True)

# model = ctransformers(model = llama_model_path,
#                       model_type = "llama",
#                       config = {
#                           "max_new_tokens": 200,
#                           "context_length": 2048,
#                           "temperature": 0.1})

model = AutoModelForCausalLM(
    model="C:/Users/mavin/Downloads/llama-2-7b-chat.ggmlv3.q8_0.bin",
    model_type="llama",
    config={
        "max_new_tokens": 200,
        "context_length": 2048,
        "temperature": 0.1
    }
)


prompt = """
use the information provided and answer the following based o the text provided /
provide correct information /
if you do not know the answer, do not make over /
show answer only /
Example:
      question: what is your name?
      answer: sam
Context : {context}
Questions: {question}
"""
def get_prompt():
  prompt_temp = PromptTemplate( template = prompt, input_variables = ["context", "question"] )
  return prompt_temp

# def ans_ret( llm_model, prompt_, vdb_ ):
#   qa = RetrievalQA.from_chain_type(
#                                    llm = llm_model,
#                                    chain_type_kwargs = {"prompt": prompt_},
#                                    retriever = vdb_.as_retriever( search_kwargs = {"k": 2} ))
#   return qa

def ans_ret(llm_model, prompt_, vdb_):
    # Step 1: Create a document chain with LLM and prompt
    doc_chain = create_stuff_documents_chain(
        llm=llm_model,
        prompt=prompt_
    )
    # Step 2: Create a retrieval chain using the document chain and retriever
    retrieval_chain = create_retrieval_chain(
        retriever=vdb_.as_retriever(search_kwargs={"k": 2}),
        combine_docs_chain=doc_chain
    )
    return retrieval_chain

def final_model():
  emb = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2",
                              model_kwargs = {"device": "cpu"})
  vdb = FAISS.load_local( "db/faissdb", embeddings= emb, allow_dangerous_deserialization= True )
  prom = get_prompt()
  ans = ans_ret(model, prom, vdb)
  return ans


LLM_Model = final_model()

st.title('LLM Model')
st.subheader("Poblem Statement")
uploaded_file = st.file_uploader("Choose a file...", type=["pdf", "txt"])
if uploaded_file is not None:
    save_path = os.path.join(Input_dir, uploaded_file.name)
    print(f"saved path  : {save_path}")
    data_load = DirectoryLoader(save_path,glob="*.pdf", loader_cls= PyPDFLoader)
    txt_load = DirectoryLoader(save_path,glob="*.txt",loader_cls=TextLoader)
    doc = data_load.load()+txt_load.load()
    split_data = RecursiveCharacterTextSplitter( chunk_size = 500, chunk_overlap = 50 )
    doc_split = split_data.split_documents(doc)
    embadding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2", model_kwargs = {"device": "cpu"})
    vect = FAISS.from_documents(doc_split, embadding)
    vect.save_local("db/faissdb")
    question1 = st.text_input("Enter your question: ")
    response = LLM_Model({"query": question1})
    print(response["answer"]) 





