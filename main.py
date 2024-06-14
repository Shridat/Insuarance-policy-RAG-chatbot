import streamlit as st
import pdfplumber
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
import re

@st.cache_data
def extract_pdf(pdf_path):
    text = ''
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

@st.cache_data
def split_text(text):
    text_splitter = CharacterTextSplitter(separator = '\n',chunk_size=1000, chunk_overlap=200, length_function=len)
    return text_splitter.split_text(text)

@st.cache_data
def clean_text(text):
    # Remove special characters and multiple new lines
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with a single space
    text = re.sub(r'[^A-Za-z0-9,.?!\s]', '', text)  # Remove non-alphanumeric characters
    return text

@st.cache_resource
def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/distilbert-base-nli-stsb-mean-tokens",model_kwargs={'device':'cpu'},encode_kwargs={'normalize_embeddings':True})
    docs = [Document(page_content=clean_text(chunk)) for chunk in chunks]
    return FAISS.from_documents(docs,embeddings)

@st.cache_resource
def load_huggingface_model():
    generator_model = GPT2LMHeadModel.from_pretrained('distilgpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    return generator_model, tokenizer

text = extract_pdf('policy-booklet-0923.pdf')
chunks = split_text(text)
doc_search = create_vector_store(chunks)
generator_model, tokenizer = load_huggingface_model()
llm = pipeline(task='text-generation',model=generator_model,tokenizer=tokenizer, max_length=1024)
hf_pipeline = HuggingFacePipeline(pipeline=llm,model_kwargs={'device':'cpu','max_new_tokens': 512,"temperature": 0})

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    Use the following context to answer the question as accurately and concisely as possible. 

    Context: {context}
    Question: {question}

    Answer:
    """
)
retriever = doc_search.as_retriever(search_type="similarity",search_kwargs={"k":3})
qa_chain = load_qa_chain(llm=hf_pipeline, chain_type="stuff", prompt=prompt_template)
qa = RetrievalQA(
    retriever=retriever,
    combine_documents_chain=qa_chain,
    return_source_documents=True,
)
def generate_response(query):
    result = qa({"query": query})
    return result["result"]
    #return response

st.title("RAG Powered Chatbot")
st.header("Ask a Question about the Policy")
query = st.text_input("Enter your question:")
if query:
    with st.spinner('Generating Response...'):
        response = generate_response(query)
        st.write("Answer:")
        st.write(response.split('Answer:')[1].strip())



