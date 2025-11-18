import os
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv
load_dotenv()

def load_data(file_path: str) -> str:

    _, ext = os.path.splitext(file_path)
    if ext.lower() != ".txt":
        raise ValueError(f"Unsupported file type: {ext}. Only .txt files are supported.")
    
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()
    
def chunk_text(text: str, file_name: str, chunk_size: int = 300, chunk_overlap: int = 50) -> List[Document]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    raw_chunks = splitter.split_text(text)
    chunks = [Document(page_content=chunk, metadata={"file_name": file_name}) for chunk in raw_chunks]

    return chunks

def build_vector_store(
    chunks: List[Document],
    embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    index_path: str = "default_faiss_index",
) -> FAISS:

    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)

    if os.path.exists(index_path):
        print(f"Loading existing FAISS vectordb index from: {index_path}")
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

    for i, doc in enumerate(chunks):
        if "doc_id" not in doc.metadata:
            doc.metadata["doc_id"] = f"doc_{i}"  

    print("Building new FAISS vectordb index ===============>>>")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_path)

    return vectorstore


def load_llm(hf_token: str = None) -> ChatOpenAI:

    return ChatOpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key= hf_token,
        model="meta-llama/Llama-3.1-8B-Instruct:novita",
        temperature=0.5
    )


def qa_chain(vectorstore, llm) -> RetrievalQA:
    prompt_template = """
    You are a helpful AI assistant. Use only the following context to answer the question.
    If the answer is not in the context, respond with "I don't know. Please ask relevant question !.".

    Context:
    {context}

    Question:
    {question}

    Answer:
    """ 
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(k=3),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )


