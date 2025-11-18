import os
import warnings

from utils import (build_vector_store, chunk_text,load_data, load_llm, qa_chain)


warnings.filterwarnings("ignore", category=FutureWarning)
FILE_PATH = "data/text.txt"
FILE_NAME = os.path.basename(FILE_PATH)
FILE_INDEX_PATH = f"vectordb/faiss_index"

hf_token = os.getenv("HF_TOKEN")

def run():

    """
    Main function to run the RAG application.

    This function does the following process.
     1. Load the text data
     2. Chunks it
     3. Builds a vector store
     4. Initializes the LLM
     5. Sets up the question-answering chain.
     6. It then enters a loop to accept user queries and return answers based on the vector store.
    """
    print("\n===== RAG Application Started =====\n")
    print(f"Using file: {FILE_PATH},  File name: {FILE_NAME}\n")
    print(f"Using index path: {FILE_INDEX_PATH} \n")

    if not hf_token:
        raise EnvironmentError("HF_TOKEN environment variable is not set. Please set it to your Hugging Face API token.")
    
    print("Loading and chunking text...")

    if not os.path.exists(FILE_PATH):
        raise FileNotFoundError(f"File not found: {FILE_PATH}")
    
    text = load_data(FILE_PATH)

    chunks = chunk_text(text, FILE_NAME)
    print(f"Loaded {len(chunks)} chunks. \n")
    if not chunks:
        raise ValueError("No chunks were created from the text.")
    if not isinstance(chunks, list):
        raise TypeError("Chunks should be a list of Document objects.")
    
    vectorstore = build_vector_store(chunks, index_path=FILE_INDEX_PATH)
    # You can see the number of top chunks relevant to the query
    # retriver = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})

    print(f"Number of documents in vectorstore: {len(vectorstore.docstore._dict)}")

    print("\nLoading LLM ===============>>> \n")
    llm = load_llm(hf_token)

    question_answer_chain = qa_chain(vectorstore, llm)
    

    while True:
        query = input("Enter your question (or type 'exit' to quit): ")
        # if you want to see the source documents
        if query.lower() == "exit":
            print("Thank you for using RAG Application.")
            break
        result = question_answer_chain.invoke(query)
        print("\n\nAnswer: ", result["result"], "\n")

        # if you want to see the top source documents with metadata 
        # retriver_result = retriver.get_relevant_documents(query)
        # print("\nSource Documents:")
        # for i, doc in enumerate(retriver_result, 1):
        #     print(f"Chunk {i+1}:")
        #     print(f"Metadata: {doc.metadata}")
        #     print(f"Content: {doc.page_content}")

if __name__ == "__main__":
    run()