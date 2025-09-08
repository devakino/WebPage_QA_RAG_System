# A RAG System that takes a webpage url as an input and load the content of the webpage to the system.
# On the system, chunks are formed of the content and them vector embeddings using FAISS.
# Then when we ask any question regarding the content on the webpage, the system answers it.


# Importing the required libraries and packages.
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import HumanMessage

# Load environment variables (for GOOGLE_API_KEY)
load_dotenv()

# Step 1: Load webpage content using the WebBaseLoader
def load_webpage(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    return docs

# Step 2: Chunk the documents sing RecursiveCharacterTextSplitter
def chunk_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],  # Sentence-aware splitting
    )
    return splitter.split_documents(docs)

# Step 3: Build or load vectorstore with cosine similarity (semantic search) usin the FAISS database
def get_vectorstore(chunks, persist_dir="faiss_store"):
    embeddings = HuggingFaceEmbeddings()

    # Checking if the directory already exist. If it exist then further work with that directory.
    if os.path.exists(os.path.join(persist_dir, "index.faiss")):
        print("[+] Loading existing FAISS vectorstore with cosine similarity (semantic search)...")
        vectorstore = FAISS.load_local(
            persist_dir,
            embeddings,
            allow_dangerous_deserialization=True,  # Safe if local and trusted
            normalize_L2=True                      # Enables cosine similarity
        )
    # If directory does not exist, create a new directory.
    else:
        print("[+] Creating new FAISS vectorstore with cosine similarity (semantic search)...")
        vectorstore = FAISS.from_documents(
            chunks,
            embeddings,
            normalize_L2=True                      # Enables cosine similarity
        )
        vectorstore.save_local(persist_dir)

    return vectorstore

# Generate answer using LLM with context from retrieved docs, using "Google Gemini-2.5-pro"
def generate_answer(llm, query, docs):
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = (
        f"Use the following context to answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\nAnswer:"
    )
    messages = [HumanMessage(content=prompt)]
    response = llm(messages)
    return response.content

# Main loop where the program starts
def main():
    # asking the user for the url input
    url = input("Enter a URL: ").strip()
    # Loading he WebPages
    docs = load_webpage(url)
    print(f"[+] Loaded {len(docs)} document(s).")
    # Creating chunks
    chunks = chunk_docs(docs)
    print(f"[+] Split into {len(chunks)} chunks.")
    # Creating embedding from the generated chunks
    vectorstore = get_vectorstore(chunks)
    # Initializing the llm
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-pro")
    # Aaking question regarding the fetched webpage.
    print("\nYou can now ask questions about the page. Type 'exit' to quit.")
    while True:
        query = input("\nYour question: ")
        if query.lower() == "exit":
            print("Goodbye!")
            break

        # Semantic retrieval: fetch top 3 relevant chunks by embedding similarity(seantic/cosine similarity)
        retrieved_docs = vectorstore.similarity_search(query, k=3)

        # Generate answer based on retrieved chunks(Passing the retrieved data to llm with the question, to get perfect answer)
        answer = generate_answer(llm, query, retrieved_docs)
        print(f"\nAnswer:\n{answer}")

if __name__ == "__main__":
    main()
