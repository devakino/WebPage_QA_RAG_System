# Creating a system where we pass the url of a webite and ask question to the system regarding the Website.
# The system can answer all the questions based on the data of the website.
# The workflow - the system scraps all the data form every website url.
# Then it forms chunks of that fetched data
# Then it forms embeddings of those chunks and store them in the FAISS database, it is different for different urls.
# And when any url is entered for the secod time, the system automatically use the earlier fetched data instead of scrapping again.
# Then whenever we ask a question, the system retrieve the mosr relevant chunk according to the cosine similarity.
# Then the chunk is passed to the llm(Google Gemini-2.5-flash) with the question and the perfect answer is generated.

# Importing the packages used for WebScrapping the provided url
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from dotenv import load_dotenv

# Importing the langchain packages that are required
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.schema import HumanMessage

# Load environment variables (for GOOGLE_API_KEY)
load_dotenv()


# Crawl internal pages of a website
def crawl_website(base_url, max_pages=30):
    visited = set()
    to_visit = [base_url]
    pages = []

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited or urlparse(url).netloc != urlparse(base_url).netloc:
            continue

        try:
            response = requests.get(url, timeout=5)
            if "text/html" not in response.headers.get("Content-Type", ""):
                continue

            visited.add(url)
            soup = BeautifulSoup(response.text, "html.parser")
            text = soup.get_text(separator="\n")
            pages.append((url, text))

            for link in soup.find_all("a", href=True):
                full_url = urljoin(url, link['href'])
                if full_url not in visited:
                    to_visit.append(full_url)

        except Exception as e:
            print(f"[!] Failed to load {url}: {e}")

    return pages


# Convert raw pages to LangChain documents
def load_website(base_url):
    pages = crawl_website(base_url)
    print(f"[+] Crawled {len(pages)} page(s) from {base_url}")
    docs = [Document(page_content=text, metadata={"source": url}) for url, text in pages]
    return docs


# Chunk documents into overlapping pieces
def chunk_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
    )
    return splitter.split_documents(docs)


# Use FAISS to store or load vectorstore
def get_vectorstore(chunks, persist_dir):
    embeddings = HuggingFaceEmbeddings()

    faiss_index_path = os.path.join(persist_dir, "index.faiss")

    if os.path.exists(faiss_index_path):
        print(f"[+] Found existing FAISS vectorstore for site: {persist_dir}")
        vectorstore = FAISS.load_local(
            persist_dir,
            embeddings,
            allow_dangerous_deserialization=True,
            normalize_L2=True
        )
    else:
        print(f"[+] Creating FAISS vectorstore for site: {persist_dir}")
        os.makedirs(persist_dir, exist_ok=True)
        vectorstore = FAISS.from_documents(
            chunks,
            embeddings,
            normalize_L2=True
        )
        vectorstore.save_local(persist_dir)

    return vectorstore


# Generate answer using Gemini + context from vectorstore
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


# Main logic
def main():
    base_url = input("Enter the website URL (e.g., https://angularjs.org): ").strip()
    domain = urlparse(base_url).netloc.replace(".", "_")
    persist_dir = os.path.join("faiss_data", domain)

    # If vectorstore exists, skip crawling
    if os.path.exists(os.path.join(persist_dir, "index.faiss")):
        print(f"[+] Using cached data for {base_url}")
        docs = []  # Not needed if we already have FAISS index
        chunks = []  # Not needed either
    else:
        docs = load_website(base_url)
        print(f"[+] Loaded {len(docs)} document(s).")
        chunks = chunk_docs(docs)
        print(f"[+] Split into {len(chunks)} chunks.")

    vectorstore = get_vectorstore(chunks, persist_dir)
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-pro")

    print(f"\nYou can now ask questions about {base_url}. Type 'exit' to quit.")
    while True:
        query = input("\nYour question: ")
        if query.lower() == "exit":
            print("Goodbye!")
            break

        retrieved_docs = vectorstore.similarity_search(query, k=3)
        answer = generate_answer(llm, query, retrieved_docs)
        print(f"\nAnswer:\n{answer}")


if __name__ == "__main__":
    main()
