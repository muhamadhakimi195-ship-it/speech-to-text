import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
# import numpy as np
# import tiktoken
os.environ.setdefault("USER_AGENT", "VoiceAIAgent/1.0")

from langchain_classic import hub
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

LANGCHAIN_TRACING_V2=os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_ENDPOINT=os.getenv("LANGCHAIN_ENDPOINT")
LANGCHAIN_API_KEY=os.getenv("LANGCHAIN_API_KEY")
OLLAMA_BASE_URL=os.getenv("OLLAMA_BASE_URL")

LLM_MODEL="ministral-3:3b"
LLM_MODEL2="llama3.2:3b"
EMBEDDING_MODEL="embeddinggemma"
DOC_SOURCE="https://www.ooidental.com/faq"

### INDEXING

# Load the web page directly without the deprecated community loader.
response = requests.get(
    DOC_SOURCE,
    headers={"User-Agent": os.environ["USER_AGENT"]},
    timeout=30,
)
response.raise_for_status()
response.encoding = response.apparent_encoding
soup = BeautifulSoup(response.text, "html.parser")

metadata = {"source": DOC_SOURCE}
if title := soup.find("title"):
    metadata["title"] = title.get_text()
if description := soup.find("meta", attrs={"name": "description"}):
    metadata["description"] = description.get("content", "No description found.")
if html := soup.find("html"):
    metadata["language"] = html.get("lang", "No language found.")

docs = [Document(page_content=soup.get_text(separator="\n", strip=True), metadata=metadata)]

#Split
text_splitter= RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits= text_splitter.split_documents(docs)

#Embed
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    ),
    )

retriever= vectorstore.as_retriever()

#### Retrieval and Generation

## prompt
prompt = ChatPromptTemplate.from_template(
    "You are a customer service of clinic that replying customers messages."
    "Answer the question using only the context below. Make it compact answer no need to be too lengthy but the explanation should be clear"
    "If the context does not contain the answer, say you don't know. "
    "Respond naturally to greetings, thanks, and acknowledgments. "
    "For those messages, do not introduce facts from the context. "
    "Use the context only when it is relevant to the user's question. "
    "Keep the answer concise. Also end with asking if there anything else i could help?\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)

## LLM
llm = ChatOllama(
    model=LLM_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.2,
    num_ctx=4096,
    num_predict=128,
)

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
rag_chain = (
    {
        "context": retriever | format_docs, 
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    |StrOutputParser()
)
import time
# Question
if __name__ == "__main__":
    while True:
        question = input("Ask CS Questions: ").strip()
        if question.lower() in ("end", "exit"):
            break
        if not question:
            continue
        print("Assistant: ", end="", flush=True)
        normalized = question.lower().strip().rstrip(".!?")

        small_talk = {
            "hi": "Hi! How can I help you today?",
            "hello": "Hello! How can I help you today?",
            "okay": "Sure. Is there anything else you'd like to ask?",
            "ok": "Sure. Is there anything else you'd like to ask?",
            "thank you": "You're welcome!",
            "thanks": "You're welcome!",
        }
        
        if normalized in small_talk:
            print(f"Assistant: {small_talk[normalized]}", flush=True)
            continue
           
        for text in rag_chain.stream(question):
            print(text, end="", flush=True)

        print()
        



