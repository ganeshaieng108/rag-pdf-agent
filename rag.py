# rag_engine.py
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

def rag_chain(pdf_path: str):
    #Load PDF
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"Total Pages: {len(pages)}")

    #Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    documents = splitter.split_documents(pages)
    print(f"Total Chunks: {len(documents)}")

    #Embed and store
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=documents,       
        persist_directory="./chroma_db"
    )

    #Build retriever + chain
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model="gpt-4", temperature=0)

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    return chain                   

def ask_question(chain, question: str) -> dict:
    result = chain.invoke({"query": question})
    answer = result["result"]
    sources = set(
        doc.metadata.get("page", 0) + 1
        for doc in result["source_documents"]
    )
    return {
        "answer": answer,
        "source_pages": sorted(sources)
    }