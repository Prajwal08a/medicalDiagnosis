import os
import time
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from tqdm.auto import tqdm
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except Exception:
    # Minimal fallback in case langchain isn't installed or has a different structure.
    class RecursiveCharacterTextSplitter:
        def __init__(self, chunk_size=500, chunk_overlap=100):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap

        def split_documents(self, documents):
            chunks = []
            for doc in documents:
                text = getattr(doc, 'page_content', '')
                meta = getattr(doc, 'metadata', {}) or {}
                start = 0
                step = max(1, self.chunk_size - self.chunk_overlap)
                while start < len(text):
                    chunk_text = text[start:start + self.chunk_size]
                    # create a simple object with expected attributes
                    chunk = type('Chunk', (), {})()
                    chunk.page_content = chunk_text
                    chunk.metadata = dict(meta)
                    chunks.append(chunk)
                    start += step
            return chunks
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from ..config.db import reports_collection
from typing import List
from fastapi import UploadFile

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rbac-diagnosis-index")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploaded_reports")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.makedirs(UPLOAD_DIR, exist_ok=True)

# initialize pinecone
pc=Pinecone(api_key=PINECONE_API_KEY)
spec=ServerlessSpec(cloud="aws",region=PINECONE_ENV)
existing_indexes=[i["name"] for i in pc.list_indexes()]

if PINECONE_INDEX_NAME not in existing_indexes:
    pc.create_index(name=PINECONE_INDEX_NAME,dimension=768,metric="dotproduct",spec=spec)
    while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
        time.sleep(1)

index=pc.Index(PINECONE_INDEX_NAME)


async def load_vectorstore(uploaded_files:List[UploadFile],uploaded:str,doc_id:str):
    """
        Save files, chunk texts, embed texts, upsert in Pinecone and write metadata to Mongo
    """

    embed_model=GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    for file in uploaded_files:
        filename=Path(file.filename).name
        save_path=Path(UPLOAD_DIR)/ f"{doc_id}_{filename}"
        content=await file.read()
        with open(save_path,"wb") as f:
            f.write(content)

    # load pdf pages
    loader=PyPDFLoader(str(save_path))
    documents=loader.load()
    splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
    chunks= splitter.split_documents(documents)

    texts=[chunk.page_content for chunk in chunks]
    ids=[f"{doc_id}-{i}" for i in range(len(chunks))]
    metadatas=[
            {
                "source": filename,
                "doc_id": doc_id,
                "uploader": uploaded,
                "page": chunk.metadata.get("page", None),
                "text": chunk.page_content[:2000]  # store snippet in metadata (avoid huge fields)
            }
            for chunk in chunks
    ]

    # get embeddings in thread
    embeddings=await asyncio.to_thread(embed_model.embed_documents,texts)
    # upsert - run in thread to avoid blocking
    def upsert():
        index.upsert(vectors=list(zip(ids,embeddings,metadatas)))


    await asyncio.to_thread(upsert)

    # save report  metadata in mongo 
    reports_collection.insert_one({
                "doc_id": doc_id,
                "filename":filename,
                "uploader": uploaded,
                "num_chunks":len(chunks),
                "uploaded_at":time.time()
                
    })