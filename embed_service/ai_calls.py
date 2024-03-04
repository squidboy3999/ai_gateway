import os
import glob
import logging
from multiprocessing import Pool
from tqdm import tqdm
import json
from typing import List
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.prompts import PromptTemplate
from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
# Summarizer we'll use for Map Reduce
from langchain.chains.summarize import load_summarize_chain
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline

from load_models import (
    load_quantized_model_gguf_ggml,
    load_quantized_model_qptq,
    load_full_model,
)

from constants import (
    CHROMA_SETTINGS,
    DOCUMENT_MAP,
    EMBEDDING_MODEL_NAME,
    PERSIST_DIRECTORY,
    MODEL_ID,
    MODEL_BASENAME,
    MAX_NEW_TOKENS,
    MODELS_PATH,
    CHROMA_SETTINGS,
    INGEST_THREADS,
    PERSIST_DIRECTORY,
    SOURCE_DIRECTORY,
    NUMBER_OF_CLUSTERS,
    CLUSTER_SAMPLES,
    QUESTION_DIRECTORY,
)
# Data Science
import numpy as np
from sklearn.cluster import KMeans

# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    # ".docx": (Docx2txtLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    #".eml": (MyElmLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PDFMinerLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    # Add more mappings for other file extensions and loaders as needed
}


def load_single_document(file_path: str) -> Document:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        try:
            return loader.load()[0]
        except Exception as e:
            logging.warning(f"Failed to load {file_path} - Exception: {e}")


    raise ValueError(f"Unsupported file extension '{ext}'")


def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    """
    Loads all documents from the source documents directory, ignoring specified files
    """
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]

    with Pool(processes=os.cpu_count()) as pool:
        results = []
        with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
            for i, doc in enumerate(pool.imap_unordered(load_single_document, filtered_files)):
                results.append(doc)
                pbar.update()

    return results

def process_documents(data, ignored_files: List[str] = []) -> List[Document]:
    """
    Load documents and split in chunks
    """
    print(f"Loading documents from {data['path']}")
    documents = load_documents(f"{SOURCE_DIRECTORY}/{data['path']}", ignored_files)
    if not documents:
        print("No new documents to load")
        return [],[]
    print(f"Loaded {len(documents)} new documents from {SOURCE_DIRECTORY}/{data['path']}")
    # "small_chunk_size","small_chunk_overlap","large_chunk_size","large_chunk_overlap"
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=data["small_chunk_size"], chunk_overlap=data["small_chunk_overlap"])
    texts = text_splitter.split_documents(documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=data["large_chunk_size"], chunk_overlap=data["large_chunk_overlap"])
    lg_texts = text_splitter.split_documents(documents)
    return texts, [[lg_text] for lg_text in lg_texts]

def docs_to_strings(text_docs):
    context=""
    metadata_string=""
    for i, doc in enumerate(text_docs):
        doc_pieces=f"{doc}".split("metadata=")
        metadata_string+=f"{i}: {doc_pieces[1]}\n"
        context+=doc_pieces[0].strip().replace("page_content=","")+"\n"
    return {"context":context,"metadata":metadata_string}


# gets a prompt from client and returns response
def call_llm(llm,prompt):
    return llm(prompt)

# gets a vector_db name, template and list of questions and returns a list of responses
def get_rag_qa_list(llm,db,questions,template):
    # TODO - make this an arg that comes in 
    retriever = db.as_retriever(search_kwargs={'k':4},return_source_documents=True)
    qa_list=[]
    for question in questions:  
        r_docs=retriever.get_relevant_documents(question)
        # out_dir=os.environ.get('OUTPUT_DIR','/var/log/ai_gate/')
        # with open(os.path.join(out_dir,f"ai_rag.txt"), "a") as _file:
        #     for r_doc in r_docs:
        #         _file.write(r_doc)
        context=""
        metadata_string=""
        for i, doc in enumerate(r_docs):
            doc_pieces=f"{doc}".split("metadata=")
            metadata_string+=f"{i}: {doc_pieces[1]}\n"
            context+=doc_pieces[0].strip().replace("page_content=","")+"\n"
        resp=call_llm(llm,template.replace("|question|",question).replace("|context|",context))
        qa_list.append({"response":resp,"context":context,"metadata":metadata_string})
    return qa_list

# creates a list of list of lists of documents that are clusted. Texts should already exist server side but client may choose by name
# client can pick number of clusters and cluster sizes, server may cache clustered collections - memoize calls.
def get_cluster_docs(embed,inputs):
    num_clusters=inputs['num_clusters']
    cluster_samples=inputs['cluster_samples']
    texts=inputs["texts"]
    vectors = embed.embed_documents([x.page_content for x in texts])

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(vectors)
    # not sure what this does
    kmeans.labels_
    closest_indices_per_cluster=[]
    # Loop through the number of clusters you have
    for i in range(num_clusters):
        closest_indices = []
        # Get the list of distances from that particular cluster center
        distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1)
        # Find the indices of the eight closest points
        closest_indices = np.argsort(distances)[:cluster_samples]
        # Append the list of indices to the main list
        closest_indices_per_cluster.append(closest_indices)

    # Flatten the list of lists and sort it to get a list of unique indices
    selected_indices = sorted(set(np.concatenate(closest_indices_per_cluster)))
    selected_docs = [texts[doc] for doc in selected_indices]
    # To organize selected_docs as a list of lists for each cluster
    organized_docs = [[texts[doc] for doc in cluster_indices] for cluster_indices in closest_indices_per_cluster]
    return organized_docs

# done when server boots up
def get_doc_vectordb(per_dir,texts,embeddings):
    try:
        state="db found"
        if not os.path.exists(os.path.join(f"{PERSIST_DIRECTORY}/{per_dir}",'chroma.sqlite3')):
            state="db created"
            Chroma.from_documents(
                texts,
                embeddings,
                persist_directory=f"{PERSIST_DIRECTORY}/{per_dir}",
                client_settings=CHROMA_SETTINGS,
            )
        db = Chroma(
            persist_directory=f"{PERSIST_DIRECTORY}/{per_dir}",
            embedding_function=embeddings,
            client_settings=CHROMA_SETTINGS
        )
        return db, state
    except Exception as e:
        return None, f"db retrieval Error - {e}"


def get_llm(model_id, model_basename, device, LOGGING):
    return load_quantized_model_gguf_ggml(model_id, model_basename, device, LOGGING)

def get_embeddings(e_model_name,device,_cache_folder):
    print(f"embeddings model name - {e_model_name}")
    print(f"device - {device}")
    print(f"cache_folder - {_cache_folder}")
    return HuggingFaceInstructEmbeddings(model_name=e_model_name, model_kwargs={"device": device}, cache_folder=_cache_folder)