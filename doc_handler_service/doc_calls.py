import os
import glob
import logging
from multiprocessing import Pool
from tqdm import tqdm
import json
from typing import List
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

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

from constants import (
    DOCUMENT_MAP,
    SOURCE_DIRECTORY,
)

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