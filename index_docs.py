import fnmatch
import os
from langchain import FAISS
import llama_index
from llama_index import Document
from langchain.embeddings import HuggingFaceEmbeddings
from typing import Dict, List
from langchain.schema import Document
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import MarkdownTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

# embedding_model = INSTRUCTOR("hkunlp/instructor-large")

persist_dir = "./docs_vector_storage"
model_name = "dolly-v2"
model_id = "databricks/dolly-v2-7b"


def filter_files(directory: str, pattern: str = "*.*") -> List[str]:
    return [
        os.path.join(root, filename)
        for root, _, files in os.walk(directory)
        for filename in files
        if fnmatch.fnmatch(filename, pattern)
    ]


def load_md_docs(
    directory: str, extra_info: Dict[str, str]
) -> List[llama_index.Document]:
    #text_splitter = MarkdownTextSplitter(chunk_size=128, chunk_overlap=16)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=24)
    md_files = filter_files(directory, "*.md")
    all_docs: List[llama_index.Document] = []
    print(f"Found {len(md_files)} markdown files in {directory}")
    for md in md_files:
        print(f"#### Loading {md}")
        md_doc = UnstructuredFileLoader(md, strategy="hi_res").load()
        md_doc_chunks = text_splitter.split_documents(md_doc)
        extra_info.update({"filename": os.path.relpath(md, directory)})
        all_docs.extend(
            [
                llama_index.Document(
                    text=md_doc_chunk.page_content, extra_info=extra_info
                )
                for md_doc_chunk in md_doc_chunks
            ]
        )
    print(f"Loaded {len(md_files)} markdown files into {len(all_docs)} chunks")

    return all_docs


def persist_docs(docs: List[llama_index.Document], index_name: str):
    embed_model_name = "hkunlp/instructor-large"
    print(f"Instantiating embedding model {embed_model_name}")

    # for doc in docs:
    #     sentence = doc.text

    #     # Represent the <domain> <text_type> for <task_objective>:
    #     # * domain is optional, and it specifies the domain of the text, e.g., science, finance, medicine, etc.
    #     # * text_type is required, and it specifies the encoding unit, e.g., sentence, document, paragraph, etc.
    #     # * task_objective is optional, and it specifies the objective of embedding, e.g., retrieve a document, classify the sentence, etc.

    #     instruction = f'Represent the markdown documentation for retrieve a document:'
    #     print(f'#### Encode embedding: {doc.extra_info["filename"]}')
    #     doc.embedding = embedding_model.encode([[instruction,sentence]])

    print(f"Creating storage context")
    db_dir = f"{persist_dir}/faiss/"
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
    langchain_docs = [
        Document(page_content=doc.text, metadata=doc.extra_info) for doc in docs
    ]

    print(f"Adding {len(docs)} documents")

    vectorstore = FAISS.from_documents(langchain_docs, HuggingFaceEmbeddings())
    vectorstore.save_local(folder_path=db_dir, index_name=index_name)


def main():
    docs = load_md_docs(
        "./docs_data/argo-workflows/",
        {
            "source": "Argo Workflows Documentation",
            "source_url": "https://github.com/argoproj/argo-workflows/tree/master/docs",
        },
    )
    persist_docs(docs, index_name="argo_workflow_docs")

    docs = load_md_docs(
        "./docs_data/dbt/",
        {
            "source": "dbt Documentation",
            "source_url": "https://github.com/dbt-labs/docs.getdbt.com/tree/current/website/docs",
        },
    )
    persist_docs(docs, index_name="dbt_docs")


if __name__ == "__main__":
    main()
