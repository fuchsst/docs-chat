import fnmatch
import os
from pathlib import Path
import shutil
import subprocess
from langchain import FAISS
import llama_index
from llama_index import Document
from langchain.embeddings import HuggingFaceEmbeddings
from typing import Dict, List
from langchain.schema import Document
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import MarkdownTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
import yaml

# embedding_model = INSTRUCTOR("hkunlp/instructor-large")

persist_dir = "./docs_vector_storage"
model_name = "dolly-v2"
model_id = "databricks/dolly-v2-7b"

def load_repo_configs(config_filename: str):
    with open(config_filename, 'r') as file:
        return yaml.safe_load(file)

def git_checkout(repo: str, branch: str, directories: List[str], target_directory:str):
    def on_readonly_file_error_handler(func, path, exc_info):
        """
        Error handler for ``shutil.rmtree``.

        If the error is due to an access error (read only file)
        it attempts to add write permission and then retries.

        If the error is for another reason it re-raises the error.
        
        Usage : ``shutil.rmtree(path, onerror=onerror)``
        """
        import stat
        # Is the error an access error?
        if not os.access(path, os.W_OK):
            os.chmod(path, stat.S_IWUSR)
            func(path)
        else:
            raise


    print(f'Checkout {repo}/{branch} to {target_directory}')
    try:
        # Check if the target directory exists, and delete it if it does
        if os.path.exists(target_directory):
            print(f"{target_directory} already present, deleting!")
            shutil.rmtree(target_directory, onerror=on_readonly_file_error_handler)

        os.mkdir(target_directory)
        os.chdir(target_directory)
        subprocess.run(['git', 'init'])
        subprocess.run(['git', 'remote', 'add', 'origin', repo])
        subprocess.run(['git', 'config', 'core.sparsecheckout', 'true'])
        with open('.git/info/sparse-checkout', 'w') as fp:
            fp.writelines(directories)
        subprocess.run(['git', 'pull', 'origin', branch])
        os.chdir('..')
        
        print(f"Successfully cloned the '{branch}' branch from {repo} to '{target_directory}'.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
        if os.path.exists(target_directory):
            shutil.rmtree(target_directory)


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
    text_splitter = RecursiveCharacterTextSplitter(separators=['\n#', '```', '---', '  \n', '\n\n\n'], chunk_size=256, chunk_overlap=24)
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

    print(f"Adding {len(docs)} documents to FAISS vectorstore")

    vectorstore = FAISS.from_documents(langchain_docs, HuggingFaceEmbeddings())
    vectorstore.save_local(folder_path=db_dir, index_name=index_name)


def main():
    repo_configs = load_repo_configs('./doc_source_repos.yml')['git-repos']
    print(f'Fetching markdown files from {repo_configs.keys()}')
    all_docs = []
    base_path= os.getcwd()
    for repo_key in repo_configs:
        print(f'# Adding documentation from {repo_key}')
        repo_conf=repo_configs[repo_key]
        target_directory=os.path.join('docs_data', repo_key)
        directories=repo_conf['directories']
        git_checkout(repo=repo_conf['repo'], branch=repo_conf['branch'], directories=directories, target_directory=target_directory)
        os.chdir(base_path)
        for dir in directories:
            print(f'## Loading markdown files from {target_directory}{dir}')
            docs = load_md_docs(
                                target_directory+dir,
                                {
                                    "source": repo_conf['description'],
                                    "source_url": repo_conf['url']+dir,
                                }
                            )
            print(f'## Got {len(docs)} document chunks')
            all_docs.extend(docs)
    persist_docs(all_docs, index_name="docs")
    print("DONE")


if __name__ == "__main__":
    main()
