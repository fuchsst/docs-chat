import time
from langchain import FAISS, HuggingFacePipeline, PromptTemplate
import torch
from llama_index.indices.query.base import BaseQueryEngine
from transformers import pipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.tools import BaseTool
from langchain.llms import OpenAI
from langchain import LLMMathChain, SerpAPIWrapper
from langchain.agents.agent_toolkits import (
    create_vectorstore_router_agent,
    VectorStoreRouterToolkit,
    VectorStoreInfo,
)


# set context window size
context_window = 2048

# model_id = "OpenAssistant/galactica-6.7b-finetuned"
model_id = "databricks/dolly-v2-3b"
model_name = "dolly-v2"
persist_dir = "./docs_vector_storage"
db_dir = f"{persist_dir}/faiss/"
device = "cuda:0"
# device='cpu'


def get_query_retreiver(index_name: str) -> BaseQueryEngine:
    print(f"Loading Vector Store {index_name}")
    faiss = FAISS.load_local(
        folder_path=db_dir, index_name=index_name, embeddings=HuggingFaceEmbeddings()
    )

    retreiver = faiss.as_retriever(search_kwargs={"k": 3})
    print(f"Loaded {index_name}")
    return retreiver


def get_model():
    print(f"Loading LLM {model_id}")
    return HuggingFacePipeline(
        pipeline=pipeline(
            model=model_id,
            torch_dtype=torch.bfloat16,
            device=device,
            return_full_text=True,
            trust_remote_code=True,
            model_kwargs={"temperature": 0.3, "max_length": 256},
        ),
        verbose=True,
    )


def main():
    argo_retreiver = get_query_retreiver("argo_workflow_docs")
    dbt_retreiver = get_query_retreiver("dbt_docs")

    llm = get_model()

    question_prompt = PromptTemplate(
        template="""Use the following part of a document to see if any of the text is relevant to answer the question: 
                    {context}
                    =========
                    Question: 
                    {question}
                    =========
                    Return any relevant text verbatim, if any:""",
        input_variables=["context", "question"],
    )

    combine_prompt = PromptTemplate(
        template="""Given the following extracted parts of a long document and a question, create a final answer. 
                    If you don't know the answer, just say that you don't know. Don't try to make up an answer.
                    QUESTION: 
                    {question}
                    =========
                    {summaries}
                    =========
                    FINAL ANSWER:""",
        input_variables=["summaries", "question"],
    )

    # qa = RetrievalQA.from_llm(
    #     llm=llm, retriever=retreiver, prompt=prompt, return_source_documents=True
    # )
    # chain = load_summarize_chain(llm=llm, chain_type="map_reduce", verbose=True)

    argo_chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="map_reduce",
        retriever=argo_retreiver,
        return_source_documents=True,
        chain_type_kwargs={
            "question_prompt": question_prompt,
            "combine_prompt": combine_prompt,
        },
    )

    dbt_chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="map_reduce",
        retriever=dbt_retreiver,
        return_source_documents=True,
        chain_type_kwargs={
            "question_prompt": question_prompt,
            "combine_prompt": combine_prompt,
        },
    )

    tools = [
        Tool(
            name="Argo Workflows Docs",
            func=argo_chain.run,
            description="Useful for when you need to answer questions about Argo Workflows. Input should be a fully formed question.",
        ),
        Tool(
            name="dbt Docs",
            func=dbt_chain.run,
            description="Useful for when you need to answer questions about dbt. Input should be a fully formed question.",
        ),
    ]

    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )

    dbt_vectorstore_info = VectorStoreInfo(
        name="dbt",
        description="dbt Documentation that describes how to define SQL pipelines using sources, models, tests and exposures",
        vectorstore=FAISS.load_local(
            folder_path=db_dir,
            index_name="dbt_docs",
            embeddings=HuggingFaceEmbeddings(),
        ),
    )
    argo_vectorstore_info = VectorStoreInfo(
        name="argo",
        description="Argo Workflows Documentation that describes how to orchestrate owrkflow pipelines",
        vectorstore=FAISS.load_local(
            folder_path=db_dir,
            index_name="argo_workflow_docs",
            embeddings=HuggingFaceEmbeddings(),
        ),
    )
    router_toolkit = VectorStoreRouterToolkit(
        vectorstores=[dbt_vectorstore_info, argo_vectorstore_info], llm=llm
    )
    agent_executor = create_vectorstore_router_agent(
        llm=llm, toolkit=router_toolkit, verbose=True
    )

    question = ""
    while True and question not in ['exit', 'quit']:
        question = input("What is your question?: ")
        start = time.time()
        # answer = agent({"question": question}, return_only_outputs=True)
        answer = dbt_chain({"question": question}, return_only_outputs=True)
        # answer = agent_executor.run(question)
        end = time.time()
        print(f'Answer (received in {round(end-start)}s): \n{answer["answer"]}')
        print(f'Sources:')
        for doc in answer["source_documents"]:
            print(doc)


if __name__ == "__main__":
    main()
