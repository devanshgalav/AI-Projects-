
import os
import tempfile
import streamlit as st

from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.utilities.arxiv import ArxivAPIWrapper

from langchain.tools import Tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate



st.set_page_config(page_title="Agentic RAG", layout="wide")

def build_vectorstore(pdf_file, openai_key):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.read())
        pdf_path = tmp.name

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = FAISS.from_documents(chunks, embeddings)

    return vectordb, chunks
def pdf_rag(question, retriever, llm):
    docs = retriever.get_relevant_documents(question)

    if not docs:
        return None, [], False

    context = "\n\n".join(
        [d.page_content for d in docs[:4]]
    )

    relevance_prompt = f"""
You are a document relevance grader.

Question:
{question}

Context:
{context}

Can the context answer the question?

Reply with ONLY:
YES
or
NO
"""

    grade = llm.invoke(
        relevance_prompt
    ).content.strip().upper()

    if "NO" in grade:
        return None, docs, False

    return context, docs, True

def build_agent(llm, tavily_key):
    os.environ["TAVILY_API_KEY"] = tavily_key

    tavily = TavilySearchResults(max_results=5)

    wiki = WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(top_k_results=3)
    )

    arxiv = ArxivQueryRun(
        api_wrapper=ArxivAPIWrapper(top_k_results=3)
    )

    tools = [
        Tool(
            name="tavily_search",
            func=tavily.run,
            description="Use for current events, recent info, pricing, company updates."
        ),
        Tool(
            name="wikipedia",
            func=wiki.run,
            description="Use for general knowledge and history."
        ),
        Tool(
            name="arxiv",
            func=arxiv.run,
            description="Use for research papers and scientific topics."
        ),
    ]

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         """You are an Agentic RAG assistant.

         Use tools only when necessary.
         Explain which source was used.
         Never hallucinate.
         """),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True
    )

st.title("Single File Agentic RAG")

with st.sidebar:
    st.header("API Keys")

    openrouter_key = st.text_input(
        "OpenRouter API Key",
        type="password"
    )

    tavily_key = st.text_input(
        "Tavily API Key",
        type="password"
    )

uploaded_pdf = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if "vectordb" not in st.session_state:
    st.session_state.vectordb = None

if uploaded_pdf and openrouter_key:
    try:
        with st.spinner("Building vector store..."):
            vectordb, chunks = build_vectorstore(
                uploaded_pdf,
                openrouter_key
            )

            st.session_state.vectordb = vectordb
            st.session_state.chunks = chunks

        st.success("PDF indexed successfully")

    except Exception as e:
        st.error(f"PDF processing failed: {e}")
            

  

question = st.text_input("Ask a question")

if st.button("Submit"):

    if not question:
        st.warning("Enter a question")
        st.stop()

    llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_key,
    model="openai/gpt-4o-mini",
    temperature=0
    )

    if st.session_state.vectordb:

        retriever = st.session_state.vectordb.as_retriever(
            search_kwargs={"k": 4}
        )

        context, docs, strong = pdf_rag(
        question,
        retriever,
        llm
)

        if strong:

            prompt = f"""
            Answer ONLY from this PDF context.

            Context:
            {context}

            Question:
            {question}
            """

            response = llm.invoke(prompt)

            st.subheader("Answer")
            st.write(response.content)

            st.info("Source Used: PDF")

            with st.expander("Retrieved Chunks"):
                for d in docs:
                    st.write(
                        f"Page: {d.metadata.get('page', 'N/A')}"
                    )
                    st.write(d.page_content[:1000])
                    st.divider()

        else:

            if not tavily_key:
                st.error(
                    "Weak PDF context and Tavily API key missing."
                )
                st.stop()

            agent_executor = build_agent(
                llm,
                tavily_key
            )

            result = agent_executor.invoke({
                "input": question
            })

            st.subheader("Answer")
            st.write(result["output"])

            st.info(
                "Source Used: Agent (Tavily/Wikipedia/arXiv)"
            )

    else:

        st.warning(
            "No PDF uploaded. Using agent directly."
        )

        if not tavily_key:
            st.error("Tavily API key required.")
            st.stop()

        agent_executor = build_agent(
            llm,
            tavily_key
        )

        result = agent_executor.invoke({
            "input": question
        })

        st.write(result["output"])
