import streamlit as st
import logging
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle, os
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed


# -------------------------------
# Setup logging
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logging.info("🚀 Starting F1 RAG Chatbot")

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv()
if "GOOGLE_API_KEY" in st.session_state:
    os.environ["GOOGLE_API_KEY"] = st.session_state["GOOGLE_API_KEY"]
if "LANGSMITH_API_KEY" in st.session_state:
    os.environ["LANGSMITH_API_KEY"] = st.session_state["LANGSMITH_API_KEY"]
os.environ["LANGSMITH_TRACING"] = "true"

st.sidebar.header("🔑 API Key Setup")

if "api_keys_submitted" not in st.session_state:
    st.session_state.api_keys_submitted = False

if not st.session_state.api_keys_submitted:
    with st.sidebar.form("api_keys_form"):
        google_api_key = st.text_input("Google API Key", type="password")
        langchain_api_key = st.text_input("LangChain API Key", type="password")
        submitted = st.form_submit_button("Save API Keys")
        
        if submitted:
            if google_api_key:
                st.session_state["GOOGLE_API_KEY"] = google_api_key
                os.environ["GOOGLE_API_KEY"] = google_api_key
            if langchain_api_key:
                st.session_state["LANGSMITH_API_KEY"] = langchain_api_key
                os.environ["LANGSMITH_API_KEY"] = langchain_api_key
            
            st.session_state.api_keys_submitted = True
            st.success("✅ API Keys saved for this session")

if "GOOGLE_API_KEY" not in st.session_state or "LANGSMITH_API_KEY" not in st.session_state:
    st.warning("Please enter your API keys in the sidebar to use the chatbot.")
    st.stop()

# -------------------------------
# Initialize session state
# -------------------------------
if "added_pages" not in st.session_state:
    st.session_state.added_pages = set()
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# -------------------------------
# 1. Load Wikipedia cache or fetch
# -------------------------------

if "docs" not in st.session_state:
    st.session_state.docs = []
    if os.path.exists("wiki_cache.pkl"):
        logging.info("📂 Loading cached Wikipedia docs...")
        with open("wiki_cache.pkl", "rb") as f:
            st.session_state.docs = pickle.load(f)
    else:
        logging.info("🌐 Fetching Wikipedia pages...")
        pages = [
            # Core F1 topics
            "Formula One",
            "List of Formula One Grands Prix",
            "Formula One tyres",
            "Formula One car",
            "Formula One engines",
            "2013 Formula One World Championship",
            "Formula One World Championship",
            "History of Formula One",
            "List of Formula One tyre records",
            "List of Formula One polesitters",
            "Formula 1: Drive to Survive",
            "List of Formula One drivers",
            "List of Formula One circuits",
            "Formula_One_regulations",
            "2026 Formula One World Championship",
            "Cadillac in Formula One",
            "2025 Formula One World Championship",
            "2024 Formula One World Championship",
            "2023 Formula One World Championship",
            "2022 Formula One World Championship",
            "2021 Formula One World Championship",
            "2020 Formula One World Championship",
            "List of Formula One driver records",
            "List of Formula One constructor records",
            "List of Formula One engine records",
            "List of Formula One race records",
            "List of Formula One World Drivers' Champions",
            "Formula One World Champion",
            "List of Formula One seasons",
            "List of female Formula One drivers",
            "Pole position",
            "Audi in Formula One",
            "List of Formula One engine manufacturers",
            "F1 (film)",
            "List of Formula One driver numbers",
            "Formula One race weekend",

            # Drivers
            "Max Verstappen",
            "Lewis Hamilton",
            "Michael Schumacher",
            "Ayrton Senna",
            "Sebastian Vettel",
            "Fernando Alonso",
            "Kimi Räikkönen",

            # Teams / Constructors
            "Mercedes in Formula One",
            "Ferrari in Formula One",
            "McLaren in Formula One",
            "Red Bull Racing",
            "Williams in Formula One",
            "Aston Martin in Formula One",
            "Alpine in Formula One",
            "Alfa Romeo Racing",
            "Haas F1 Team",
            "AlphaTauri in Formula One",
            "Renault in Formula One",
            "Toro Rosso in Formula One",
            "Sauber in Formula One",
            "Lotus in Formula One",
            "Brabham in Formula One",
            "Tyrrell Racing",
            "Shadow Racing Cars",
            "Minardi in Formula One",
            "Jordan Grand Prix",
            "BMW Sauber",
            "Honda Racing F1 Team",
            "Toyota Racing",
            "Super Aguri F1 Team",
            "Caterham F1 Team",
            "Marussia F1 Team",
            "HRT F1 Team",
            "Caterham F1 Team",

            # Records
            "Youngest Formula One drivers",
            "Fastest laps in Formula One",
            "Longest winning streaks in Formula One",
            "Most career wins in Formula One",
            "Formula One driver fatalities",

            # Circuits
            "Circuit de Monaco",
            "Silverstone Circuit",
            "Monza Circuit",
            "Spa-Francorchamps",
            "Suzuka Circuit",

            # Technology
            "Formula One suspension",
            "Formula One aerodynamics",
            "Hybrid power units in Formula One",

            # Tyres & Weather
            "Pirelli in Formula One",
            "Formula One tyre strategies",
            "Wet-weather races in Formula One",

            # Historical seasons
            "2010 Formula One World Championship",
            "2011 Formula One World Championship",
            "2012 Formula One World Championship",
            "2014 Formula One World Championship",
            "2015 Formula One World Championship",
            "2016 Formula One World Championship",
            "2017 Formula One World Championship",
            "2018 Formula One World Championship",
            "2019 Formula One World Championship",

            # Miscellaneous / Cultural
            "Formula One controversies",
            "F1 safety car",
            "Formula One fan culture",
            "Formula One media coverage",

            #Rivalries
            "List of Formula One rivalries",  # General page if available
            "Senna–Prost rivalry",
            "Hamilton–Rosberg rivalry",
            "Vettel–Alonso rivalry",
            "Verstappen–Hamilton rivalry",
            "Schumacher–Hakkinen rivalry",
            "Norris–Piastri rivalry",
            "Verstappen–Norris rivalry"
        ]
        with st.spinner("⏳ Fetching Wikipedia pages asynchronously, please wait..."):
            def fetch_wiki_page(page_name):
                try:
                    loader = WikipediaLoader(query=page_name, lang="en", load_max_docs=1)
                    return loader.load()
                except Exception as e:
                    logging.warning(f"❌ Failed to load page {page_name}: {e}")
                    return []

            # Use ThreadPoolExecutor to fetch pages concurrently
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {executor.submit(fetch_wiki_page, page): page for page in pages}
                for future in as_completed(futures):
                    st.session_state.docs.extend(future.result())

            # Save cache after all pages loaded
            with open("wiki_cache.pkl", "wb") as f:
                pickle.dump(st.session_state.docs, f)


    

# -------------------------------
# Initialize text splitter and embeddings
# -------------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# -------------------------------
# Load or create FAISS index
# -------------------------------
if "vector_store" not in st.session_state or st.session_state.vector_store is None:
    if os.path.exists("faiss_index"):
        logging.info("📂 Loading FAISS index from disk...")
        st.session_state.vector_store = FAISS.load_local(
            "faiss_index", embeddings, allow_dangerous_deserialization=True
        )
    else:
        logging.info("⚡ Creating FAISS index from docs...")
        split_docs = splitter.split_documents(st.session_state.docs)
        st.session_state.vector_store = FAISS.from_documents(split_docs, embeddings)
        st.session_state.vector_store.save_local("faiss_index")
        logging.info("✅ FAISS index created and saved.")


# -------------------------------
# 2. Add new Wikipedia page
# -------------------------------
new_link = st.text_input(
    "🔗 Add a Wikipedia link for F1 "
    "(e.g. https://en.wikipedia.org/wiki/2023_Formula_One_World_Championship)"
)

if st.button("Add Page") and new_link:
    if new_link not in st.session_state.added_pages:
        page_title = new_link.split("/")[-1]
        st.write(f"📄 Adding page: **{page_title}**")
        logging.info(f"➕ Adding user-specified page: {page_title}")

        # Load new page
        loader = WikipediaLoader(query=page_title, lang="en", load_max_docs=1)
        new_docs = loader.load()
        st.session_state.docs.extend(new_docs)

        # Split documents including new page
        # Split only the new documents
        split_new_docs = splitter.split_documents(new_docs)

        # Add new docs to existing vector store
        st.session_state.vector_store.add_documents(split_new_docs)

        # Save updated index
        st.session_state.vector_store.save_local("faiss_index")
        logging.info("✅ FAISS index rebuilt with new page")

        # Save updated cache
        with open("wiki_cache.pkl", "wb") as f:
            pickle.dump(st.session_state.docs, f)
        logging.info("✅ Updated wiki_cache.pkl with new page")

        # Remember this page has been added
        st.session_state.added_pages.add(new_link)



vector_store = st.session_state.vector_store

# -------------------------------
# 4. Setup retriever + LLM
# -------------------------------
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 20})
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

workflow = StateGraph(state_schema=MessagesState)

def call_model(state: MessagesState):
    logging.info("🤖 Entered call_model()")
    
    system_prompt = (
        "You are a factual assistant for Formula 1. "
        "Use the provided context to answer as accurately as possible. "
        "If the context does not fully answer, provide your best factual answer and indicate it may be incomplete."
    )

    # Current user query
    query = state["messages"][-1].content
    logging.info(f"❓ User Query: {query}")

    # Retrieve relevant documents from FAISS
    retrieved_docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    logging.info(f"📑 Retrieved {len(retrieved_docs)} docs, context size: {len(context)} chars")

    # Include only previous user messages as memory
    messages_history = [
        HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"])
        for msg in st.session_state.messages
    ]

    # Build full messages list
    messages = [
        SystemMessage(content=system_prompt),
        SystemMessage(content=f"Here are some reference documents:\n{context}")
    ] + messages_history + [
        HumanMessage(content=query)
    ]

    logging.info("⚡ Sending query to LLM...")
    response = llm.invoke(messages)
    logging.info("✅ LLM responded")

    return {"messages": [response]}

workflow.add_node("model", call_model)
workflow.add_edge(START, "model")
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# -------------------------------
# 5. Streamlit Chatbot UI
# -------------------------------
st.set_page_config(page_title="🏎️ F1 RAG Chatbot", page_icon="🏎️", layout="wide")
st.title("🏎️ Formula 1 Chatbot (RAG + Memory)")

# Setup session history
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "f1_chat"
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about Formula 1..."):
    logging.info(f"📝 User asked: {prompt}")
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response = app.invoke(
        {"messages": [HumanMessage(content=prompt)]},
        config={"configurable": {"thread_id": st.session_state.thread_id}}
    )
    answer = response["messages"][-1].content
    logging.info(f"🤖 Assistant Answer: {answer[:100]}...")

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
