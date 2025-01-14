import streamlit as st
import logging
import json
from datetime import datetime

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain.tools.retriever import create_retriever_tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver

# -------------------------
# Logging Configuration
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[logging.FileHandler('chat_history.log')]
)
logger = logging.getLogger(__name__)

# -------------------------
# LLM and Database Setup
# -------------------------
embedding_llm = OllamaEmbeddings(model="llama3.2")
db = Chroma(
    persist_directory="C:\\Users\\porka\\Desktop\\CHATBOT\\support-chatbot\\app\\chroma",
    embedding_function=embedding_llm,
    collection_name="vermac-support"
)

retriever_tool = create_retriever_tool(
    db.as_retriever(search_type='mmr'),
    name="vermac_search",
    description="""Search for information about Vermac store, 
    including store information, coffee drink menus, specialty coffee beans menu, 
    and bean fact sheet."""
)

config = {"configurable": {"thread_id": "thread1"}}

llm = ChatOllama(model="llama3.2", temperature=0)
tools = [retriever_tool]

# -------------------------
# System Message
# -------------------------
SYSTEM_MESSAGE = """You are a helpful receptionist at Vermac. Your name is Keven.
You will answer politely and professionaly."""

# -------------------------
# Memory (Checkpoint) Setup
# -------------------------
memory = MemorySaver()

# -------------------------
# Create the Agent
# -------------------------
langgraph_agent_executor = create_react_agent(
    llm,
    tools,
    state_modifier=SYSTEM_MESSAGE,
    checkpointer=memory
)

# -------------------------
# Helper Functions
# -------------------------
def log_messages(response):
    """Log messages from the response to a file."""
    for message in response['messages']:
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'role': message.type,
            'content': message.content,
            'additional_kwargs': message.additional_kwargs
        }
        logger.info(json.dumps(log_entry))

def chat(user_input):
    """Invoke the agent with the user input and log the response."""
    response = langgraph_agent_executor.invoke(
        {"messages": [("human", user_input)]},
        config
    )
    log_messages(response)
    return response

# -------------------------
# Streamlit App
# -------------------------
def main():
    st.set_page_config(page_title="Keven, the Vermac Receptionist", layout="centered")

    st.title("Keven, the Vermac Receptionist")
    st.write("Welcome to Vermac!")
    st.write("Feel free to ask about our products.")

    # Initialize session state to store messages (user + assistant)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display existing messages in a chat-like UI
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # This creates the input box at the bottom of the page
    user_input = st.chat_input("Type your question or greeting here...")
    if user_input:
        # Store user message
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Display user message immediately in UI
        with st.chat_message("user"):
            st.write(user_input)

        # Here you would call your LLM or agent with the user_input
        assistant_reply = chat(user_input)['messages'][-1].content
        
        # Store and display assistant message
        st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
        with st.chat_message("assistant"):
            st.write(assistant_reply)

if __name__ == "__main__":
    main()
