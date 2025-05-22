import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler

# Streamlit setup
st.set_page_config(page_title="Text to Math Solver & Search Assistant", page_icon="🧮")
st.title("🧮 Math Problem Solver using Google Gemma 2")

# API Key Input
groq_api_key = st.sidebar.text_input("Enter your Groq API Key", type="password")
if not groq_api_key:
    st.info("Please provide your Groq API Key to continue.")
    st.stop()

# Initialize Groq LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_api_key)

# Wikipedia Tool
wikipedia_tool = Tool(
    name="Wikipedia Search",
    func=WikipediaAPIWrapper().run,
    description="Useful for looking up general knowledge and topics."
)

# Math Expression Calculator Tool
calculator_tool = Tool(
    name="Calculator",
    func=LLMMathChain.from_llm(llm=llm).run,
    description="Use this tool for simple math expressions like 2 + 2 * 5."
)

# Reasoning Tool for Word Problems
reasoning_prompt = PromptTemplate(
    input_variables=["question"],
    template="""
You are a math reasoning expert. Solve the following word problem step by step and give the final answer.
Question: {question}
Answer:
"""
)

reasoning_chain = LLMChain(llm=llm, prompt=reasoning_prompt)
reasoning_tool = Tool(
    name="Math Reasoning Tool",
    func=reasoning_chain.run,
    description="Use this for solving word problems or logic-based math questions."
)

# Initialize agent with all tools
agent = initialize_agent(
    tools=[wikipedia_tool, calculator_tool, reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True,
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I’m your Math & Knowledge Assistant. Ask me anything."}
    ]

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# User input
question = st.text_area("Enter your question:")

if st.button("Find My Answer"):
    if question:
        with st.spinner("Thinking..."):
            st.session_state.messages.append({"role": "user", "content": question})
            st.chat_message("user").write(question)

            cb_handler = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            try:
                response = agent.run(question, callbacks=[cb_handler])
            except Exception as e:
                response = f"⚠️ Error: {str(e)}"

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)
    else:
        st.warning("Please enter a question.")
