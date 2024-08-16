import streamlit as st
from Final_Year_project import RAG_Chain

st.title("CampusQuery")

sidebar = st.sidebar
st.sidebar.title("About")
sidebar.write("")
sidebar.write("The CampusQuery is able to answer the questions related to the Dr. Rammanohar Lohia Avadh University, handle a variety of question types, provide accurate and relevant answers, and be easy to use for both technical and non-technical users. This chatbot is completely conversational and have a memory which store the previous chats and answers the query with a relation to the previous queries.")
sidebar.write("")
sidebar.write("**Our Mentor :**")
sidebar.write("Dr. Avadhesh Kumar Dixit")
sidebar.write("")
sidebar.write("**Meet The Developers :**")
sidebar.write("1) Manas Barnwal")
sidebar.write("2) Devesh Mishra")
sidebar.write("3) Priyanka")

# Initialize session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
if prompt := st.chat_input("Hi.. I am rmlu Bot, How can I help you"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Concatenate all user messages into a single query string
    query = " ".join([msg["content"] for msg in st.session_state.messages if msg["role"] == "user"])

    # Get response from RAG_Chain
    response = RAG_Chain(query)  # Assuming response is a string

    # Display the assistant's response
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
