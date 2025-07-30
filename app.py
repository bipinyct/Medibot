import os
import streamlit as st

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceEndpoint
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()  # ✅ Load .env vars into os.environ


# Set FAISS vectorstore path
DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def load_llm(huggingface_repo_id, HF_TOKEN):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={
            "token": HF_TOKEN,
            "max_length": "512"
        }
    )
    return llm

def format_response(result, source_documents):
    formatted = f"**🧠 Answer:**\n{result.strip()}\n\n"

    if source_documents:
        formatted += "**📚 Source(s):**\n"
        for i, doc in enumerate(source_documents, start=1):
            content = doc.page_content.strip().replace("\n", " ")
            snippet = content[:300] + ("..." if len(content) > 300 else "")
            formatted += f"{i}. {snippet}\n\n"
    else:
        formatted += "_No source documents found._"

    return formatted

def main():
    st.title("🤖 MediBot – Your AI Medical Assistant")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("📝 Ask a medical question...")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        # 👉 Basic greeting detection
        greetings = [
            "hi", "hello", "hey", "yo", "what's up", "sup", "namaste",
            "good morning", "good evening", "good afternoon"
        ]

        # If prompt is casual greeting → friendly reply
        if prompt.strip().lower() in greetings:
            response_text = "👋 Hey there! I'm MediBot — your AI health assistant. Ask me any medical-related question!"
            st.chat_message('assistant').markdown(response_text)
            st.session_state.messages.append({'role': 'assistant', 'content': response_text})
            return

        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer user's question.
        If you don't know the answer, just say you don't know. Do NOT make up anything. 
        Do not include anything outside the provided context.

        Context: {context}
        Question: {question}

        Start your answer directly without any small talk.
        """

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("❌ Failed to load the vector store.")
                return

            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatGroq(
                    model_name="meta-llama/llama-4-maverick-17b-128e-instruct",  # Free + fast via Groq
                    temperature=0.0,
                    groq_api_key=os.environ["GROQ_API_KEY"],
                ),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response = qa_chain.invoke({'query': prompt})

            result = response["result"]
            source_documents = response["source_documents"]

            formatted_response = format_response(result, source_documents)

            st.chat_message('assistant').markdown(formatted_response)
            st.session_state.messages.append({'role': 'assistant', 'content': formatted_response})

        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")

if __name__ == "__main__":
    main()
