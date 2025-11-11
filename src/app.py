"""
CIS Benchmarks RAG Application

A Streamlit-based RAG (Retrieval-Augmented Generation) application for querying
CIS Benchmarks using AWS S3 Vectors and OpenAI GPT models.
"""

import os
from hashlib import sha256
from typing import Dict, Any

import streamlit as st
import boto3
from dotenv import load_dotenv

from openai_services import OpenAIEmbeddingService, OpenAIGenerationService
from rag_pipeline import (
    RetrievalService, PromptAugmenter, QueryProcessor,
    ProcessorConfig, RetrievalConfig, CosineSimilarity
)
from log_time import ProcessTimer
from helpers import load_config


def setup_page_config() -> None:
    """Configure Streamlit page settings and branding."""
    st.set_page_config(
        page_title="CIS Benchmarks Retrieval", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Set up branding
    jd_logo = "images/jd-logo.png"
    logo_white = "images/logo-white.png"
    st.logo(jd_logo, size="large", link="https://jdfortress.com", icon_image=logo_white)
    st.title("CIS Benchmarks RAG (Demo)")


def load_environment_config() -> Dict[str, str]:
    """Load and return environment configuration."""
    load_dotenv()
    
    return {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "bucket": load_config("AWS_S3_BUCKET"),
        "vector_bucket": load_config("VECTOR_S3_BUCKET"),
        "vector_region": load_config("VECTOR_REGION"),
        "vector_index": load_config("VECTOR_INDEX"),
        "vector_dim": load_config("VECTOR_DIMENSION"),
        "demo_username": os.getenv("DEMO_USERNAME"),
        "demo_password": os.getenv("DEMO_PASSWORD"),
    }


def check_authentication(username: str, password: str) -> None:
    """Handle user authentication."""
    def password_entered():
        entered_combo = st.session_state["username"] + st.session_state["password"]
        expected_combo = username + password
        
        if sha256(entered_combo.encode()).hexdigest() == sha256(expected_combo.encode()).hexdigest():
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Username", value="", key="username")
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.stop()
    elif not st.session_state["password_correct"]:
        st.error("❌ Incorrect password")
        st.stop()


def initialize_services(config: Dict[str, str]) -> Dict[str, Any]:
    """Initialize and return all required services."""
    with st.spinner("Initializing LLM..."):
        session = boto3.Session(region_name=config["vector_region"])
        s3_client = session.client("s3")
        vector_svc = session.client("s3vectors")
        
        embedding_service = OpenAIEmbeddingService(
            config["api_key"], 
            load_config('embedding_model')
        )
        generation_service = OpenAIGenerationService(
            config["api_key"], 
            load_config('inference_model')
        )
        
        retrieval_service = RetrievalService(
            vector_svc, 
            s3_client,
            load_config('reranker_model')
        )
        augmenter = PromptAugmenter('rag_prompt.md')

        services = {
            "aws_session": session,
            "s3_client": s3_client,
            "vector_client": vector_svc,
            "embedding_service": embedding_service,
            "generation_service": generation_service,
            "retrieval_service": retrieval_service,
            "augmenter": augmenter,
        }
        
        st.success(f"LLM initialized: {generation_service.model}", icon="✅")
        return services


def setup_sidebar() -> tuple[int, float]:
    """Setup sidebar controls and return configuration values."""
    # Question suggestions section
    with st.sidebar:
        st.header("💡 Query Suggestions")
        st.markdown("*Click any question/query and give it a try:*")
        
        # Define question categories and questions
        questions = {
            "🖥️ System Security": [
                "How can I verify that DHCP servers aren't installed on my systems and what should I do if they are?",
                "I want to make sure my machines aren't acting as DHCP servers to reduce attack surface."
            ],
            "☁️ Microsoft 365": [
                "What's the recommended number of global administrators in Microsoft 365, and how do I audit it?",
                "I need to ensure that I have the right number of global administrators configured in my Microsoft 365 tenant."
            ],
            "📁 AWS Storage": [
                "How do I implement and validate the use of Amazon EFS to improve my storage deployment process?",
                "I want to simplify file system management by using Amazon's managed EFS solution."
            ],
            "🗄️ Oracle Cloud (OCI)": [
                "How do I check whether object storage versioning is enabled in OCI, and why is it important?",
                "I want to ensure that my object storage buckets in Oracle Cloud have versioning enabled.",
                "What steps can I take to confirm that no object storage buckets in my OCI environment are publicly accessible?",
                "I'm trying to make sure all my Oracle Cloud buckets are private and not publicly visible."
            ]
        }
        
        # Display clickable questions by category
        for category, question_list in questions.items():
            st.markdown(f"**{category}:**")
            for question in question_list:
                # Create a shorter button label for display
                button_label = question[:60] + "..." if len(question) > 60 else question
                if st.button(button_label, key=f"btn_{hash(question)}", use_container_width=True):
                    st.session_state.selected_question = question
                    st.rerun()
            st.markdown("")
    
    # Retrieval settings section
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 Retrieval Settings")
    
    top_k = st.sidebar.slider("Top K Chunks", 1, 10, 3)
    similarity_threshold = st.sidebar.slider("Similarity Threshold", 0.0, 1.0, 0.32)
    
    return top_k, similarity_threshold


def process_user_query(user_input: str, top_k: int, similarity_threshold: float) -> None:
    """Process user query and update chat history."""
    pt = ProcessTimer()
    
    with st.spinner("Generating response..."):
        current_config = ProcessorConfig(
            retrieval=RetrievalConfig(
                top_k=top_k,
                similarity_threshold=similarity_threshold
            )
        )
        
        processor = QueryProcessor(
            embedding_service=st.session_state.base_services["embedding_service"],
            retrieval_service=st.session_state.base_services["retrieval_service"],
            prompt_augmenter=st.session_state.base_services["augmenter"],
            generation_service=st.session_state.base_services["generation_service"],
            config=current_config
        )
        
        pt.mark("RAG Processing Query")
        response = processor.process_query(user_input)
        pt.mark("RAG Processing Query")
        
        st.session_state.chat_history.append({"user": user_input, "bot": response})


def display_chat_history() -> None:
    """Display the chat history."""
    for exchange in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(exchange["user"])
        with st.chat_message("assistant"):
            st.markdown(exchange["bot"])


def display_footer() -> None:
    """Display footer at the bottom of the page."""
    # Add some spacing to push footer down
    st.markdown("<br>" * 3, unsafe_allow_html=True)
    
    # Footer with fixed positioning
    st.markdown(
        """
        <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: transparent;
            color: #888;
            text-align: center;
            padding: 10px 0;
            font-size: 12px;
            z-index: 999;
        }
        .footer a {
            color: #888;
            text-decoration: none;
        }
        .footer a:hover {
            color: #666;
        }
        </style>
        <div class="footer">
            Built by <a href="https://jdfortress.com/labs" target="_blank">JD Fortress Labs</a>. 
            Copyright © 2025. All rights reserved.
        </div>
        """,
        unsafe_allow_html=True
    )


def main() -> None:
    """Main application function."""
    # Setup page configuration
    setup_page_config()
    
    # Load environment configuration
    config = load_environment_config()
    
    # Check authentication
    check_authentication(config["demo_username"], config["demo_password"])
    
    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Initialize services (only once)
    if "base_services" not in st.session_state:
        st.session_state.base_services = initialize_services(config)
    
    # Setup sidebar controls
    top_k, similarity_threshold = setup_sidebar()
    
    # Handle selected question from sidebar
    if "selected_question" in st.session_state:
        selected_question = st.session_state.selected_question
        del st.session_state.selected_question  # Clear after use
        if config["api_key"]:
            process_user_query(selected_question, top_k, similarity_threshold)
    
    # Handle user input
    user_input = st.chat_input("Ask a question...")
    if user_input and config["api_key"]:
        process_user_query(user_input, top_k, similarity_threshold)
    
    # Display chat history
    display_chat_history()
    
    # Display footer
    display_footer()


if __name__ == "__main__":
    main()