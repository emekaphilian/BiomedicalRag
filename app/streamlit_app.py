"""
Streamlit web interface for biomedical RAG system.
Provides interactive document upload and Q&A interface with safety warnings.
"""

import streamlit as st
import logging

logger = logging.getLogger(__name__)


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Biomedical RAG System",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Add medical disclaimer
    st.warning(
        "⚠️ **MEDICAL DISCLAIMER**: This system is for research and educational purposes only. "
        "It should NOT be used for medical diagnosis or treatment decisions. "
        "Always consult with qualified healthcare professionals."
    )
    
    st.title("🏥 Biomedical Document Q&A System")
    st.markdown("Retrieval-Augmented Generation with Safety Constraints")
    
    # TODO: Implement UI components
    pass


if __name__ == "__main__":
    main()
