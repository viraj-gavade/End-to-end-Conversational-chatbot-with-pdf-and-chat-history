# End-to-End Conversational Chatbot with PDF and Chat History

This project is a Streamlit-based conversational chatbot that allows users to upload PDF files and interact with their content using conversational queries. The chatbot leverages Retrieval-Augmented Generation (RAG) and maintains chat history for context-aware responses.

## Features
- **PDF Upload**: Upload a PDF and chat with its content.
- **Conversational RAG**: Uses LangChain's RAG pipeline for context-aware question answering.
- **Chat History**: Maintains session-based chat history for improved context.
- **Embeddings & Vector Store**: Utilizes HuggingFace embeddings and Chroma vector store for document retrieval.
- **Environment Variables**: Uses `.env` for API keys and tokens.

## Getting Started

### Prerequisites
- Python 3.8+
- Install dependencies from `requirements.txt`

### Installation
1. Clone the repository.
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project root with the following variables:
   ```env
   HF_TOKEN=your_huggingface_token
   GROQ_API_KEY=your_groq_api_key
   ```

### Running the App
Run the Streamlit app:
```powershell
streamlit run app.py
```

## Usage
1. Enter a session ID (or use the default).
2. Upload a PDF file.
3. Ask questions about the PDF content.
4. View responses and chat history.

## Main Technologies
- [Streamlit](https://streamlit.io/)
- [LangChain](https://python.langchain.com/)
- [HuggingFace Transformers](https://huggingface.co/)
- [ChromaDB](https://www.trychroma.com/)

## File Structure
- `app.py`: Main application code.
- `requirements.txt`: Python dependencies.

## License
This project is for educational purposes.
