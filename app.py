import gradio as gr
import re
from youtube_transcript_api import YouTubeTranscriptApi  # For extracting transcripts from YouTube videos
from langchain.text_splitter import RecursiveCharacterTextSplitter  # For splitting text into manageable segments
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes  # For specifying model types
from ibm_watsonx_ai import APIClient, Credentials  # For API client and credentials management
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams  # For managing model parameters
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods  # For defining decoding methods
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings  # For interacting with IBM's LLM and embeddings
from ibm_watsonx_ai.foundation_models.utils import get_embedding_model_specs  # For retrieving model specifications
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes  # For specifying types of embeddings
from langchain_community.vectorstores import FAISS  # For efficient vector storage and similarity search
from langchain.chains import LLMChain  # For creating chains of operations with LLMs
from langchain.prompts import PromptTemplate  # For defining prompt templates
from pypdf import PdfReader


# PDF reading utilities

def extract_text_from_pdf(pdf_path: str) -> str:
    if not pdf_path:
        return ""
    try:
        reader = PdfReader(pdf_path)
    except Exception as e:
        return f"ERROR reading PDF: {e}"

    pages_text = []
    for i, page in enumerate(reader.pages):
        try:
            t = page.extract_text() or ""
            t = t.replace("\x00", " ").strip()
            if t:
                pages_text.append(f"[Page {i+1}]\n{t}")
        except Exception:
            continue

    return "\n\n".join(pages_text)


def build_session_from_pdf(pdf_path: str):
    """
    Builds per-user session artifacts:
    - extracted text
    - FAISS index (embedded chunks)
    Returns: (status_msg, document_text, faiss_index)
    """
    if not pdf_path:
        return "Please upload a PDF.", "", None

    document_text = extract_text_from_pdf(pdf_path)
    if not document_text or document_text.startswith("ERROR"):
        return (document_text or "No extractable text found in the PDF."), "", None

    # Chunk only once
    chunks = chunk_document(document_text)

    # Credentials + embedding model (same as your current setup)
    model_id, credentials, client, project_id = setup_credentials()
    embedding_model = setup_embedding_model(credentials, project_id)

    # Build FAISS once per upload (per user)
    faiss_index = create_faiss_index(chunks, embedding_model)

    return "PDF loaded. Index built. Ready for Q&A.", document_text, faiss_index

def summarize_pdf_from_state(pdf_path: str, document_text_state: str):
    """
    Summarize using the already-extracted text stored in session state.
    Returns: summary_text
    """
    # If state empty (e.g. user clicks regenerate first), fallback to reading
    document_text = document_text_state or extract_text_from_pdf(pdf_path)

    if not pdf_path:
        return "Please upload a PDF."
    if not document_text or document_text.startswith("ERROR"):
        return document_text or "No extractable text found in the PDF."

    model_id, credentials, client, project_id = setup_credentials()
    llm = initialize_watsonx_llm(model_id, credentials, project_id, define_parameters())

    summary_prompt = create_summary_prompt()
    summary_chain = create_summary_chain(llm, summary_prompt)

    return summary_chain.run({"document_text": document_text})

def answer_question_from_state(pdf_path: str, user_question: str, faiss_index_state):
    """
    Answer questions using the per-session FAISS index stored in gr.State().
    Returns: answer_text
    """
    if not pdf_path:
        return "Please upload a PDF."
    if not user_question or not user_question.strip():
        return "Please type a question."

    if faiss_index_state is None:
        # If index isn't built (e.g. user didn't trigger upload event), build it now
        status, doc_text, faiss_index_state = build_session_from_pdf(pdf_path)
        if faiss_index_state is None:
            return status  # contains error msg

    model_id, credentials, client, project_id = setup_credentials()
    llm = initialize_watsonx_llm(model_id, credentials, project_id, define_parameters())

    qa_prompt = create_qa_prompt_template()
    qa_chain = create_qa_chain(llm, qa_prompt)

    return generate_answer(user_question, faiss_index_state, qa_chain, k=7)

def reset_state():
    """
    Clears per-user session state.
    Returns: (status, document_text_state, faiss_index_state, summary_box, answer_box, question_box)
    """
    return "State cleared. Upload a PDF to start.", "", None, "", "", ""




def chunk_document(document_text, chunk_size=1200, chunk_overlap=150):
    """
    Chunk extracted PDF text into segments for retrieval.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_text(document_text)



#Watsonx 
def setup_credentials():
    # Define the model ID for the WatsonX model being used
    model_id = "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
   
    # Set up the credentials by specifying the URL for IBM Watson services
    credentials = Credentials(url="https://us-south.ml.cloud.ibm.com")
   
    # Create an API client using the credentials
    client = APIClient(credentials)
   
    # Define the project ID associated with the WatsonX platform
    project_id = "skills-network"
   
    # Return the model ID, credentials, client, and project ID for later use
    return model_id, credentials, client, project_id


def define_parameters():
    return {
        GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
        GenParams.MAX_NEW_TOKENS: 900,
    }


def initialize_watsonx_llm(model_id, credentials, project_id, parameters):
    return WatsonxLLM(
        model_id=model_id,
        url=credentials.get("url"),
        project_id=project_id,
        params=parameters
    )


def setup_embedding_model(credentials, project_id):
    return WatsonxEmbeddings(
        model_id="ibm/slate-30m-english-rtrvr-v2",
        url=credentials["url"],
        project_id=project_id
    )


def create_faiss_index(chunks, embedding_model):
    return FAISS.from_texts(chunks, embedding_model)


def retrieve(query, faiss_index, k=7):
    return faiss_index.similarity_search(query, k=k)



# Prompts


def create_summary_prompt():
    template = """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an AI assistant tasked with summarizing PDF documents. Provide a concise, informative summary that captures the key ideas, structure, and important details.

    Instructions:
    1. Summarize the document in a single concise paragraph.
    2. If the document includes sections (e.g., Abstract, Methods, Results), reflect that structure briefly.
    3. Ignore page markers like [Page 1], [Page 2] in your summary.
    4. Focus on the actual content and meaning of the document.<|eot_id|><|start_header_id|>user<|end_header_id|>
    Please summarize the following PDF content:

    {document_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    return PromptTemplate(
        input_variables=["document_text"],
        template=template
    )


def create_qa_prompt_template():
    qa_template = """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an expert assistant answering questions based on the provided PDF content.
    Your responses must be:
    1. Accurate and grounded in the provided context
    2. Clear, well-organized, and directly addressing the user's question
    3. Free from repetition
    If the answer is not present in the context, say you cannot find it in the PDF content provided and suggest what to search for or which section might contain it.<|eot_id|>

    <|start_header_id|>user<|end_header_id|>
    Relevant PDF Context:
    {context}

    Question:
    {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    return PromptTemplate(
        input_variables=["context", "question"],
        template=qa_template
    )


def create_summary_chain(llm, prompt, verbose=True):
    return LLMChain(llm=llm, prompt=prompt, verbose=verbose)


def create_qa_chain(llm, prompt_template, verbose=True):
    return LLMChain(llm=llm, prompt=prompt_template, verbose=verbose)


def generate_answer(question, faiss_index, qa_chain, k=7):
    relevant_context_docs = retrieve(question, faiss_index, k=k)

    # Convert docs to a single context string (FAISS returns Documents)
    # Each Document usually has .page_content
    context_text = "\n\n".join(
        [getattr(d, "page_content", str(d)) for d in relevant_context_docs]
    )

    return qa_chain.predict(context=context_text, question=question)



# Gradio app functions


# We store the current document text globally
processed_document_text = ""
cached_faiss_index = None  #cache


def summarize_pdf(pdf_file):
    global processed_document_text

    if pdf_file is None:
        return "Please upload a PDF file."

    processed_document_text = extract_text_from_pdf(pdf_file)

    if not processed_document_text or processed_document_text.startswith("ERROR"):
        return processed_document_text or "No extractable text found in the PDF."

    model_id, credentials, client, project_id = setup_credentials()
    llm = initialize_watsonx_llm(model_id, credentials, project_id, define_parameters())

    summary_prompt = create_summary_prompt()
    summary_chain = create_summary_chain(llm, summary_prompt)

    summary = summary_chain.run({"document_text": processed_document_text})
    return summary


def answer_question(pdf_file, user_question):
    global processed_document_text, cached_faiss_index

    if pdf_file is None:
        return "Please upload a PDF file."

    if not user_question or not user_question.strip():
        return "Please type a question."

    # If we haven't extracted text yet (or PDF changed), extract again
    if not processed_document_text:
        processed_document_text = extract_text_from_pdf(pdf_file)

    if not processed_document_text or processed_document_text.startswith("ERROR"):
        return processed_document_text or "No extractable text found in the PDF."

    # Build FAISS index once and reuse
    if cached_faiss_index is None:
        chunks = chunk_document(processed_document_text)

        model_id, credentials, client, project_id = setup_credentials()
        embedding_model = setup_embedding_model(credentials, project_id)
        cached_faiss_index = create_faiss_index(chunks, embedding_model)

    # LLM + QA chain
    model_id, credentials, client, project_id = setup_credentials()
    llm = initialize_watsonx_llm(model_id, credentials, project_id, define_parameters())

    qa_prompt = create_qa_prompt_template()
    qa_chain = create_qa_chain(llm, qa_prompt)

    answer = generate_answer(user_question, cached_faiss_index, qa_chain, k=7)
    return answer





# Gradio UI

with gr.Blocks() as interface:
    gr.Markdown("<h2 style='text-align: center;'>CV Summarizer and Q&A</h2>")

    # Upload
    pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"], type="filepath")

    # ✅ Per-session state (no globals)
    document_text_state = gr.State("")     # stores extracted text
    faiss_index_state = gr.State(None)     # stores FAISS index

    # ===== Summary section =====
    with gr.Column():
        summary_output = gr.Textbox(label="Candidate's Summary", lines=6)
        summarize_btn = gr.Button("Regenerate Summary")

    gr.Markdown("---")

    # ===== Q&A section =====
    with gr.Column():
        question_input = gr.Textbox(
            label="Ask a Question About the Candidate",
            placeholder="Type your question here..."
        )
        answer_output = gr.Textbox(label="Answer", lines=6)
        question_btn = gr.Button("Ask Question")

    reset_btn = gr.Button("Reset / Clear")
    status = gr.Textbox(label="Status", interactive=False)

    # ✅ 1) When PDF is uploaded/changed: build session (text + FAISS)
    pdf_input.change(
        fn=build_session_from_pdf,
        inputs=pdf_input,
        outputs=[status, document_text_state, faiss_index_state]
    )

    # ✅ 2) Auto-generate summary after upload (uses extracted text from state)
    pdf_input.change(
        fn=summarize_pdf_from_state,
        inputs=[pdf_input, document_text_state],
        outputs=summary_output
    )

    # ✅ 3) Button to regenerate summary
    summarize_btn.click(
        fn=summarize_pdf_from_state,
        inputs=[pdf_input, document_text_state],
        outputs=summary_output
    )

    # ✅ 4) Q&A uses per-session FAISS index
    question_btn.click(
        fn=answer_question_from_state,
        inputs=[pdf_input, question_input, faiss_index_state],
        outputs=answer_output
    )

    # ✅ 5) Reset session
    reset_btn.click(
        fn=reset_state,
        inputs=[],
        outputs=[status, document_text_state, faiss_index_state, summary_output, answer_output, question_input]
    )

interface.launch(server_name="0.0.0.0", server_port=7860)

