# app.py
# -----------------------------------------------------------
# ChatPDF (Streamlit + LangChain + Chroma-InMemory + LCEL)
# - PDF 업로드 → 메모리에만 저장 (디스크 저장 없음)
# - 질문 입력 → 답변 출력
# -----------------------------------------------------------

import os
import tempfile
import time
import shutil
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma  # ✅ Chroma 사용
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from operator import itemgetter

# =========================
# Streamlit 기본 UI 구성
# =========================
st.set_page_config(page_title="ChatPDF", page_icon="📄", layout="centered")
st.title("📄 ChatPDF (RAG with LCEL - In-Memory Chroma)")
st.write("---")

# API KEY 확인
if not os.getenv("OPENAI_API_KEY"):
    st.warning("⚠️ OPENAI_API_KEY 환경변수가 설정되어 있지 않습니다. .env를 확인하세요.")

# 세션 상태 초기화
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# =========================
# PDF → 문서 리스트로 변환
# =========================
def pdf_to_documents(file) -> list:
    """업로드된 PDF 파일을 임시경로에 저장 후 문서 리스트로 변환"""
    temp_dir = tempfile.TemporaryDirectory()
    temp_path = os.path.join(temp_dir.name, file.name)
    with open(temp_path, "wb") as f:
        f.write(file.getvalue())

    loader = PyPDFLoader(temp_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
    )
    docs = splitter.split_documents(pages)
    return docs

# =========================
# In-Memory Chroma 생성
# =========================
def create_chroma_in_memory(docs):
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="temp-inmemory"  # 메모리 모드에서는 이름만 쓰면 됨
        # persist_directory 지정하지 않음 → 메모리 모드
    )
    return vectorstore

# =========================
# LCEL 체인 구성
# =========================
def build_lcel_chain(vectorstore, model_name="gpt-3.5-turbo"):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = PromptTemplate.from_template(
        """
        너는 문서를 기반으로 질문에 답변하는 유용한 AI야.
        다음 문서를 참고해서 사용자 질문에 간결하고 정확하게 답변해줘.

        문서:
        {context}

        질문:
        {question}
        """
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    llm = ChatOpenAI(model=model_name, temperature=0)
    parser = StrOutputParser()

    chain = (
        {
            "question": RunnablePassthrough() | itemgetter("question"),
            "context": itemgetter("question") | retriever | RunnableLambda(format_docs),
        }
        | prompt
        | llm
        | parser
    )
    return chain

# =========================
# PDF 업로드 → 임베딩 생성
# =========================
uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("📚 PDF를 읽고 메모리에 임베딩 중입니다..."):
        docs = pdf_to_documents(uploaded_file)
        st.session_state.vectorstore = create_chroma_in_memory(docs)
        st.session_state.retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
    st.info("✅ 문서가 메모리에 업로드되었습니다. 아래에 질문해보세요.")
st.write("---")

# =========================
# 질문 입력 → 답변 출력
# =========================
st.header("PDF에게 질문해보세요!")
question = st.text_input("질문을 입력하세요")

if st.button("질문하기"):
    if uploaded_file is None:
        st.warning("먼저 PDF를 업로드해주세요.")
    elif st.session_state.vectorstore is None:
        st.warning("인덱싱이 아직 완료되지 않았습니다. 잠시 후 다시 시도해주세요.")
    elif not question.strip():
        st.warning("질문을 입력해주세요.")
    else:
        with st.spinner("🧠 답변을 생성 중입니다..."):
            chain = build_lcel_chain(st.session_state.vectorstore, model_name="gpt-3.5-turbo")
            answer = chain.invoke({"question": question})
        st.write(answer)
