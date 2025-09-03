# app.py
# -----------------------------------------------------------
# ChatPDF (Streamlit + LangChain + Chroma + LCEL)
# - PDF 업로드 시: 기존 DB 폴더가 있으면 안전 삭제 후 재생성(윈도우 파일락 대응)
# - 질문 입력 → 스피너 대기 → RAG로 답변 출력
# -----------------------------------------------------------
# 사전 설치:
# pip install -U streamlit langchain langchain-openai langchain-chroma python-dotenv pypdf

import os
import gc
import time
import shutil
import tempfile
import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # .env에서 OPENAI_API_KEY 로드

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from operator import itemgetter

# =========================
# 페이지 기본 UI
# =========================
st.set_page_config(page_title="ChatPDF", page_icon="📄", layout="centered")
st.title("📄 ChatPDF (RAG with LCEL)")
st.write("---")

if not os.getenv("OPENAI_API_KEY"):
    st.warning("⚠️ OPENAI_API_KEY 환경변수가 설정되어 있지 않습니다. .env를 확인하세요.")

# 세션 상태
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "persist_dir" not in st.session_state:
    st.session_state.persist_dir = "./db/chromadb"
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# =========================
# 안전 삭제 유틸 (윈도우 락 대응)
# =========================
def safe_rmtree(path: str, retries: int = 8, delay: float = 0.2):
    """윈도우 파일락을 고려해 rmtree를 여러 번 재시도."""
    last_err = None
    for _ in range(retries):
        try:
            shutil.rmtree(path)
            return
        except PermissionError as e:
            last_err = e
            time.sleep(delay)
    if last_err:
        raise last_err

# =========================
# 업로드 → 문서 변환
# =========================
def pdf_to_documents(file) -> list:
    """업로드된 파일을 임시경로에 저장 후 페이지/청크 단위 문서 리스트로 변환."""
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
# Chroma 재생성(자동 초기화)
# =========================
def recreate_chroma(docs, persist_dir="./db/chromadb", collection_name="esg"):
    import chromadb
    from langchain_openai import OpenAIEmbeddings
    from langchain_chroma import Chroma

    # 1) 같은 경로로 PersistentClient 연결
    client = chromadb.PersistentClient(path=persist_dir)

    # 2) 기존 컬렉션이 있으면 삭제 (파일락 없이 안전)
    try:
        client.delete_collection(collection_name)
    except Exception:
        # 없으면 무시
        pass

    # 3) 새 임베딩 + 새 컬렉션 생성 (동일 persist_dir 재사용)
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=collection_name,
        client=client,                 # ← 중요: 같은 client 사용
        persist_directory=persist_dir  # ← 경로 유지
    )
    return vectorstore

# =========================
# LCEL 체인 (질문→검색→프롬프트→LLM→파싱)
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

    # 입력: {"question": "..."} 를 기대
    # question은 그대로 보존, context는 question→retriever→format으로 생성
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
# 업로드 UI & 자동 인덱싱
# =========================
uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("📚 업로드한 PDF를 읽고 임베딩을 생성 중입니다..."):
        docs = pdf_to_documents(uploaded_file)
        st.session_state.vectorstore = recreate_chroma(
            docs,
            persist_dir=st.session_state.persist_dir,
            collection_name="esg"
        )
        st.session_state.retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
    #st.success(f"인덱싱 완료! (총 청크 수: {len(docs)})")
    st.info("문서가 업로드되었습니다. 이제 아래에 질문을 입력해 보세요.")
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

# # =========================
# # 디버그 정보 (선택)
# # =========================
# with st.expander("디버그 정보"):
#     st.write("persist_dir:", st.session_state.persist_dir)
#     st.write("vectorstore 생성됨:", st.session_state.vectorstore is not None)
