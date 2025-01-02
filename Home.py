# (venv)환경에서 streamlit run Home.py 명령어 실행
from crawl_data import get_vectorstore
from rag_system import get_medical_chain, get_retriever, format_docs
from langchain_community.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks import StdOutCallbackHandler
import streamlit as st
import os

# messages에 message 저장


def save_message(message, role):
    st.session_state["messages"].append({"role": role, "message": message})

# ui에 message를 추가합니다.


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

# ui에 기존에 주고받은 message를 그립니다.


def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


# 페이지 설정
st.set_page_config(
    page_title="Dr. Navi",
    page_icon="🩺",
)

# 챗봇 콜백 핸들러 (좌라라라락 쓰게 만들어줍니다.)


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        # 챗봇이 대화를 시작할 때마다 새로운 메시지를 초기화합니다.
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        # 챗봇이 대화를 끝낼 때 메세지를 messages에 저장합니다.
        save_message(self.message, "ai")
        # print()

    # 토큰이 도착할때마다 메세지를 그려줍니다.
    def on_llm_new_token(self, token: str, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)
        # print(token, end="")


# llm 생성
# stream이 true인 llm
llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.1,
    model="gpt-4o-mini",
    streaming=True,
    callbacks=[ChatCallbackHandler()],
)

st.title('🩺Dr. Navi')
st.markdown("의료 상담 챗봇입니다. 증상을 입력하면 진료과 추천, 주의사항 등을 안내해드립니다. 아래 정보를 입력해주세요.")
st.markdown("팀원 : 김민철, 임동성, 지용석, 최유정, 허현준")
with st.expander("사용자 정보 입력"):
    gender = st.radio("성별을 선택해주세요", ["남성", "여성"], index=None)
    age = st.number_input("나이를 입력해주세요", min_value=0, max_value=120, value=20)

# message 기록 저장 초기화 -> messages
if "messages" not in st.session_state:
    st.session_state["messages"] = []
    save_message(
        "안녕하세요! 의료 상담 챗봇입니다. 증상을 입력하면 진료과 추천, 주의사항 등을 안내해드립니다.\n\nex) 콧물과 기침이 나와요", "ai")

# 이전 대화 기록을 그립니다.
paint_history()


# 사용자 입력을 받습니다.
message = st.chat_input("증상을 입력해주세요...")
if message:  # 사용자가 입력한 메시지가 있으면
    # 사용자가 입력한 메시지를 그립니다.
    send_message(message, "human")
    # 사용자가 입력한 메시지를 llm에 전달합니다.
    with st.spinner("Thinking..."):
        # vectorstore을 pinecone에서 가져옵니다.
        vectorstore = get_vectorstore()
        # medical_chain을 가져옵니다.
        medical_chain = get_medical_chain(llm)
        # retriver를 가져옵니다.
        retriver = get_retriever(vectorstore)

        if gender is None:
            gender = "남성"
        if age is None:
            age = 20
    # ai가 대답한 메시지를 그립니다.
    with st.chat_message("ai"):
        medical_chain.invoke({
            "context": format_docs(retriver.invoke(message)),
            "input": message,
            "sex": gender,
            "age": age
        })
