import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
import tempfile
import os
import time
from langchain import hub
from langchain_core.prompts import PromptTemplate

# !pip install openai-whisper
import whisper
whispermodel = whisper.load_model("base")


from dotenv import load_dotenv
load_dotenv()
import openai
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]

# openai.api_key=os.getenv("OPENAI_API_KEY")
# openai.api_key= os.environ.get("OPENAI_API_KEY")

# conda install -c conda-forge ffmpeg
def transcribe_audio(file_path):
    result = whispermodel.transcribe(file_path)     # 오류발생, 인터넷에 isper model FileNotFoundError: [WinError 2] 또는 ffmpeg 검색
    return result['text']

def summarize_text(text):
    llm = ChatOpenAI(model="gpt-3.5-turbo")            # gpt-4o
    text_splitter = CharacterTextSplitter()
    docs = text_splitter.create_documents([text])
    prompt = PromptTemplate.from_template("Always answer in Korean! Act as an expert copywriter specializing in content optimization for SEO. Your task is to take a given YouTube transcript and transform it into a well-structured and engaging article. Your objectives are as follows:\n\nContent Transformation: Begin by thoroughly reading the provided YouTube transcript. Understand the main ideas, key points, and the overall message conveyed.\n\nSentence Structure: While rephrasing the content, pay careful attention to sentence structure. Ensure that the article flows logically and coherently.\n\nKeyword Identification: Identify the main keyword or phrase from the transcript. It's crucial to determine the primary topic that the YouTube video discusses.\n\nKeyword Integration: Incorporate the identified keyword naturally throughout the article. Use it in headings, subheadings, and within the body text. However, avoid overuse or keyword stuffing, as this can negatively affect SEO.\n\nUnique Content: Your goal is to make the article 100% unique. Avoid copying sentences directly from the transcript. Rewrite the content in your own words while retaining the original message and meaning.\n\nSEO Friendliness: Craft the article with SEO best practices in mind. This includes optimizing meta tags (title and meta description), using header tags appropriately, and maintaining an appropriate keyword density.\n\nEngaging and Informative: Ensure that the article is engaging and informative for the reader. It should provide value and insight on the topic discussed in the YouTube video.\n\nProofreading: Proofread the article for grammar, spelling, and punctuation errors. Ensure it is free of any mistakes that could detract from its quality.\n\nBy following these guidelines, create a well-optimized, unique, and informative article that would rank well in search engine results and engage readers effectively.\n\nTranscript:{transcript}")
    chain = prompt | llm
    # chain = load_summarize_chain(llm, chain_type="stuff", verbose=False)
    summary = chain.invoke({"transcript": docs})
    return summary.content


# Streamlit app
st.title("유튜브 뉴스 영상 SEO 컨텐츠 기사로 만들기")
st.write("음성 파일을 업로드하면 기사글을 작성해줍니다.")

# pip install pytube
from pytube import YouTube
from pathlib import Path

st.markdown(f"<a href='https://www.youtube.com/watch?v=4EzXnCfB5oU' style='font-size:14px;'>예시: https://www.youtube.com/watch?v=4EzXnCfB5oU</a>", unsafe_allow_html=True)
address = st.text_input('유튜브 주소를 입력하고 엔터를 눌러주세요.')
if address:
    try:
        # 임시 디렉토리에 파일 다운로드
        with tempfile.TemporaryDirectory() as tmpdirname:
            yt = YouTube(address)
            audio_stream = yt.streams.filter(only_audio=True).first()
            file_path = audio_stream.download(output_path=tmpdirname)
            file_name = os.path.basename(file_path)
            
            st.success("오디오 파일이 다운로드되었습니다.")
            # 파일을 다운로드할 수 있는 링크 제공
            with open(file_path, "rb") as file:
                btn = st.download_button(
                    label="오디오 파일 다운로드",
                    data=file,
                    file_name=file_name,
                    mime="audio/mp4"
                )
    except Exception as e:
        st.error(f"오디오 파일을 다운로드하는 중 오류가 발생했습니다: {e}")



audio_file = st.file_uploader("다운로드 폴더에 다운받은 오디오파일을 업로드해주세요.", type=["wav", "mp3", "mp4","m4a"])

if audio_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(audio_file.read())
        temp_file_path = temp_file.name

    print('temp_file_path:',temp_file_path)
    st.audio(temp_file_path, format="audio/wav")
    transcription = transcribe_audio(temp_file_path)
    st.subheader("Transcription")
    full_response = ""
    message_placeholder = st.empty()
    for chunk in transcription.split(" "):
        full_response += chunk + " "
        time.sleep(0.1)
        message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    summary = summarize_text(transcription)
    st.subheader("Summary")
    st.write(summary)
        
    os.remove(temp_file_path)
