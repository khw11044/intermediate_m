import os
import streamlit as st
from pytube import YouTube
import whisper
import openai
from langchain_openai import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain import hub
from langchain_core.prompts import PromptTemplate

# Whisper 모델 불러오기
model = whisper.load_model("base")

# # OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]

def download_audio(youtube_url):
    yt = YouTube(youtube_url)
    stream = yt.streams.filter(only_audio=True).first()
    out_file = stream.download(output_path=os.path.expanduser('~/Downloads'))
    base, ext = os.path.splitext(out_file)
    audio_file = base + '.mp3'
    if os.path.exists(audio_file):
        os.remove(audio_file)  # 기존 파일이 존재하면 삭제
    os.rename(out_file, audio_file)
    return audio_file

def transcribe_audio(audio_path):
    result = model.transcribe(audio_path)
    return result["text"]


def summarize_text(text):
    llm = ChatOpenAI(model="gpt-3.5-turbo")            # gpt-4o
    text_splitter = CharacterTextSplitter()
    docs = text_splitter.create_documents([text])
    prompt = PromptTemplate.from_template("Always answer in Korean! Act as an expert copywriter specializing in content optimization for SEO. Your task is to take a given YouTube transcript and transform it into a well-structured and engaging article. Your objectives are as follows:\n\nContent Transformation: Begin by thoroughly reading the provided YouTube transcript. Understand the main ideas, key points, and the overall message conveyed.\n\nSentence Structure: While rephrasing the content, pay careful attention to sentence structure. Ensure that the article flows logically and coherently.\n\nKeyword Identification: Identify the main keyword or phrase from the transcript. It's crucial to determine the primary topic that the YouTube video discusses.\n\nKeyword Integration: Incorporate the identified keyword naturally throughout the article. Use it in headings, subheadings, and within the body text. However, avoid overuse or keyword stuffing, as this can negatively affect SEO.\n\nUnique Content: Your goal is to make the article 100% unique. Avoid copying sentences directly from the transcript. Rewrite the content in your own words while retaining the original message and meaning.\n\nSEO Friendliness: Craft the article with SEO best practices in mind. This includes optimizing meta tags (title and meta description), using header tags appropriately, and maintaining an appropriate keyword density.\n\nEngaging and Informative: Ensure that the article is engaging and informative for the reader. It should provide value and insight on the topic discussed in the YouTube video.\n\nProofreading: Proofread the article for grammar, spelling, and punctuation errors. Ensure it is free of any mistakes that could detract from its quality.\n\nBy following these guidelines, create a well-optimized, unique, and informative article that would rank well in search engine results and engage readers effectively.\n\nTranscript:{transcript}")
    chain = prompt | llm
    # chain = load_summarize_chain(llm, chain_type="stuff", verbose=False)
    summary = chain.invoke({"transcript": docs})
    return summary.content

def main():
    st.title("현우의 YouTube 뉴스 오디오 추출 및 요약 서비스")

    youtube_url = st.text_input("YouTube 뉴스 링크를 입력하세요:")
    st.markdown(f"<a href='https://www.youtube.com/watch?v=4EzXnCfB5oU' style='font-size:14px;'>예시 링크) https://www.youtube.com/watch?v=4EzXnCfB5oU</a>", unsafe_allow_html=True)

    if st.button("오디오 파일 다운로드"):
        with st.spinner("오디오 파일을 다운로드 중..."):
            
            audio_path = download_audio(youtube_url)
            st.success(f"오디오 파일이 다운로드되었습니다: {audio_path}")
            st.session_state.audio_path = audio_path

    if "audio_path" in st.session_state:
        st.audio(st.session_state.audio_path, format='audio/mp3')

        with st.spinner("오디오 파일을 텍스트로 변환 중..."):
            transcription = transcribe_audio(st.session_state.audio_path)
            st.subheader("Transcription")
            full_response = ""
            message_placeholder = st.empty()
            for chunk in transcription.split(" "):
                full_response += chunk + " "
                message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
        
        if st.button("요약하기"):
            
            with st.spinner("텍스트를 요약 중..."):
                summary = summarize_text(transcription)
                st.success("텍스트 요약이 완료되었습니다.")
                st.write("요약 결과:")
                st.write(summary)
                
            with open(audio_path, "rb") as file:
                file_name = os.path.basename(audio_path)
                btn = st.download_button(
                    label="오디오 파일 다운로드",
                    data=file,
                    file_name=file_name,
                    mime="audio/mp4"
                )

            # 사용 후 파일 삭제
            if os.path.exists(st.session_state.audio_path):
                os.remove(st.session_state.audio_path)
                del st.session_state.audio_path
    
if __name__ == "__main__":
    main()
