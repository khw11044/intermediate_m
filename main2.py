import os
import streamlit as st
from pytube import YouTube
import whisper
from langchain_openai import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain import hub
from langchain_core.prompts import PromptTemplate

# Whisper 모델 불러오기
model = whisper.load_model("base")

# OpenAI API 키 설정
# os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
# os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]

def download_audio(youtube_url):
    yt = YouTube(youtube_url)
    stream = yt.streams.filter(only_audio=True).first()
    out_file = stream.download(output_path=os.path.expanduser('~/Downloads'))
    base, ext = os.path.splitext(out_file)
    audio_file = base + '.mp3'
    os.rename(out_file, audio_file)
    return audio_file

def transcribe_audio(audio_path):
    result = model.transcribe(audio_path)
    return result["text"]

def summarize_text(text):
    llm = ChatOpenAI(model="gpt-3.5-turbo")  # 또는 "gpt-4"
    text_splitter = CharacterTextSplitter()
    docs = text_splitter.create_documents([text])
    prompt = PromptTemplate.from_template(
        "Always answer in Korean! Act as an expert copywriter specializing in content optimization for SEO. "
        "Your task is to take a given YouTube transcript and transform it into a well-structured and engaging article. "
        "Your objectives are as follows:\n\n"
        "Content Transformation: Begin by thoroughly reading the provided YouTube transcript. Understand the main ideas, key points, and the overall message conveyed.\n\n"
        "Sentence Structure: While rephrasing the content, pay careful attention to sentence structure. Ensure that the article flows logically and coherently.\n\n"
        "Keyword Identification: Identify the main keyword or phrase from the transcript. It's crucial to determine the primary topic that the YouTube video discusses.\n\n"
        "Keyword Integration: Incorporate the identified keyword naturally throughout the article. Use it in headings, subheadings, and within the body text. However, avoid overuse or keyword stuffing, as this can negatively affect SEO.\n\n"
        "Unique Content: Your goal is to make the article 100% unique. Avoid copying sentences directly from the transcript. Rewrite the content in your own words while retaining the original message and meaning.\n\n"
        "SEO Friendliness: Craft the article with SEO best practices in mind. This includes optimizing meta tags (title and meta description), using header tags appropriately, and maintaining an appropriate keyword density.\n\n"
        "Engaging and Informative: Ensure that the article is engaging and informative for the reader. It should provide value and insight on the topic discussed in the YouTube video.\n\n"
        "Proofreading: Proofread the article for grammar, spelling, and punctuation errors. Ensure it is free of any mistakes that could detract from its quality.\n\n"
        "By following these guidelines, create a well-optimized, unique, and informative article that would rank well in search engine results and engage readers effectively.\n\n"
        "Transcript:{transcript}"
    )
    chain = prompt | llm
    summary = chain.invoke({"transcript": docs})
    return summary.content

def main():
    st.title("YouTube 뉴스 오디오 추출 및 요약 서비스")

    youtube_url = st.text_input("YouTube 뉴스 링크를 입력하세요:")
    
    if st.button("오디오 파일 다운로드"):
        with st.spinner("오디오 파일을 다운로드 중..."):
            audio_path = download_audio(youtube_url)
            st.success(f"오디오 파일이 다운로드되었습니다: {audio_path}")
            st.session_state.audio_path = audio_path

    if "audio_path" in st.session_state:
        st.audio(st.session_state.audio_path, format='audio/mp3')

        if st.button("오디오 파일을 업로드하여 요약"):
            with st.spinner("오디오 파일을 텍스트로 변환 중..."):
                text = transcribe_audio(st.session_state.audio_path)
                st.success("오디오 파일이 텍스트로 변환되었습니다.")

            with st.spinner("텍스트를 요약 중..."):
                summary = summarize_text(text)
                st.success("텍스트 요약이 완료되었습니다.")
                st.write("요약 결과:")
                st.write(summary)

if __name__ == "__main__":
    main()