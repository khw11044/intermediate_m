import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain import hub
import yt_dlp
import whisper
import openai

# Whisper 모델 불러오기
model = whisper.load_model("base")

# OpenAI API 키 설정
openai.api_key= os.environ.get("OPENAI_API_KEY")

def download_audio(youtube_url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.expanduser('~/Downloads/%(title)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=True)
        file_name = ydl.prepare_filename(info_dict)
        audio_file = file_name.rsplit('.', 1)[0] + '.mp3'
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
    st.title("YouTube 뉴스 오디오 추출 및 요약 서비스")

    youtube_url = st.text_input("YouTube 뉴스 링크를 입력하세요:")

    if st.button("오디오 파일 다운로드"):
        with st.spinner("오디오 파일을 다운로드 중..."):
            audio_path = download_audio(youtube_url)
            st.success(f"오디오 파일이 다운로드되었습니다: {audio_path}")
            st.session_state.audio_path = audio_path

    if "audio_path" in st.session_state:
        st.audio(st.session_state.audio_path, format='audio/mp3')
        with st.spinner("오디오 파일을 업로드하여  텍스트로 변환 중..."):
            transcription = transcribe_audio(st.session_state.audio_path)
            st.success("오디오 파일이 텍스트로 변환되었습니다.")
            st.subheader("Transcription")
            full_response = ""
            message_placeholder = st.empty()
            for chunk in transcription.split(" "):
                full_response += chunk + " "
                # time.sleep(0.1)
                message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)

        if st.button("오디오 파일을 요약"):

            with st.spinner("텍스트를 요약 중..."):
                summary = summarize_text(transcription)
                st.success("텍스트 요약이 완료되었습니다.")
                st.subheader("요약 결과:")
                st.write(summary)

if __name__ == "__main__":
    main()
