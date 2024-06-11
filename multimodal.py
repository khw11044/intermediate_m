import os
import streamlit as st
import yt_dlp
import whisper
import openai

# Whisper 모델 불러오기
model = whisper.load_model("base")

# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]

def download_audio(youtube_url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.expanduser('~/Downloads/%(title)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True,  # 로그 출력을 비활성화
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
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"다음 텍스트를 요약해줘:\n\n{text}",
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7,
    )
    summary = response.choices[0].text.strip()
    return summary

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
