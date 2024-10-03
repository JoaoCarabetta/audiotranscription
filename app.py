import streamlit as st
import sounddevice as sd
import numpy as np
import io
import wave
from google.cloud import aiplatform
from vertexai.preview.generative_models import GenerativeModel, Part
from google.oauth2 import service_account
import json
import pandas as pd

# Set up Streamlit app
st.title("Audio Recorder, Player, and Transcriber")

# Audio parameters
sample_rate = 44100  # CD quality
channels = 1  # Mono
duration = 10  # Recording duration in seconds

# Set up Google Cloud credentials
credentials = service_account.Credentials.from_service_account_file("serviceacc.json")

# Initialize Vertex AI with explicit credentials
aiplatform.init(
    project="rj-escritorio-dev", location="us-central1", credentials=credentials
)
model = GenerativeModel("gemini-1.5-flash-002")

# Initialize session state variables
if "audio_data" not in st.session_state:
    st.session_state.audio_data = None
if "recording" not in st.session_state:
    st.session_state.recording = False
if "transcription" not in st.session_state:
    st.session_state.transcription = None


def toggle_recording():
    """Toggle recording state"""
    if st.session_state.recording:
        # Stop recording
        sd.stop()
        st.session_state.recording = False
    else:
        # Start recording and playback
        playback_data = np.zeros((int(duration * sample_rate), channels))
        st.session_state.audio_data = sd.playrec(
            playback_data, samplerate=sample_rate, channels=channels
        )
        st.session_state.recording = True


def audio_to_wav_bytes(audio_data):
    """Convert audio data to WAV file bytes"""
    byte_io = io.BytesIO()
    with wave.open(byte_io, "wb") as wave_file:
        wave_file.setnchannels(channels)
        wave_file.setsampwidth(2)  # 2 bytes for int16
        wave_file.setframerate(sample_rate)
        wave_file.writeframes((audio_data * 32767).astype(np.int16).tobytes())
    return byte_io.getvalue()


def process_input(input_data, input_type):
    """Process input (audio or text) using Gemini"""
    prompt = (
        "Por favor, analise esta entrada e extraia informações médicas clínicas básicas. "
        "Forneça a saída como um objeto JSON com a seguinte estrutura:\n"
        "{\n"
        "  'transcricao': 'Transcrição completa do áudio (ou texto original para entrada de texto)',\n"
        "  'info_clinica': {\n"
        "    'nome_paciente': 'Nome do paciente, se mencionado',\n"
        "    'idade': 'Idade do paciente, se mencionada',\n"
        "    'sintomas': ['Lista de sintomas mencionados'],\n"
        "    'diagnostico': 'Qualquer diagnóstico mencionado',\n"
        "    'medicamentos': ['Lista de medicamentos mencionados'],\n"
        "    'acompanhamento': 'Quaisquer instruções de acompanhamento mencionadas',\n"
        "    'codigos_cid': [\n"
        "      {\n"
        "        'codigo': 'Código CID',\n"
        "        'descricao': 'Descrição do código em português',\n"
        "        'explicacao': 'Explicação detalhada de por que este CID foi selecionado',\n"
        "        'probabilidade': 'Valor numérico entre 0 e 1 indicando a probabilidade deste ser o CID correto'\n"
        "      }\n"
        "    ]\n"
        "  }\n"
        "}"
        "Para o campo 'codigos_cid', forneça vários códigos CID (Classificação Internacional de Doenças) "
        "que possam ser relevantes para as condições, sintomas ou diagnósticos mencionados. Use a versão mais recente (CID-11) "
        "se possível, mas CID-10 também é aceitável. Para cada CID, inclua:\n"
        "1. O código CID\n"
        "2. Sua descrição em português\n"
        "3. Uma explicação detalhada de por que este CID foi selecionado com base nas informações fornecidas\n"
        "4. Uma estimativa de probabilidade (entre 0 e 1) de que este seja o CID correto\n\n"
        "Forneça múltiplos CIDs possíveis, ordenados do mais provável ao menos provável. "
        "A probabilidade deve ser baseada na confiança da correspondência entre as informações fornecidas e o código CID. "
        "Certifique-se de que todos os códigos CID, suas descrições e explicações estejam em português. "
        "Inclua CIDs alternativos que possam ser relevantes, mesmo que tenham uma probabilidade menor."
    )

    if input_type == "audio":
        audio_part = Part.from_data(input_data, mime_type="audio/wav")
        response = model.generate_content([audio_part, prompt])
    else:  # text input
        response = model.generate_content([input_data, prompt])

    # Parse the JSON response
    response_text = "\n".join(response.text.split("\n")[1:-2])
    print(response_text)
    result = json.loads(response_text)

    # Transform JSON to DataFrame
    info_clinica = result["info_clinica"]
    df_info = pd.DataFrame(
        {
            "Campo": [
                "Nome do Paciente",
                "Idade",
                "Sintomas",
                "Diagnóstico",
                "Medicamentos",
                "Acompanhamento",
            ],
            "Valor": [
                info_clinica["nome_paciente"],
                info_clinica["idade"],
                ", ".join(info_clinica["sintomas"]),
                info_clinica["diagnostico"],
                ", ".join(info_clinica["medicamentos"]),
                info_clinica["acompanhamento"],
            ],
        }
    )

    df_cid = pd.DataFrame(info_clinica["codigos_cid"])

    return result["transcricao"], df_info, df_cid


# Add radio button for input selection
input_type = st.radio("Select input type:", ("Record Audio", "Upload Audio", "Text"))

if input_type == "Record Audio":
    # Create a single button for recording control
    if st.button("Start/Stop Recording"):
        toggle_recording()

    # Display recording status
    st.write(f"Recording: {'Yes' if st.session_state.recording else 'No'}")

    # Play recorded audio and process
    if st.session_state.audio_data is not None and not st.session_state.recording:
        wav_bytes = audio_to_wav_bytes(st.session_state.audio_data)
        st.audio(wav_bytes, format="audio/wav")

        if st.button("Process Audio"):
            with st.spinner("Processing..."):
                st.session_state.transcription = process_input(wav_bytes, "audio")
            st.success("Processing complete!")

elif input_type == "Upload Audio":
    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])
    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")
        if st.button("Process Uploaded Audio"):
            with st.spinner("Processing..."):
                st.session_state.transcription = process_input(
                    uploaded_file.getvalue(), "audio"
                )
            st.success("Processing complete!")

else:  # Text input
    # Text input area
    text_input = st.text_area("Enter the text to process:")

    if st.button("Process Text"):
        if text_input:
            with st.spinner("Processing..."):
                st.session_state.transcription = process_input(text_input, "text")
            st.success("Processing complete!")
        else:
            st.warning("Please enter some text to process.")

# Display processed result
if st.session_state.transcription:
    st.subheader("Relatório Processado:")

    transcricao, df_info, df_cid = st.session_state.transcription

    st.subheader("Transcrição:")
    st.write(transcricao)

    st.subheader("Informações Clínicas:")
    st.table(df_info.set_index("Campo"))

    st.subheader("Códigos CID Sugeridos:")
    st.table(df_cid.set_index("codigo"))
