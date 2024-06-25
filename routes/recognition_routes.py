
from fastapi import APIRouter, File, UploadFile, HTTPException
import librosa
import io

from transformers import Wav2Vec2Processor
from schemas.AudioResponse import AudioResponse
from services.recognition_service import AgeGenderModel, RecognitionService

router_recognition = APIRouter(
    tags=["Recognition"],
    prefix="/recognition",
    responses={
        400: {"description": "Bad Request",
              "content": {"application/json": {"example": {"error": "Invalid audio file"}}}},
        413: {"description": "Request Entity Too Large",
              "content": {"application/json": {"example": {"error": "File too large"}}}},
        415: {"description": "Unsupported Media Type",
              "content": {"application/json": {"example": {"error": "Unsupported file format"}}}},
        422: {"description": "Unprocessable Entity",
              "content": {"application/json": {"example": {"error": "Invalid parameters"}}}},
        500: {"description": "Internal Server Error",
              "content": {"application/json": {"example": {"error": "Internal server error"}}}}
    }
)


GENDERS = ("Female", "Male", "Child")


def get_gender(genders_probability: [float]):
    max_gp = max(genders_probability)
    pos = list(genders_probability).index(max_gp)
    return GENDERS[pos]


@router_recognition.post("/process-audio/", response_model=AudioResponse)
async def process_audio(audio: UploadFile = File(...)):
    # Verificar el tipo de archivo
    if not audio.content_type.startswith('audio/'):
        raise HTTPException(status_code=415, detail="File must be an audio file (e.g. .wav, .mp3)")

    # Verificar el tamaño del archivo (por ejemplo, límite de 30MB)
    max_size = 30 * 1024 * 1024
    if audio.size > max_size:
        raise HTTPException(status_code=413, detail=f"File too large (max {max_size/1048576} MB)")
    await audio.seek(0)

    try:
        # Leer el contenido del archivo
        contents = await audio.read()
        audio_data = io.BytesIO(contents)

        # Procesar el audio
        y, sr = librosa.load(audio_data, sr=16000)

        # Obtener predicciones
        device = 'cpu'
        model_name = 'audeering/wav2vec2-large-robust-24-ft-age-gender'
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model = AgeGenderModel.from_pretrained(model_name)

        recognition_service_instance = RecognitionService(
            model=model,
            processor=processor,
            device=device
        )
        predictions = recognition_service_instance.process_func(y, sr)
        age_float = float(predictions[0][0])
        age = int(round(round(age_float, 2) * 100))

        genders_probability: [] = predictions[0][1:]
        gender = get_gender(genders_probability)

        response = AudioResponse(
            age=age,
            gender=gender
        )

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")
