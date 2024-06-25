import os
import numpy as np
import pytest
from transformers import Wav2Vec2Processor
from assertpy import assert_that
from services.recognition_service import RecognitionService, AgeGenderModel


class TestRecognition:

    recognition_service = None

    @pytest.fixture(autouse=True)
    def setup(self):
        device = 'cpu'
        model_name = 'audeering/wav2vec2-large-robust-24-ft-age-gender'
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model = AgeGenderModel.from_pretrained(model_name)
        self.recognition_service = RecognitionService(
            model=model,
            processor=processor,
            device=device
        )

    def test_recognition_is_instance_of_np_array(self):
        audio_files = os.listdir('../audios')
        for audio_file in audio_files:
            audio, sr = self.recognition_service.load_and_process_audio_from_file_path(f'../audios/{audio_file}')
            predictions = self.recognition_service.process_func(audio, sr, embeddings=True)
            print("Predicciones (Edad, género femenino, género masculino, género infantil) Embeddings (primeros 5 valores:")
            embeddings = predictions[0][:5]
            print(f"Embeddings: \n {embeddings}")
            print(f"Predictions: \n {predictions}")
            assert_that(predictions).is_instance_of(np.ndarray)
            assert_that(embeddings).is_instance_of(np.ndarray)
