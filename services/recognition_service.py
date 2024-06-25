import librosa
import numpy as np
import torch
import torch.nn as nn
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)
import transformers

transformers.logging.set_verbosity_error()  # Supress warnings


class ModelHead(nn.Module):
    r"""Classification head."""

    def __init__(self, config, num_labels):
        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x


class AgeGenderModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = ModelHead(config, 1)
        self.gender = ModelHead(config, 3)
        self.init_weights()

    def forward(
            self,
            input_values,
    ):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits_age = self.age(hidden_states)
        logits_gender = torch.softmax(self.gender(hidden_states), dim=1)

        return hidden_states, logits_age, logits_gender


class RecognitionService:
    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device

    def __call__(self, x: np.ndarray, sampling_rate: int, embeddings: bool = False) -> np.ndarray:
        return self.process_func(x, sampling_rate, embeddings)

    @staticmethod
    def load_and_process_audio_from_file_path(file_path, target_sr=16000):
        audio_signal, sampling_rate = librosa.load(file_path, sr=None)

        if sampling_rate != target_sr:
            audio_signal = librosa.resample(audio_signal, orig_sr=sampling_rate, target_sr=target_sr)

        return audio_signal, target_sr

    def process_func(self, x: np.ndarray, sampling_rate: int, embeddings: bool = False) -> np.ndarray:
        r"""
        Predict age and gender or extract embeddings from raw audio signal.

        Args:
            x (np.ndarray): The raw audio signal.
            sampling_rate (int): The sampling rate of the audio signal.
            embeddings (bool, optional): If True, the function will return the embeddings. Defaults to False.

        Returns:
            np.ndarray: If embeddings is False, the function returns a numpy array with the following structure:
                        [[age, gender_female, gender_male, gender_child]]
                        Example: [[0.33793038, 0.2715511, 0.2275236, 0.5009253]]
                        If embeddings is True, the function returns the pooled hidden states of the last transformer layer.
                        Example: [[0.024444, 0.0508722, 0.04930823, ..., 0.07247854, -0.0697901, -0.0170537]]
        """

        # run through processor to normalize signal
        # always returns a batch, so we just get the first entry
        # then we put it on the device
        y = self.processor(x, sampling_rate=sampling_rate)
        y = y['input_values'][0]
        y = y.reshape(1, -1)
        y = torch.from_numpy(y).to(self.device)

        # run through model
        with torch.no_grad():
            y = self.model(y)
            if embeddings:
                y = y[0]
            else:
                y = torch.hstack([y[1], y[2]])

        # convert to numpy
        y = y.detach().cpu().numpy()

        return y
