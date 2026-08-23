"""Primary inference backend: audeering/wav2vec2-large-robust-24-ft-age-gender.

Why this model (see README "Model choice rationale" for the fuller version):
  * It's a wav2vec2-large-robust encoder specifically fine-tuned for age
    regression + 3-way gender classification (female/male/child) on top of
    speech, as opposed to a general-purpose SER/ASR model repurposed for
    this. "robust" in the base checkpoint name means it was pretrained
    across multiple noisy/clean corpora, which matters for our
    truck-cab/warehouse-floor audio conditions.
  * It ships pretrained weights on the Hugging Face Hub (a "publicly
    available model weight" per the assignment's portability constraint) —
    no training pipeline or labeled data of our own required.
  * It runs acceptably on CPU for a single ~5s clip (a few hundred ms after
    the encoder is warm), which fits the <500ms latency target without
    needing GPU infrastructure.

The class definitions below (ModelHead / AgeGenderModel) are copied from the
model card's own example code, since this checkpoint uses a custom
classification head that isn't loadable via a generic AutoModel class.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np

from app.config import settings

logger = logging.getLogger("voiceattr.model")

# Age bracket boundaries, in years, mapped from the model's continuous
# age output (which is trained on CommonVoice-derived age labels).
_AGE_BRACKETS = [
    (0, 30, "18-30"),
    (30, 45, "31-45"),
    (45, 60, "46-60"),
    (60, 200, "60+"),
]

_GENDER_LABELS = ("female", "male", "child")


def years_to_bracket(age_years: float) -> str:
    for lo, hi, label in _AGE_BRACKETS:
        if lo <= age_years < hi:
            return label
    return "60+"


@dataclass
class ModelPrediction:
    gender_prediction: str
    gender_confidence: float
    age_prediction: str
    age_confidence: float
    age_years_estimate: float
    raw_gender_probs: dict[str, float]
    backend: str = "wav2vec2-age-gender"


class ModelLoadError(Exception):
    pass


class AgeGenderModel:
    """Lazy-loaded, process-wide singleton wrapping the HF model + processor.

    Thread-safety: a lock serializes the (rare) load, and a bounded
    semaphore caps concurrent forward passes (see config.INFERENCE_CONCURRENCY)
    so a burst of requests degrades to queueing rather than starving CPU
    threads against each other.
    """

    _instance: "AgeGenderModel | None" = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._load_lock = threading.Lock()
        self._sem = threading.BoundedSemaphore(settings.INFERENCE_CONCURRENCY)
        self._model = None
        self._processor = None
        self._device = None
        self._loaded = False
        self._load_error: str | None = None

    @classmethod
    def get(cls) -> "AgeGenderModel":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            self._do_load()

    def _do_load(self) -> None:
        try:
            import torch
            from transformers import Wav2Vec2Processor
            from transformers.models.wav2vec2.modeling_wav2vec2 import (
                Wav2Vec2Model,
                Wav2Vec2PreTrainedModel,
            )
        except ImportError as exc:  # pragma: no cover - exercised only when torch/transformers absent
            self._load_error = f"torch/transformers not available: {exc}"
            raise ModelLoadError(self._load_error) from exc

        import torch.nn as nn

        class ModelHead(nn.Module):
            """Classification head (identical to the model card's reference impl)."""

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

        class AgeGenderHFModel(Wav2Vec2PreTrainedModel):
            def __init__(self, config):
                super().__init__(config)
                self.config = config
                self.wav2vec2 = Wav2Vec2Model(config)
                self.age = ModelHead(config, 1)
                self.gender = ModelHead(config, 3)
                self.init_weights()

            def forward(self, input_values):
                outputs = self.wav2vec2(input_values)
                hidden_states = outputs[0]
                hidden_states = torch.mean(hidden_states, dim=1)
                logits_age = self.age(hidden_states)
                logits_gender = torch.softmax(self.gender(hidden_states), dim=1)
                return hidden_states, logits_age, logits_gender

        try:
            device = settings.MODEL_DEVICE
            if device == "cuda" and not torch.cuda.is_available():
                logger.warning("MODEL_DEVICE=cuda requested but no CUDA device found; using cpu")
                device = "cpu"

            logger.info("loading model %s on %s (first run downloads weights)", settings.AGE_GENDER_MODEL_NAME, device)
            processor = Wav2Vec2Processor.from_pretrained(settings.AGE_GENDER_MODEL_NAME)
            model = AgeGenderHFModel.from_pretrained(settings.AGE_GENDER_MODEL_NAME)
            model.to(device)
            model.eval()

            self._torch = torch
            self._processor = processor
            self._model = model
            self._device = device
            self._loaded = True
            self._load_error = None
            logger.info("model loaded successfully")
        except Exception as exc:  # noqa: BLE001 - want to convert *any* load failure to a typed error
            self._load_error = str(exc)
            logger.exception("failed to load age/gender model")
            raise ModelLoadError(str(exc)) from exc

    def predict(self, samples: np.ndarray, sample_rate: int) -> ModelPrediction:
        if not self._loaded:
            self.ensure_loaded()

        with self._sem:
            torch = self._torch
            y = self._processor(samples, sampling_rate=sample_rate)
            y = y["input_values"][0]
            y = y.reshape(1, -1)
            y = torch.from_numpy(np.asarray(y, dtype=np.float32)).to(self._device)

            with torch.no_grad():
                _, logits_age, logits_gender = self._model(y)

            age_norm = float(logits_age.detach().cpu().numpy().reshape(-1)[0])
            gender_probs = logits_gender.detach().cpu().numpy().reshape(-1)

        # The model's age head is trained to output age/100.
        age_years = max(0.0, age_norm * 100.0)
        age_bracket = years_to_bracket(age_years)
        # Confidence for a regression output is approximated by how close
        # the point estimate sits to the *nearest* bracket boundary: a
        # prediction of 29.8y for the 18-30 bracket is much more confident
        # than 30.4y for 31-45. This is a heuristic calibration, not a
        # property the model was trained to expose (see README > limitations).
        boundaries = [b[0] for b in _AGE_BRACKETS[1:]]  # [30, 45, 60]
        dist_to_boundary = min(abs(age_years - b) for b in boundaries) if boundaries else 10.0
        age_confidence = float(np.clip(0.5 + dist_to_boundary / 40.0, 0.35, 0.95))

        gender_idx = int(np.argmax(gender_probs))
        gender_label_raw = _GENDER_LABELS[gender_idx]
        gender_confidence = float(gender_probs[gender_idx])
        raw_gender_probs = {label: float(p) for label, p in zip(_GENDER_LABELS, gender_probs)}

        # API contract only exposes male/female/unknown. A "child" call (or
        # a low-confidence adult call) is surfaced as unknown rather than
        # forced into male/female, since misgendering with false confidence
        # is worse than admitting uncertainty in a customer-facing call.
        if gender_label_raw == "child" or gender_confidence < 0.45:
            gender_prediction = "unknown"
        else:
            gender_prediction = gender_label_raw

        return ModelPrediction(
            gender_prediction=gender_prediction,
            gender_confidence=round(gender_confidence, 4),
            age_prediction=age_bracket,
            age_confidence=round(age_confidence, 4),
            age_years_estimate=round(age_years, 1),
            raw_gender_probs=raw_gender_probs,
        )
