import importlib.util
import re
import sys
import types
import warnings

import torch
import torch.nn as nn


warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


if importlib.util.find_spec("sklearn") is None:
    class _FallbackOneHotEncoder:
        def __setstate__(self, state):
            self.__dict__.update(state)

        def transform(self, values):
            categories = [str(value) for value in self.categories_[0]]
            value = str(values[0][0])
            row = [0.0] * len(categories)
            if value in categories:
                row[categories.index(value)] = 1.0
            return [row]

    sklearn_module = types.ModuleType("sklearn")
    preprocessing_module = types.ModuleType("sklearn.preprocessing")
    encoders_module = types.ModuleType("sklearn.preprocessing._encoders")
    encoders_module.OneHotEncoder = _FallbackOneHotEncoder
    preprocessing_module._encoders = encoders_module
    sklearn_module.preprocessing = preprocessing_module
    sys.modules["sklearn"] = sklearn_module
    sys.modules["sklearn.preprocessing"] = preprocessing_module
    sys.modules["sklearn.preprocessing._encoders"] = encoders_module


class CharCNNWithClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_classes,
        emb_dim,
        num_filters,
        kernel_sizes,
        age_dim,
        hidden_dim,
        dropout=0.35,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=emb_dim,
                    out_channels=num_filters,
                    kernel_size=k,
                )
                for k in kernel_sizes
            ]
        )
        self.classifier = nn.Sequential(
            nn.Linear(num_filters * len(kernel_sizes) + age_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, text_ids, age_features):
        x = self.embedding(text_ids)
        x = x.transpose(1, 2)

        pooled = []
        for conv in self.convs:
            z = torch.relu(conv(x))
            z = torch.max(z, dim=2).values
            pooled.append(z)

        text_vec = torch.cat(pooled, dim=1)
        combined = torch.cat([text_vec, age_features], dim=1)
        return self.classifier(combined)


class VaccineCharCNNPredictor:
    def __init__(self, checkpoint_path, device=None):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.checkpoint = checkpoint
            self.state_dict = checkpoint["model_state_dict"]
        else:
            self.checkpoint = {}
            self.state_dict = checkpoint

        self.char2id = self.checkpoint.get("char2id", {})
        self.max_len = int(self.checkpoint.get("max_len", 200))
        self.classes = list(self.checkpoint.get("classes", []))
        self.age_encoder = self.checkpoint.get("age_bucket")
        self.age_dim = int(self.checkpoint.get("age_dim", 0))
        self.model = self._build_model()

    def _build_model(self):
        sd = self.state_dict

        vocab_size = sd["embedding.weight"].shape[0]
        emb_dim = sd["embedding.weight"].shape[1]
        conv_keys = sorted(
            [k for k in sd.keys() if re.fullmatch(r"convs\.\d+\.weight", k)],
            key=lambda x: int(x.split(".")[1]),
        )
        kernel_sizes = tuple(sd[k].shape[2] for k in conv_keys)
        num_filters = sd[conv_keys[0]].shape[0]
        hidden_dim = sd["classifier.0.weight"].shape[0]
        classifier_input_dim = sd["classifier.0.weight"].shape[1]
        conv_out_dim = num_filters * len(kernel_sizes)
        self.age_dim = classifier_input_dim - conv_out_dim
        num_classes = sd["classifier.3.weight"].shape[0]

        model = CharCNNWithClassifier(
            vocab_size=vocab_size,
            num_classes=num_classes,
            emb_dim=emb_dim,
            num_filters=num_filters,
            kernel_sizes=kernel_sizes,
            age_dim=self.age_dim,
            hidden_dim=hidden_dim,
            dropout=0.35,
        )
        model.load_state_dict(sd, strict=True)
        model.to(self.device)
        model.eval()
        return model

    def _encode_text(self, text):
        text = (text or "").lower()
        unk_id = self.char2id.get("<UNK>", self.char2id.get("<unk>", 1))
        ids = [self.char2id.get(ch, unk_id) for ch in text[: self.max_len]]
        ids += [0] * (self.max_len - len(ids))
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def _age_to_months(self, age_input):
        if age_input is None:
            return None

        raw = str(age_input).strip().lower()
        match = re.search(r"\d+(?:\.\d+)?", raw)
        if not match:
            return None

        value = float(match.group())
        if any(unit in raw for unit in ["เดือน", "month", "mo"]):
            return value
        if any(unit in raw for unit in ["ปี", "year", "yr"]):
            return value * 12
        return value * 12

    def _age_bucket(self, age_input):
        months = self._age_to_months(age_input)
        if months is None:
            return "unknown"

        if months == 0:
            return "แรกเกิด"
        if months == 1:
            return "1เดือน"
        if months == 2:
            return "2เดือน"
        if months == 4:
            return "4เดือน"
        if months == 6:
            return "6เดือน"
        if 9 <= months <= 12:
            return "9-12เดือน"
        if 13 <= months <= 17:
            return "13-17เดือน"
        if months == 18:
            return "18เดือน"
        if 24 <= months <= 30:
            return "2-2.5ปี"
        if 31 <= months <= 47:
            return "31-47เดือน"
        if 48 <= months <= 72:
            return "4-6ปี"
        if 73 <= months <= 131:
            return "73-131เดือน"
        if 132 <= months <= 155:
            return "11-12ปี"
        return "นอกช่วงตาราง"

    def _encoder_categories(self):
        if self.age_encoder is not None and hasattr(self.age_encoder, "categories_"):
            return [str(value) for value in self.age_encoder.categories_[0]]
        return [
            "11-12ปี",
            "13-17เดือน",
            "18เดือน",
            "1เดือน",
            "2-2.5ปี",
            "2เดือน",
            "31-47เดือน",
            "4-6ปี",
            "4เดือน",
            "6เดือน",
            "73-131เดือน",
            "9-12เดือน",
            "unknown",
            "นอกช่วงตาราง",
            "แรกเกิด",
        ]

    def _encode_age(self, age_input):
        months = self._age_to_months(age_input)
        bucket = self._age_bucket(age_input)

        if self.age_encoder is not None and hasattr(self.age_encoder, "transform"):
            encoded = self.age_encoder.transform([[bucket]])
            if hasattr(encoded, "toarray"):
                encoded = encoded.toarray()
            features = list(encoded[0])
        else:
            categories = self._encoder_categories()
            features = [1.0 if bucket == category else 0.0 for category in categories]

        age_months = 0.0 if months is None else months
        numeric_features = [age_months / 216.0, (age_months / 12.0) / 18.0]
        features = numeric_features + features

        if len(features) < self.age_dim:
            features += [0.0] * (self.age_dim - len(features))
        elif len(features) > self.age_dim:
            features = features[: self.age_dim]

        return torch.tensor([features], dtype=torch.float32, device=self.device)

    def predict_one(self, symptom_input, age_input, input_text="", return_top_k=3):
        text = " ".join(
            part.strip()
            for part in [symptom_input or "", input_text or ""]
            if part and part.strip()
        )
        text_ids = self._encode_text(text)
        age_features = self._encode_age(age_input)

        with torch.no_grad():
            logits = self.model(text_ids, age_features)
            probabilities = torch.softmax(logits, dim=1)[0]
            k = min(return_top_k, probabilities.numel())
            top_probs, top_indices = torch.topk(probabilities, k)

        labels = [
            self.classes[idx] if idx < len(self.classes) else str(idx)
            for idx in top_indices.tolist()
        ]
        return {
            "top_3_labels": labels,
            "top_3_probabilities": top_probs.cpu().tolist(),
        }
