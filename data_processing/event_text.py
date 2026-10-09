"""Encode original source-event text with the frozen released CLIP/PCA recipe."""

import json
from pathlib import Path

import numpy as np
import torch


class EventTextEncoder:
    def __init__(self, project, output):
        self.output = Path(output)
        embedding_root = (
            Path(project)
            / "datasets/annotations/frankenstein-dataset/text_embeddings/clip"
        )
        self.embeddings = np.load(embedding_root / "clip.npy", mmap_mode="r")
        self.slices = np.load(embedding_root / "clip_slice.npy")
        self.index = json.loads((embedding_root / "clip_index.json").read_text())
        self.components = np.load(
            Path(project) / "outputs_amass/unified_direct_20261008/data/pca.npz"
        )
        official = json.loads(
            (Path(project) / "pretrained/official/config.json").read_text()
        )
        self.parts = official["data"]["text_encoder"]["body_part_order"]
        self.new_embeddings = {}
        self.model = None
        self.parity = {}

    def cached_embedding(self, label):
        start, stop = self.slices[self.index[label]]
        value = np.asarray(self.embeddings[start:stop], dtype=np.float32)
        assert value.shape == (1, 512)
        return value[0]

    def encode(self, label):
        if label in self.index:
            value = self.cached_embedding(label)
        else:
            if label not in self.new_embeddings:
                import clip

                device = "cuda" if torch.cuda.is_available() else "cpu"
                if self.model is None:
                    checkpoint = "/mnt/sda2/dataset/mdm_refine_phc_mvp_v1/assets/clip/ViT-B-32.pt"
                    self.model, _ = clip.load(checkpoint, device=device)
                    self.model.eval()
                    with torch.no_grad():
                        for reference in ["kick", "wave"]:
                            encoded = (
                                self.model.encode_text(
                                    clip.tokenize([reference]).to(device)
                                )
                                .float()
                                .cpu()
                                .numpy()[0]
                            )
                            cached = self.cached_embedding(reference)
                            cosine = float(
                                np.dot(encoded, cached)
                                / (np.linalg.norm(encoded) * np.linalg.norm(cached))
                            )
                            assert cosine > 0.9999
                            self.parity[reference] = cosine
                with torch.no_grad():
                    self.new_embeddings[label] = (
                        self.model.encode_text(clip.tokenize([label]).to(device))
                        .float()
                        .cpu()
                        .numpy()[0]
                    )
            value = self.new_embeddings[label]
        reduced = (value - self.components["mean"]) @ self.components["components"].T
        return value, reduced

    def make_arrays(self, label, source_frames, start_time, end_time, active_parts):
        global_text, reduced_text = self.encode(label)
        local_text = np.zeros((len(source_frames), 408), np.float32)
        local_mask = np.zeros_like(local_text, dtype=bool)
        active = (source_frames / 20 >= start_time) & (source_frames / 20 < end_time)
        for part_index, part in enumerate(self.parts):
            if part in active_parts:
                local_text[active, part_index * 51 : (part_index + 1) * 51] = (
                    reduced_text
                )
                local_mask[active, part_index * 51 : (part_index + 1) * 51] = True
        return global_text, local_text, local_mask

    def save(self):
        labels = sorted(self.new_embeddings)
        np.savez(
            self.output / "new_event_embeddings.npz",
            **{
                str(label_index): self.new_embeddings[label]
                for label_index, label in enumerate(labels)
            }
        )
        (self.output / "new_event_embeddings.json").write_text(
            json.dumps(dict(labels=labels, cached_cosine_parity=self.parity), indent=2)
        )
