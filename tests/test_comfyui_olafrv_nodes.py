#!/usr/bin/env python

"""Tests for `comfyui_olafrv` package."""

import os
import torch
import pytest
import hashlib
from src.comfyui_olafrv.nodes_text_encoders import ORvTextEncoderGoogleEmbeddingGemma3
from src.comfyui_olafrv.nodes_image_analysis import ORvEmbeddingsHeatmap, ORvEmbeddingsSpectrogram
from src.comfyui_olafrv.utils.ImageTools import tensor_to_pil, tensor_to_bytes
import catimage


@pytest.fixture
def text_encoder():
    """Create node instance."""
    return ORvTextEncoderGoogleEmbeddingGemma3()


@pytest.fixture
def embedding_heightmap():
    """Create node instance."""
    return ORvEmbeddingsHeatmap()


@pytest.fixture
def embedding_spectrogram():
    """Create node instance."""
    return ORvEmbeddingsSpectrogram()


def print_img(filename: str, img: torch.Tensor):
    print(f"\nshape: {img.shape}, dtype: {img.dtype}")
    img_pil = tensor_to_pil(img)

    curdir = os.path.dirname(os.path.abspath(__file__))
    output_filename = os.path.join(curdir, f"{filename}.png")
    img_pil.save(output_filename)
    print(f"file: {output_filename}")

    img_bytes = tensor_to_bytes(img, format="PNG")
    hash_val = hashlib.sha256(img_bytes).hexdigest()
    print(f"hash(buffer): {hash_val}")

    print("preview with 'catimage' library:")
    print(catimage.generateHDColour(output_filename, maxLen=48, trueColour=True, char="\u2584"))


def test_text_encoder_initialization(text_encoder):
    """Test that the node can be instantiated."""
    assert isinstance(text_encoder, ORvTextEncoderGoogleEmbeddingGemma3)


def test_text_encoder(text_encoder, embedding_heightmap, embedding_spectrogram):
    """Test the the node metadata, input types/output types and methods."""

    # Verify required structure
    input_types = text_encoder.INPUT_TYPES()
    print(f"Input types: {input_types}")
    assert "required" in input_types
    assert "text" in input_types["required"]
    assert "model_path" in input_types["optional"]

    # Verify metadata
    assert text_encoder.RETURN_TYPES == ("EMBEDDING", "STRING")
    assert text_encoder.OUTPUT_IS_LIST == (False, False)
    assert text_encoder.FUNCTION == "encode"
    assert text_encoder.CATEGORY == "olafrv/text_encoders"

    # Test multiline text input
    text = "woman\nin a\nlagoon"
    print(f"Test text: {repr(text)}")

    # Note: This test will fail if the model is not available locally
    (embedding, string) = text_encoder.encode(text=text, model_path=None)

    assert isinstance(embedding, torch.Tensor)
    print(f"Embedding shape: {embedding.shape}, dtype: {embedding.dtype}")

    assert isinstance(string, str)
    print(f"String output: {string}")

    embedding_heightmap_output = embedding_heightmap.to_heatmap(embedding)
    assert isinstance(embedding_heightmap_output, tuple)
    print_img("test_heatmap", embedding_heightmap_output[0])

    embedding_spectrogram_output = embedding_spectrogram.to_spectrogram(embedding)
    assert isinstance(embedding_spectrogram_output, tuple)
    print_img("test_spectrogram", embedding_spectrogram_output[0])

    print("Node test completed!")
