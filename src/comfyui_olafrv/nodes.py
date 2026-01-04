from .nodes_text_encoders import ORvTextEncoderGoogleEmbeddingGemma3, ORvStringConsoleDebug, ORvTextStripNonLatin
from .nodes_image_analysis import ORvEmbeddingsHeatmap, ORvEmbeddingsSpectrogram, ORVImageSizeBestFitResolution

# A dictionary that contains all nodes you want to
# export with their names (should be globally unique)
NODE_CLASS_MAPPINGS = {
    "ORvTextEncoderGoogleEmbeddingGemma3": ORvTextEncoderGoogleEmbeddingGemma3,  # noqa: F403, F405
    "ORvStringConsoleDebug": ORvStringConsoleDebug,  # noqa: F403, F405
    "ORvTextStripNonLatin": ORvTextStripNonLatin,  # noqa: F403, F405
    "ORvEmbeddingsHeatmap": ORvEmbeddingsHeatmap,  # noqa: F403, F405
    "ORvEmbeddingsSpectrogram": ORvEmbeddingsSpectrogram,  # noqa: F403, F405
    "ORVImageSizeBestFitResolution": ORVImageSizeBestFitResolution,  # noqa: F403, F405
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "ORvTextEncoderGoogleEmbeddingGemma3": "ORv Text Encoder Google Embedding Gemma 3",
    "ORvStringConsoleDebug": "ORv String Console Debug",
    "ORvTextStripNonLatin": "ORv Text Strip Non-Latin Chars",
    "ORvEmbeddingsHeatmap": "ORv Embeddings Heatmap",
    "ORvEmbeddingsSpectrogram": "ORv Embeddings Spectrogram",
    "ORVImageSizeBestFitResolution": "ORV Image Size Best Fit Resolution",
}
