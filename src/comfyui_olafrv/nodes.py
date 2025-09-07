from .nodes_image_analysis import ORvEmbeddingsHeatmap, ORvEmbeddingsSpectrogram
from .nodes_text_encoders import ORvTextEncoderGoogleEmbeddingGemma3, ORvStringConsoleDebug

# A dictionary that contains all nodes you want to
# export with their names (should be globally unique)
NODE_CLASS_MAPPINGS = {
    "ORvTextEncoderGoogleEmbeddingGemma3": ORvTextEncoderGoogleEmbeddingGemma3,
    "ORvEmbeddingsHeatmap": ORvEmbeddingsHeatmap,
    "ORvEmbeddingsSpectrogram": ORvEmbeddingsSpectrogram,
    "ORvStringConsoleDebug": ORvStringConsoleDebug,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "ORvTextEncoderGoogleEmbeddingGemma3": "ORv Text Encoder Google Embedding Gemma 3",
    "ORvEmbeddingsHeatmap": "ORv Embeddings Heatmap",
    "ORvEmbeddingsSpectrogram": "ORv Embeddings Spectrogram",
    "ORvShowText": "ORv Show Text",
}
