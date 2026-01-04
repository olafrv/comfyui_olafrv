import torch
import pathlib
import folder_paths
import platform
from .utils.GoogleEmbeddingsGemma3 import GoogleEmbeddingsGemma3

CATEGORY = "olafrv/text_encoders"


def get_default_model_path() -> str:
    """Get the default model path for Google Embedding Gemma 3"""
    default_model_paths: list[str] = folder_paths.get_folder_paths("text_encoders")
    for path in default_model_paths:
        found_path = pathlib.Path(path)
        print(f"Checking path: {found_path}")
        if found_path.exists():
            final_found_path = found_path.joinpath("google", "embeddinggemma-300M")
            if final_found_path.exists():
                print(f"Found existing model path: {final_found_path}")
                return str(final_found_path)
    raise RuntimeError("ERROR: missing model path for Google Embedding Gemma 3")


default_model_path: str = get_default_model_path()
embedding_model: GoogleEmbeddingsGemma3 | None = None


class ORvTextEncoderGoogleEmbeddingGemma3:
    """Node to encode text using Google Embedding Gemma 3 model."""

    @classmethod
    def INPUT_TYPES(cls):
        global default_model_path
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "dynamicPrompts": False, "tooltip": "Text to be encoded into a pytorch tensor"})
            },
            "optional": {
                "model_path": ("STRING", {"default": default_model_path, "tooltip": "Path to the Google Embedding Gemma 3 model"})
            },
        }

    RETURN_TYPES = ("EMBEDDING", "STRING")
    OUTPUT_IS_LIST = (False, False)
    OUTPUT_TOOLTIPS = ("Embedding tensor from input text.", "Text representation of the embedding tensor.")
    FUNCTION = "encode"
    CATEGORY = CATEGORY
    DESCRIPTION = "Node to encode text using Google Embedding Gemma 3 model."

    def encode(self, text: str, model_path: str | None) -> tuple[torch.Tensor, str]:
        """
        Encode the input text into a embedding pytorch tensor using
        Google Embedding Gemma 3 model.
        """
        global embedding_model

        # Initialize the embedding model if not already done
        if embedding_model is None:
            if not model_path:
                model_path = default_model_path
            embedding_model = GoogleEmbeddingsGemma3(model_path)

        print(f"Input Text: {text}")

        # Calculate the embeddings tensor
        embedding_tensor = embedding_model.calculate_embeddings(text=text, convert_to_tensor=True)

        print(f"Embedding Tensor (Before Reshape): {embedding_tensor.shape}")

        # (Decided to) Follow the conditioning format used by ComfyUI whichs
        # typically expects shape [batch_size, sequence_length, embedding_dim]
        if embedding_tensor.dim() == 1:
            # Add batch and sequence dimensions [768] -> [1, 1, 768]
            embedding_tensor = embedding_tensor.unsqueeze(0).unsqueeze(0)
        elif embedding_tensor.dim() == 2:
            # Add batch dimension [N, 768] -> [1, N, 768]
            embedding_tensor = embedding_tensor.unsqueeze(0)

        print(f"Embedding Tensor (After Reshape): {embedding_tensor.shape}")

        return (
            embedding_tensor,
            repr(embedding_tensor),
        )


class ORvStringConsoleDebug:
    """Print to the ComfyUI console to debug text strings."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "See the ComfyUI console (>_) for the output.",
                        "dynamicPrompts": False,
                        "forceInput": True,
                        "tooltip": "Text to be logged to the console",
                    },
                )
            }
        }

    INPUT_IS_LIST = (False,)
    RETURN_TYPES = ("STRING",)
    FUNCTION = "log"
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (False,)
    OUTPUT_TOOLTIPS = ("The same text that was input.",)
    CATEGORY = CATEGORY
    DESCRIPTION = "Print to the ComfyUI console to debug text strings."

    def log(self, text: str) -> tuple[str,]:
        """
        Return the input text string.
        """
        print(f"{self.__class__.__name__} - Text:\n{text}")
        return (text,)


class ORvTextStripNonLatin:
    """Strip all non-Latin characters from the input text"""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"text": ("STRING", {"multiline": True, "dynamicPrompts": False, "tooltip": "Text to be processed"})}}

    RETURN_TYPES = ("STRING",)
    OUTPUT_IS_LIST = (False,)
    OUTPUT_TOOLTIPS = ("Text with non-Latin characters removed.",)
    FUNCTION = "strip_non_latin"
    CATEGORY = CATEGORY
    DESCRIPTION = "Strip all non-Latin characters from the input text (including Chinese, Japanese, Korean, Arabic, etc.)"

    def strip_non_latin(self, text: str) -> tuple[str,]:
        """
        Strip all non-Latin characters that cannot be encoded in cp1252/latin-1
        """

        # Only apply cp1252 encoding on Windows
        if platform.system() == "Windows":
            stripped_text = text.encode("cp1252", "ignore").decode("cp1252")
        else:
            # On non-Windows systems, use latin-1 (ISO-8859-1) which is similar to cp1252
            stripped_text = text.encode("latin-1", "ignore").decode("latin-1")

        # Log if characters were removed
        if len(text) != len(stripped_text):
            removed_count = len(text) - len(stripped_text)
            print(f"WARN: Removed {removed_count} non-Latin chars, resulted in text: '{stripped_text}'")

        # Replace multiple spaces with a single space
        stripped_text = " ".join(stripped_text.split())

        # Clean up comma artifacts like ", ,"
        stripped_text = stripped_text.replace(", ,", ",").strip()

        return (stripped_text,)
