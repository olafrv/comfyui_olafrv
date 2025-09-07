import torch
from .utils.ImageTools import tensor_to_heatmap, tensor_to_spectrogram

CATEGORY = "olafrv/image_analysis"


class ORvEmbeddingsHeatmap:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"embedding": ("EMBEDDING", {"tooltip": "Input embedding to convert to heatmap image", "forceInput": True})}}

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (False,)
    OUTPUT_TOOLTIPS = ("Heatmap representation of the input embedding.",)
    FUNCTION = "to_heatmap"
    CATEGORY = CATEGORY
    DESCRIPTION = "Converts an input embedding to a heatmap representation."

    def to_heatmap(self, embedding) -> tuple[torch.Tensor]:
        """
        Convert the input embedding to a heatmap representation.
        """
        heatmap_image = tensor_to_heatmap(embedding)
        return (heatmap_image,)


class ORvEmbeddingsSpectrogram:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"embedding": ("EMBEDDING", {"tooltip": "Input embedding to convert to spectrogram image", "forceInput": True})}
        }

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (False,)
    OUTPUT_TOOLTIPS = ("Spectrogram representation of the input embedding.",)
    FUNCTION = "to_spectrogram"
    CATEGORY = CATEGORY
    DESCRIPTION = "Converts an input embedding to a spectrogram representation."

    def to_spectrogram(self, embedding) -> tuple[torch.Tensor]:
        """
        Convert the input embedding to a spectrogram representation.
        """
        spectrogram_image = tensor_to_spectrogram(embedding)
        return (spectrogram_image,)
