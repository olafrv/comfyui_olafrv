import math
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


class ORVImageSizeBestFitResolution:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE", {"tooltip": "Input image to calculate best fit SDXL resolution", "forceInput": True})}}

    RETURN_TYPES = (
        "INT",
        "INT",
        "INT",
        "INT",
    )
    RETURN_NAMES = (
        "width",
        "height",
        "min_dimension",
        "max_dimension",
    )
    OUTPUT_IS_LIST = (
        False,
        False,
        False,
        False,
    )
    OUTPUT_TOOLTIPS = ("Get the next SDXL resolution fitting an image, aspect ratio and orientation.",)
    FUNCTION = "get_best_fit_resolution"
    CATEGORY = CATEGORY
    DESCRIPTION = "Get the next SDXL resolution fitting an image, aspect ratio and orientation."

    def get_best_fit_resolution(self, image):
        """
        Calculate the next fit SDXL resolution for the given image.
        https://platform.stability.ai/docs/api-reference#tag/SDXL-1.0
        The resolution of the generated image will be 1 megapixel (1048576 pixels).
        The image will be resized to fit the given aspect ratio and orientation.
        Returns: width, height, min_dimension, max_dimension.
        """
        PIXEL_MAX = 1048576  # 1 Megapixel
        PIXEL_INC = 64

        # Get image dimensions (ComfyUI images are in format [batch, height, width, channels])
        batch, height, width, channels = image.shape

        # Calculate aspect ratio
        aspect_ratio = width / height

        # Ensure maximum dimensions conserving aspect ratio
        if width > 1536:
            width = 1536
            height = int(width / aspect_ratio)
        if height > 1536:
            height = 1536
            width = int(height * aspect_ratio)

        # Ensure minimum dimensions conserving aspect ratio
        if width < 640:
            width = 640
            height = int(width / aspect_ratio)
        if height < 640:
            height = 640
            width = int(height * aspect_ratio)

        # Round to nearest PIXEL_INC while preserving aspect ratio
        width_inc_count = math.ceil(width / PIXEL_INC)
        height_inc_count = math.ceil(height / PIXEL_INC)

        # Ensure the total pixel count does not exceed 1 Megapixel
        while (width_inc_count * PIXEL_INC) * (height_inc_count * PIXEL_INC) > PIXEL_MAX:
            # Reduce the larger dimension to preserve aspect ratio better
            if width_inc_count >= height_inc_count:
                width_inc_count -= 1
            else:
                height_inc_count -= 1

        # Calculate new width and height
        new_width = width_inc_count * PIXEL_INC
        new_height = height_inc_count * PIXEL_INC

        # Calculate min and max dimensions
        min_dimension = min(new_width, new_height)
        max_dimension = max(new_width, new_height)

        return (
            new_width,
            new_height,
            min_dimension,
            max_dimension,
        )
