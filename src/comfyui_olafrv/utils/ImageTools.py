import torch
import torch.nn.functional as F
import numpy as np
import matplotlib as mpl
from PIL import Image
import io


@staticmethod
def tensor_to_comfyui_image(img: torch.Tensor) -> torch.Tensor:
    """
    Changes the order of dimensions in a PyTorch tensor.
    https://docs.comfy.org/custom-nodes/backend/images_and_masks
    https://docs.comfy.org/custom-nodes/backend/tensors
    Here’s why:
    * Original shape: PyTorch image tensors are usually in
      [batch, channels, height, width] or [N, C, H, W] format.
    * ComfyUI requirement: ComfyUI expects images in
      [batch, height, width, channels] or [N, H, W, C] format.
    The code:
    * tensor = tensor.permute(0, 2, 3, 1).contiguous()
      permute(0, 2, 3, 1) rearranges the axes:
      * 0 (batch) stays first,
      * 2 (height) moves to second,
      * 3 (width) moves to third,
      * 1 (channels) moves to last.
      * .contiguous() ensures the tensor’s memory layout
        matches the new order, which is important for efficient processing.
    """
    return img.permute(0, 2, 3, 1).contiguous()


@staticmethod
def tensor_to_bytes(img: torch.Tensor, format="PNG") -> bytes:
    """
    Convert a [1,H,W,C] torch tensor in [0,1] to a BytesIO buffer.
    """
    buf = io.BytesIO()
    img_pil = tensor_to_pil(img)
    img_pil.save(buf, format=format)
    return buf.getvalue()


@staticmethod
def tensor_to_pil(img: torch.Tensor) -> Image.Image:
    """
    Convert a [1,H,W,C] torch tensor in [0,1] to a PIL Image.
    """
    arr = img.cpu().numpy()
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]  # Remove batch dimension if present
    if arr.dtype != "uint8":
        arr = (arr * 255).clip(0, 255).astype("uint8")  # Convert to uint8 if needed

    img_pil = Image.fromarray(arr)

    return img_pil


@staticmethod
def tensor_to_heatmap(embedding: torch.Tensor) -> torch.Tensor:
    """
    Color heatmap visualization image of the embedding tensor
    - Reshape [1,768] embedding → (24,32)
    - Apply plasma colormap
    - Return [1,H,W,C] torch tensor, float32, in [0,1]
    """
    arr = embedding.detach().cpu().numpy().flatten().astype(np.float32)
    arr_norm = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

    grid = arr_norm.reshape(24, 32)  # reshape [768] → [24,32]

    # Use matplotlib colormap (plasma)
    cmap = mpl.colormaps.get_cmap("plasma")  # use matplotlib colormap
    colored = cmap(grid)[:, :, :3]  # RGBA → RGB

    # Convert to torch tensor in [0,1]
    tensor = torch.from_numpy(colored).permute(2, 0, 1).unsqueeze(0)  # [1,3,48,16]

    # Upscale with torch (bicubic), keep float32 precision
    tensor = F.interpolate(tensor, size=(512, 512), mode="bicubic", align_corners=False)

    return tensor_to_comfyui_image(tensor)  # [1,512,512,3], float32, in [0,1]


@staticmethod
def tensor_to_spectrogram(embedding: torch.Tensor) -> torch.Tensor:
    """
    Spectrogram-like visualization image ofr the embedding tensor
    - Reshape [1,768] embedding → (48,16)
    - Apply viridis colormap
    - Return [1,H,W,C] torch tensor, float32, in [0,1]
    """
    arr = embedding.detach().cpu().numpy().flatten().astype(np.float32)
    arr_norm = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

    # Reshape to a "spectrogram"-like 2D grid
    grid = arr_norm.reshape(48, 16)

    # Apply viridis colormap (values in [0,1])
    cmap = mpl.colormaps.get_cmap("viridis")
    colored = cmap(grid)[:, :, :3]  # RGBA → RGB, shape (48,16,3), values in [0,1]

    # Convert to torch tensor in [0,1]
    tensor = torch.from_numpy(colored).permute(2, 0, 1).unsqueeze(0)  # [1,3,48,16]

    # Upscale with torch (bicubic), keep float32 precision
    tensor = F.interpolate(tensor, size=(512, 512), mode="bicubic", align_corners=False)

    return tensor_to_comfyui_image(tensor)  # [1,512,512,3], float32, in [0,1]
