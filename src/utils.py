import torch


def patched_resize_positional_embeddings(
    positional_embeddings, spatial_shapes, max_length
):
    """Resizes positional embeddings for flexible input resolutions."""
    batch_size = spatial_shapes.shape[0]
    embed_dim = positional_embeddings.shape[-1]
    source_dtype = positional_embeddings.dtype

    height, width = spatial_shapes[0, 0], spatial_shapes[0, 1]

    # Interpolate positional embeddings.
    pos_embeds = positional_embeddings.permute(2, 0, 1).unsqueeze(0)
    resized = torch.nn.functional.interpolate(
        pos_embeds, size=[height, width], mode="bilinear", align_corners=False
    )

    resized = (
        resized.reshape(1, embed_dim, -1).permute(0, 2, 1).to(source_dtype)
    )

    if batch_size > 1:
        resized = resized.expand(batch_size, -1, -1)

    padding_len = max_length - (height * width)
    padding = torch.zeros(
        batch_size,
        padding_len,
        embed_dim,
        device=positional_embeddings.device,
        dtype=source_dtype,
    )

    return torch.cat([resized, padding], dim=1)


def naflex_collate_fn(batch, processor):
    """Processes images and collates embeddings."""
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    images = [item["image"] for item in batch]
    text_embeds = [item["text_embedding"] for item in batch]

    inputs = processor(images=images, return_tensors="pt", padding=True)
    inputs["text_embedding"] = torch.stack(text_embeds)
    return inputs
