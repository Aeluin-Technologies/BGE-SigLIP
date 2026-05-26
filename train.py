import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from transformers import AutoProcessor
from src.model import SiglipBgeAligner
from src.dataset import FlickrAlignDataset
from src.utils import naflex_collate_fn
from tqdm import tqdm


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE, EPOCHS, LR = 64, 3, 1e-4

    df = pd.read_csv("./flickr30k-dataset/captions.txt")
    processor = AutoProcessor.from_pretrained(
        "google/siglip2-so400m-patch16-naflex"
    )
    model = SiglipBgeAligner().to(device)

    for param in model.vision_model.parameters():
        param.requires_grad = False
    for param in model.vision_model.encoder.layers[-2:].parameters():
        param.requires_grad = True

    dataset = FlickrAlignDataset(df, None, "./flickr30k-dataset/Images")
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: naflex_collate_fn(b, processor),
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )
    scaler = GradScaler("cuda")

    model.train()
    for epoch in range(EPOCHS):
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
            optimizer.zero_grad(set_to_none=True)
            text_embeds = batch.pop("text_embedding").to(device)
            vision_inputs = {k: v.to(device) for k, v in batch.items()}

            with autocast("cuda"):
                image_embeds = model(**vision_inputs)
                logit_scale = model.logit_scale.exp()
                logits = (
                    logit_scale * (image_embeds @ text_embeds.t())
                    + model.logit_bias
                )
                loss = (
                    -torch.nn.functional.logsigmoid(
                        torch.eye(logits.size(0), device=device) * logits
                    )
                    .sum(dim=-1)
                    .mean()
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    torch.save(model.state_dict(), "model_checkpoint.pt")
    print("Training complete. Checkpoint saved.")


if __name__ == "__main__":
    train()
