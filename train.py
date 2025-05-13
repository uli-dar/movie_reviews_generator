import sys
import warnings
import torch
from tqdm import tqdm
from loguru import logger

logger.remove()
logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
warnings.filterwarnings("ignore")


def evaluate(model, criterion, test_dataloader) -> float:
    model.eval()
    perplexity = []
    with torch.no_grad():
        for batch in test_dataloader:
            logits = model(batch["input_ids"]).flatten(start_dim=0, end_dim=1)
            loss = criterion(logits, batch["target_ids"].flatten())
            perplexity.append(torch.exp(loss).item())
    perplexity = sum(perplexity) / len(perplexity)
    return perplexity


def train_model(
    model, model_name, train_dataloader, test_dataloader, epochs, optimizer, criterion, writer
):
    losses = []
    perplexity = []
    best_perplexity = 1e4
    best_model = None
    for epoch in tqdm(range(epochs)):
        epoch_losses = []
        model.train()
        for batch in tqdm(train_dataloader, desc=f"Training epoch {epoch + 1}"):
            optimizer.zero_grad()
            logits = model(batch["input_ids"]).flatten(start_dim=0, end_dim=1)
            loss = criterion(logits, batch["target_ids"].flatten())
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        losses.append(sum(epoch_losses) / len(epoch_losses))
        writer.add_scalars(
            "Train Loss", {f"{model_name}": sum(epoch_losses) / len(epoch_losses)}, epoch
        )

        current_perplexiry = evaluate(model, criterion, test_dataloader)
        perplexity.append(current_perplexiry)
        writer.add_scalars(
            "Test perplexity", {f"{model_name}": current_perplexiry}, epoch
            )

        if current_perplexiry < best_perplexity:
            best_perplexity = current_perplexiry
            best_model = model

    return losses, perplexity, best_model
