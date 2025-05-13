import sys
import mlflow
import warnings
from loguru import logger
import argparse

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import utils
from train import train_model
import rnn_models

logger.remove()
logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
warnings.filterwarnings("ignore")
mlflow.set_tracking_uri(uri="http://127.0.0.1:8081")
mlflow.set_experiment("Movie_Review_Gen")


def get_data(run_id: str, device: torch.device):
    # loading data after data prep
    train_ds = mlflow.artifacts.download_artifacts(
        artifact_uri=f"runs:/{run_id}/train.npy"
    )
    train_ds = np.load("data/train.npy", allow_pickle=True)
    test_ds = mlflow.artifacts.download_artifacts(
        artifact_uri=f"runs:/{run_id}/test.npy"
    )
    test_ds = np.load("data/test.npy", allow_pickle=True)
    print(f"{train_ds.shape}, {test_ds.shape=}")
    metadata = mlflow.artifacts.download_artifacts(
        artifact_uri=f"runs:/{run_id}/metadata.csv"
    )
    metadata = pd.read_csv("data/metadata.csv", index_col=0)

    # getting the hyperparams
    metadata = pd.pivot_table(data=metadata, columns="0", values="1")
    pad_id = int(metadata["pad_id"].values[0])
    batch_size = int(metadata["batch_size"].values[0])
    custom_collidor = utils.Collator_with_padding(pad_id=pad_id, device=device)

    train_dataloader = DataLoader(
        train_ds, collate_fn=custom_collidor, batch_size=batch_size
    )
    test_dataloader = DataLoader(
        test_ds, collate_fn=custom_collidor, batch_size=batch_size
    )
    return train_dataloader, test_dataloader, metadata


if __name__ == "__main__":
    utils.seed_everything(42)

    device = "cuda" if torch.cuda.is_available() else "mps"
    writer = SummaryWriter("runs/Movie_Review_Gen")
    print(device)
    with mlflow.start_run() as run:
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", type=str)
        parser.add_argument("--epochs", default=15, type=int)
        parser.add_argument("--hidden_dim", default=256, type=int)

        data_prep_run_id = "id"
        train_dataloader, test_dataloader, metadata = get_data(data_prep_run_id, device)
        logger.info("Train & test dataloaders are ready for training")

        params = {}
        params["epochs"] = parser.parse_args().epochs
        model_name = parser.parse_args().model
        params["hidden_dim"] = parser.parse_args().hidden_dim
        params["vocab_size"] = int(metadata["vocab_size"].values[0])
        params["batch_size"] = int(metadata["batch_size"].values[0])
        if "GRU" in model_name:
            model = rnn_models.GRU_LanguageModel(
                hidden_dim=params["hidden_dim"], vocab_size=params["vocab_size"]
            ).to(device)
        else:
            model = rnn_models.LSTM_LanguageModel(
                hidden_dim=params["hidden_dim"], vocab_size=params["vocab_size"]
            ).to(device)

        criterion = nn.CrossEntropyLoss(ignore_index=int(metadata["pad_id"].values[0]))
        optimizer = torch.optim.Adam(model.parameters())
        logger.info("Model initialized")

        mlflow.set_tag("Model type", f"{model_name}")
        logger.info("Training started")
        loss_history, metric_history, best_model = train_model(
            model,
            train_dataloader,
            test_dataloader,
            params["epochs"],
            optimizer,
            criterion,
            writer,
        )

        mlflow.log_metric("Train loss", min(loss_history))
        mlflow.log_metric("Test perplexity", min(metric_history))

        mlflow.pytorch.log_model(best_model, model_name)

        writer.close()
