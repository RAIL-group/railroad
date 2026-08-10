import random
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from interruption.learning.data import CSVPickleDataset
from interruption.learning.utils import prepare_gcn_input, split_dataset
from interruption.learning.models.gcn import AnticipateGCN
from railroad.environment.procthor.resources import get_procthor_10k_dir, DEFAULT_RESOURCES_BASE

# setting of environment variables
# TODO - is this needed?
# os.environ["TOKENIZERS_PARALLELISM"] = "true"

# dataset specifications
TRAIN_DATASET_PATH = get_procthor_10k_dir() / "procthor_data_201.csv"
TEST_DATASET_PATH = None

# if supplying only 1 dataset, specify how it should be split for train and test datasets
TRAIN_TEST_SPLIT = 1

# specify learning hyperparameters
HYPERPARAMETERS = {
    "num_epochs": 500,
    "learning_rate": 0.01,
    "batch_size": 8,
    "lr_decay_factor": 0.5,
    "early_stopping_num_epochs": 100,
    "num_lrs": 3
}

# output directories specifications
EXPERIMENT_NAME = "experiment6"
LOG_DIRECTORY = DEFAULT_RESOURCES_BASE / f"run_logs/{EXPERIMENT_NAME}"
OUTPUT_MODEL_DIRECTORY = DEFAULT_RESOURCES_BASE / "models"


def main() -> None:
    """
    Training pipeline for interruption-based anticipatory planning.
    """
    os.makedirs(OUTPUT_MODEL_DIRECTORY, exist_ok=True)

    # setup TensorBoard logging
    writer = SummaryWriter(LOG_DIRECTORY)

    # setup dataloaders for specified dataset
    train_loader, test_loader = _get_train_test_dataloaders()

    # load model
    device_str = "cpu"
    if torch.cuda.is_available():
        device_str = "cuda"
    elif torch.backends.mps.is_available():
        device_str = "mps"
    device = torch.device(device_str)
    model = AnticipateGCN().to(device)

    # setup optimizer and scheduler
    optimizer = torch.optim.Adagrad(
        model.parameters(),
        lr=HYPERPARAMETERS["learning_rate"]
    )

    # decay the learning rate every X epochs
    decay_step_size = int(HYPERPARAMETERS["num_epochs"] / HYPERPARAMETERS["num_lrs"])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=decay_step_size,
        gamma=HYPERPARAMETERS["lr_decay_factor"]
    )

    # TODO - what is an anomaly when it comes to gradients?
    # torch.autograd.set_detect_anomaly(False)

    training_loop(model, device, optimizer, scheduler, (train_loader, test_loader), writer)


# helper functions
def training_loop(
    model: AnticipateGCN,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    data_loaders: tuple[DataLoader, DataLoader],
    writer: SummaryWriter,
) -> None:
    """
    Training loop for interruption-based anticipatory planning, where the actual 
    training of a GCN occurs.
    """
    train_loader, test_loader = data_loaders
    # keep track of lowest losses seen during training for early stopping
    lowest_training_loss = float("inf")
    # lowest_validation_loss = float("inf")
    early_stopping_counter = 0

    # epoch-based training loop
    for epoch in range(1, int(HYPERPARAMETERS["num_epochs"])+1):
        train_loss, error_flag = train_epoch(
            model,
            device,
            optimizer,
            train_loader,
            writer,
            epoch
        )
        # validate training loss
        if error_flag != -1:
            print(f"NaN or Inf detected in train loss at batch {error_flag} of epoch {epoch}")
            break
        scheduler.step()
        # update early stopping counter
        early_stopping_counter+=1

        # check model's performance on the validation set
        val_loss, error_flag = evaluate(model, device, test_loader)
        # log validation loss and learning rate
        writer.add_scalar("EpochLoss/val_total_loss", val_loss, epoch)
        writer.add_scalar("LR/learning_rate", scheduler.get_last_lr()[-1], epoch)

        # check validation loss
        if error_flag != -1:
            print(f"NaN or Inf detected in validation loss at batch {error_flag} of epoch {epoch}")
            break

        print(f"Epoch {epoch:05d} | avg train loss {train_loss:.4f}| avg val loss {val_loss:.4f}")

        # TODO - later on it might be better to only evaluate on the validation loss
        # for right now, overfitting is chill
        if train_loss < lowest_training_loss:
            lowest_training_loss = train_loss
            early_stopping_counter = 0
            # write out best model
            torch.save(
                model.state_dict(),
                OUTPUT_MODEL_DIRECTORY / f"best_model_{EXPERIMENT_NAME}.pt"
            )

        # check for early stoppage
        if early_stopping_counter >= HYPERPARAMETERS["early_stopping_num_epochs"]:
            print(f"Early stopping at epoch {epoch}")
            break


def train_epoch(
    model: AnticipateGCN,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    writer: SummaryWriter,
    epoch: int
) -> tuple[float, int]:
    """
    Helper function for performing one training pass over the whole training dataset(epoch).
    Returns the average loss for the epoch, or NaN/Inf on error.
    """
    error_flag = -1
    model.train()
    total_loss = 0.0
    # iterate over the training dataset in batches
    for i, batch in enumerate(train_loader):
        batch = batch.to(device)
        batch_data = _convert_batch_format(batch)
        optimizer.zero_grad()
        # expected value preds
        out = model.forward(batch_data, device)
        # mean absolute error for batch
        loss = model.loss(out, batch, str(device), None, i)
        # log the total_loss for the batch
        batch_total_loss = loss.item() * len(batch)
        writer.add_scalar(
            "BatchLoss/train_total_loss",
            batch_total_loss,
            epoch * len(train_loader) + i
        )

        # validate compute loss of the batch
        if not torch.isfinite(loss):
            error_flag = i
            break

        loss.backward()
        optimizer.step()
        # update total loss for epoch
        total_loss += batch_total_loss

    return total_loss / len(train_loader.dataset), error_flag # type: ignore


@torch.no_grad()
def evaluate(
    model: AnticipateGCN,
    device: torch.device,
    test_loader: DataLoader
) -> tuple[float, int]:
    """
    Helper function for evaluating the model during training on an epoch of the 
    test dataset.
    Returns the average loss for the epoch, or NaN/Inf on error.
    """
    error_flag = -1
    model.eval()
    total_loss = 0.0
    for i, batch in enumerate(test_loader):
        batch = batch.to(device)
        batch_data = _convert_batch_format(batch)
        # compute expected value preds
        out = model.forward(batch_data, device)
        loss = model.loss(out, batch, str(device), None, i)
        # validate compute loss of the batch
        if not torch.isfinite(loss):
            error_flag = i
            break
        total_loss += loss.item() * len(batch)
    return (
        total_loss / len(test_loader.dataset) # type: ignore
        if test_loader.dataset
        else 0,
        error_flag
     )


def _convert_batch_format(batch) -> dict[str, torch.Tensor]:
    """
    Helper function for converting a batch returned by a DataLoader
    to the data format expected by forward method of AnticipateGCN.
    """
    return {
        "batch_index": batch.batch,
        "edge_data": batch.edge_index,
        "edge_features": batch.edge_attr,
        "latent_features": batch.x
    }


def _get_train_test_dataloaders() -> tuple[DataLoader, DataLoader]:
    # load dataset
    dataset = CSVPickleDataset(
        os.fspath(TRAIN_DATASET_PATH),
        preprocess_function=prepare_gcn_input
    )

    if TEST_DATASET_PATH is None:
        # split the full dataset
        train_dataset, test_dataset = split_dataset(dataset, TRAIN_TEST_SPLIT)
        print("Number of training graphs:", len(train_dataset))
        print("Number of testing graphs:", len(test_dataset))

    train_loader = DataLoader(
        train_dataset, # type: ignore
        batch_size=int(HYPERPARAMETERS["batch_size"]),
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, # type: ignore
        batch_size=int(HYPERPARAMETERS["batch_size"]),
        shuffle=False
    )
    return train_loader, test_loader


if __name__ == "__main__":
    # Always freeze your random seeds
    torch.manual_seed(8616)
    random.seed(8616)

    main()
