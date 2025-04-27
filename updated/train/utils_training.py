import torch
import numpy as np
from torch_geometric.loader import DataLoader

def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    total_loss = []
    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        pred = model(batch)
        pred = pred[:, :batch.x.shape[1]]
        loss = loss_fn(pred, batch.x)
        total_loss.append(float(loss))
        loss.backward()
        optimizer.step()
    return np.mean(total_loss)

def eval_loop(dataloader, model, loss_fn, record_data=False):
    model.eval()
    loss = []
    all_data = []
    with torch.no_grad():
        for batch in dataloader:
            pred = model(batch)
            pred = pred[:, :batch.x.shape[1]]
            loss.append(float(loss_fn(pred, batch.x)))
            if record_data:
                all_data.append(batch.x)
    return np.mean(loss), all_data

def train_model(train_dataloader, test_dataloader, signal_dataloader, model, loss_fn, optimizer, epochs):
    model.train_hist = []
    model.val_hist = []
    model.signal_hist = []

    for epoch in range(epochs):
        print(f"Epoch [{epoch+1}/{epochs}]")
        train_loss = train_loop(train_dataloader, model, loss_fn, optimizer)
        val_loss, _ = eval_loop(test_dataloader, model, loss_fn)
        signal_loss, _ = eval_loop(signal_dataloader, model, loss_fn)

        model.train_hist.append(train_loss)
        model.val_hist.append(val_loss)
        model.signal_hist.append(signal_loss)

        print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Signal Loss: {signal_loss:.4f}")

    model.background_train_loss = model.train_hist
    model.background_test_loss = model.val_hist
    model.signal_loss = model.signal_hist

    return model.train_hist, model.val_hist, model.signal_hist