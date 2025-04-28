import torch
import numpy as np
from torch_geometric.loader import DataLoader

# Train loop function
def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    total_loss = []
    for batch, X in enumerate(dataloader):
        optimizer.zero_grad()
        
        # Compute prediction and loss
        pred = model(X)
        pred = pred[:, :X.x.shape[1]]

        loss = loss_fn(pred, X.x)
        total_loss.append(float(loss))

        # Backpropagation
        loss.backward()
        optimizer.step()

    return np.mean(total_loss)

# Evaluation loop function
def eval_loop(dataloader, model, loss_fn, test=False, signal=False):
    model.eval()
    loss = []
    data = []
    with torch.no_grad():
        for X in dataloader:
            pred = model(X)
            pred = pred[:, :X.x.shape[1]]
            loss.append(float(loss_fn(pred, X.x)))
            data.append(X.x)

    if test: 
        model.test_data = data
        model.background_test_loss = loss
    elif signal: 
        model.signal_data = data
        model.signal_loss = loss

    return loss

# Overall training function
def train_model(train_dataloader, test_dataloader, signal_dataloader, model, loss_fn, optimizer, epochs, batch_size):
    model.train_hist = []
    model.val_hist = []
    model.signal_hist = []

    for epoch in range(epochs):
        print(f"Epoch [{epoch+1}/{epochs}]")

        train_loss = train_loop(train_dataloader, model, loss_fn, optimizer)
        val_loss = eval_loop(test_dataloader, model, loss_fn, test=True, signal=False)
        signal_loss = eval_loop(signal_dataloader, model, loss_fn, test=False, signal=True)

        model.train_hist.append(train_loss)
        model.val_hist.append(np.mean(val_loss))
        model.signal_hist.append(np.mean(signal_loss))

        print(f"train loss: {train_loss:.4f}")
        print(f"test loss: {np.mean(val_loss):.4f}")
        print(f"signal loss: {np.mean(signal_loss):.4f}")

    model.background_train_loss = model.train_hist

    return model.train_hist, model.val_hist, model.signal_hist

def train_model(train_dataloader, test_dataloader, signal_dataloader, model, loss_fn, optimizer, epochs, batch_size):
    model.train_hist = []
    model.val_hist = []
    model.signal_hist = []
    for epoch in (range(epochs)):
        print(f"Epoch [{epoch+1}/{epochs}]")
        train_loss = []
        val_loss = []
        signal_loss = []
        train_loop(train_dataloader, model, loss_fn, optimizer)
        val_loss.extend((eval_loop(test_dataloader, model, loss_fn, test=True, signal=False)))
        train_loss.extend((eval_loop(train_dataloader, model, loss_fn, test=False, signal=False)))
        signal_loss.extend((eval_loop(signal_dataloader, model, loss_fn, test=False, signal=True)))

        model.train_hist.append(np.mean(train_loss))
        model.val_hist.append(np.mean(val_loss))
        model.signal_hist.append(np.mean(signal_loss))
#         writer.add_scalar("Loss/val", model.val_hist[-1], epoch)

        print("train loss ", np.mean(train_loss))
        print("test loss ", np.mean(val_loss))
        print("signal loss", np.mean(signal_loss))
        

    model.background_test_loss = val_loss
    model.background_train_loss = train_loss
    model.signal_loss = signal_loss
    return train_loss, val_loss, eval_loop(signal_dataloader, model, loss_fn, test=False, signal=True)
