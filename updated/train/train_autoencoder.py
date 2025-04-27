import torch
from torch_geometric.loader import DataLoader
from models.autoencoder import JetGraphAutoencoder
from train.utils_training import train_model

def run_autoencoder_training(train_graphs, test_graphs, signal_graphs, smallest_dim=16, num_reduced_edges=16, batch_size=64, epochs=20, initial_lr=5e-6):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    n_feats = train_graphs[0].x.shape[1]
    model = JetGraphAutoencoder(
        num_features=n_feats,
        smallest_dim=smallest_dim,
        num_reduced_edges=num_reduced_edges
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr)
    loss_fn = torch.nn.MSELoss()

    train_dataloader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)
    signal_dataloader = DataLoader(signal_graphs, batch_size=batch_size, shuffle=False)

    train_loss, val_loss, signal_loss = train_model(
        train_dataloader,
        test_dataloader,
        signal_dataloader,
        model,
        loss_fn,
        optimizer,
        epochs
    )

    return model, train_loss, val_loss, signal_loss