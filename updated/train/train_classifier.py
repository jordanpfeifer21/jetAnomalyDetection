import torch
from torch_geometric.loader import DataLoader
from models.classifier import JetGraphAutoencoderClassification
from sklearn.metrics import accuracy_score


def run_classifier_training(graphs, smallest_dim=16, num_reduced_edges=16, batch_size=10, epochs=20, initial_lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    n_feats = graphs[0].x.shape[1]
    topk_layer = None  # Optional: you can plug TopKPooling here if needed

    model = JetGraphAutoencoderClassification(
        num_features=n_feats,
        smallest_dim=smallest_dim,
        topk=topk_layer,
        num_reduced_edges=num_reduced_edges
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr)
    criterion = torch.nn.MSELoss()

    train_loader = DataLoader(graphs, batch_size=batch_size, shuffle=True)

    model.train()
    losses, accuracies = [], []

    for epoch in range(epochs):
        total_loss = 0
        all_preds, all_labels = [], []

        for data in train_loader:
            optimizer.zero_grad()
            output = model(data, knn=False, topk=False)
            truth = torch.reshape(data.y.float(), (data.num_graphs, 1))
            loss = criterion(output, truth)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = (output >= 0.5).int()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(truth.cpu().numpy())

        avg_loss = total_loss / len(graphs)
        acc = accuracy_score(all_labels, all_preds)
        losses.append(avg_loss)
        accuracies.append(acc)

        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {acc:.4f}")

    return model, losses, accuracies