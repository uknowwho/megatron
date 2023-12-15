import torch

from data import Data
from model import Megatron

torch.manual_seed(1337)
torch.autograd.set_detect_anomaly(True)


def get_batch(data, context_length, batch_size):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    samples = torch.randint(low=0, high=len(data) - context_length, size=(batch_size,))
    xb = torch.stack([data[sample : sample + context_length] for sample in samples])
    yb = torch.stack(
        [data[sample + 1 : sample + context_length + 1] for sample in samples]
    )
    xb, yb = xb.to(device), yb.to(device)
    return xb, yb


@torch.no_grad()  # No need to update gradients during evaluation
def estimate_loss(model, train_data, val_data, batch_size, eval_iters=50):
    model.eval()
    train_losses = torch.zeros(eval_iters)
    val_losses = torch.zeros(eval_iters)
    for iter in range(eval_iters):
        X_train, y_train = get_batch(train_data, model.context_length, batch_size)
        _, loss_train = model(X_train, y_train)
        train_losses[iter] = loss_train.item()

        X_val, y_val = get_batch(val_data, model.context_length, batch_size)
        _, loss_val = model(X_val, y_val)
        val_losses[iter] = loss_val.item()

    model.train()
    return torch.mean(train_losses), torch.mean(val_losses)


def train(
    model,
    train_data,
    val_data,
    optimizer,
    batch_size=16,
    train_iters=5000,
    eval_iters=50,
):
    for iter in range(train_iters):
        model.train()
        batch_data, batch_targets = get_batch(
            train_data, model.context_length, batch_size
        )
        _, batch_loss = model(batch_data, batch_targets)
        optimizer.zero_grad(set_to_none=True)
        batch_loss.backward()
        optimizer.step()

        if iter % 100 == 0:
            train_loss, val_loss = estimate_loss(
                model, train_data, val_data, batch_size, eval_iters
            )
            print(f"The training loss at iteration {iter} is: {train_loss}")
            print(f"The validation loss at iteration {iter} is: {val_loss}")

    torch.save(model, "megatron.pt")


if __name__ == "__main__":
    print("Loading data...")
    d = Data("all_articles.txt")
    data_tensor, vocab_size = d.load_data()
    train_data, val_data = d.train_test_split(data_tensor)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Initializing model...")
    model = Megatron(
        vocab_size,
        embed_length=32,
        context_length=64,
        dropout=0.15,
        num_heads=2,
        num_blocks=3,
    )
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    print("Training model")
    train(model, train_data, val_data, optimizer)
    print("Saved model as megatron.pt")
