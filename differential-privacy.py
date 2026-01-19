# pip install torch torchvision torchaudio
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

# pip install opacus
from opacus import PrivacyEngine

# --- Training with Differential Privacy (Opacus) ---

# 1. Simulate a simple model and data
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = SimpleModel().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.05)
criterion = nn.BCEWithLogitsLoss()

# Simulate a DataLoader
torch.manual_seed(42)
X_data = torch.randn(1000, 10)
y_data = torch.randint(0, 2, (1000, 1)).float()  # shape (N, 1) for BCEWithLogitsLoss

dataset = TensorDataset(X_data, y_data)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

# 2. Ethical Step: Initialize the Privacy Engine (DP-SGD)
# Key knobs:
# - noise_multiplier: higher => more privacy (but potentially lower accuracy)
# - max_grad_norm: gradient clipping bound per-sample
privacy_engine = PrivacyEngine()

model, optimizer, data_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=data_loader,
    noise_multiplier=1.1,
    max_grad_norm=1.0,
)

# 3. Training Loop (DP)
def train_with_dp(model, optimizer, data_loader, epochs=1):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0

        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(data)                 # shape (batch, 1)
            loss = criterion(logits, target)     # target shape (batch, 1)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(data_loader)
        print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f}")

    # 4. Ethical Step: Report the final privacy budget (epsilon)
    delta = 1e-5
    epsilon = privacy_engine.get_epsilon(delta=delta)
    print(f"\n[PRIVACY]: Trained with ε = {epsilon:.2f}, δ = {delta}")

train_with_dp(model, optimizer, data_loader, epochs=3)

print(
    "\n[ETHICAL NOTE]: Differential Privacy (DP-SGD) clips per-sample gradients and adds noise, "
    "limiting how much any single person's data can influence the trained model. "
    "This reduces privacy risks such as membership inference and data reconstruction."
)