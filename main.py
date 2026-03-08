"""
Transfer Learning REAL com PyTorch
Modelo: ResNet18 pré-treinada no ImageNet
Dataset: CIFAR-10 (2 classes: gato vs cachorro)

Instalar: pip install torch torchvision matplotlib
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np

# ── Config ────────────────────────────────────────────────────────
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS    = 5
BATCH     = 32
N_TREINO  = 500   # poucos dados — cenário típico de transfer learning
CLASSES   = [3, 5]  # CIFAR-10: 3=cat, 5=dog
print(f"Device: {DEVICE}\n")

# ── Dataset ───────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],   # média ImageNet
                         [0.229, 0.224, 0.225]),
])

print("Baixando CIFAR-10...")
train_full = datasets.CIFAR10(root="./data", train=True,  download=True, transform=transform)
test_full  = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

def filtrar(dataset, classes, n=None):
    """Filtra só as classes desejadas e limita amostras."""
    idx = [i for i, (_, y) in enumerate(dataset) if y in classes]
    if n:
        idx = idx[:n]
    sub = Subset(dataset, idx)
    # remapeia labels para 0 e 1
    sub.dataset.targets = [classes.index(dataset.targets[i]) if dataset.targets[i] in classes
                           else dataset.targets[i] for i in range(len(dataset))]
    return sub

train_ds = filtrar(train_full, CLASSES, N_TREINO)
test_ds  = filtrar(test_full,  CLASSES)

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH)

print(f"Treino: {len(train_ds)} amostras | Teste: {len(test_ds)} amostras\n")

# ── Função de treino ──────────────────────────────────────────────
def treinar(model, loader_tr, loader_te, epochs, label):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    hist_acc, hist_loss = [], []

    print(f"{'─'*50}")
    print(f"  {label}")
    print(f"{'─'*50}")
    print(f"  {'Época':>5}  {'Loss':>8}  {'Acurácia':>10}")

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for X, y in loader_tr:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out  = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Avaliação
        model.eval()
        corretos = 0
        with torch.no_grad():
            for X, y in loader_te:
                X, y = X.to(DEVICE), y.to(DEVICE)
                corretos += (model(X).argmax(1) == y).sum().item()

        acc  = corretos / len(loader_te.dataset)
        loss_ep = total_loss / len(loader_tr)
        hist_acc.append(acc)
        hist_loss.append(loss_ep)
        print(f"    {ep:>3}/{epochs}    {loss_ep:.4f}    {acc:.4f}")

    print()
    return hist_acc, hist_loss

# ── Modelo 1: DO ZERO ─────────────────────────────────────────────
model_zero = models.resnet18(weights=None)          # sem pesos pré-treinados
model_zero.fc = nn.Linear(512, 2)                   # 2 classes: gato/cachorro
model_zero = model_zero.to(DEVICE)

acc_zero, loss_zero = treinar(model_zero, train_loader, test_loader, EPOCHS, "ResNet18 — DO ZERO")

# ── Modelo 2: TRANSFER LEARNING ───────────────────────────────────
model_tl = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # pesos ImageNet

# Congela TODAS as camadas
for param in model_tl.parameters():
    param.requires_grad = False

# Substitui só a última camada (única que vai treinar)
model_tl.fc = nn.Linear(512, 2)   # requires_grad=True por padrão
model_tl = model_tl.to(DEVICE)

params_treinaveis = sum(p.numel() for p in model_tl.parameters() if p.requires_grad)
params_total      = sum(p.numel() for p in model_tl.parameters())
print(f"Parâmetros treináveis: {params_treinaveis:,} de {params_total:,} total\n")

acc_tl, loss_tl = treinar(model_tl, train_loader, test_loader, EPOCHS, "ResNet18 — TRANSFER LEARNING")

# ── Resultados CLI ────────────────────────────────────────────────
print("=" * 50)
print("  RESULTADO FINAL")
print("=" * 50)
print(f"  Do zero      →  acurácia: {acc_zero[s-1]:.4f}")
print(f"  Transfer     →  acurácia: {acc_tl[-1]:.4f}")
print(f"  Ganho        →  +{(acc_tl[-1] - acc_zero[-1])*100:.1f}%\n")

# ── Gráfico ───────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Transfer Learning Real — ResNet18 (PyTorch)", fontsize=14, fontweight="bold")

ep = range(1, EPOCHS + 1)

ax1.plot(ep, acc_zero, "o-", color="#e05c5c", lw=2, label="Do zero")
ax1.plot(ep, acc_tl,   "s-", color="#4a9eff", lw=2, label="Transfer Learning")
ax1.set_title("Acurácia (teste)")
ax1.set_xlabel("Época"); ax1.set_ylabel("Acurácia")
ax1.legend(); ax1.grid(True, alpha=0.3); ax1.set_ylim(0, 1.05)

ax2.plot(ep, loss_zero, "o-", color="#e05c5c", lw=2, label="Do zero")
ax2.plot(ep, loss_tl,   "s-", color="#4a9eff", lw=2, label="Transfer Learning")
ax2.set_title("Loss (treino)")
ax2.set_xlabel("Época"); ax2.set_ylabel("Loss")
ax2.legend(); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("resultado.png", dpi=150)
plt.show()
print("Gráfico salvo em resultado.png")
