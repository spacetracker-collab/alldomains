# ==========================================

# AI META-FIELD DEMO (Single File)

# ==========================================

import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------------

# 1. Graph Definition

# -----------------------------

# Number of nodes

n = 5

# Adjacency matrix (graph structure)

A = torch.tensor([
[1,1,0,0,0],
[1,1,1,0,0],
[0,1,1,1,0],
[0,0,1,1,1],
[0,0,0,1,1]
], dtype=torch.float32)

# Feature matrix (node features)

d = 3
X = torch.randn(n, d)

# -----------------------------

# 2. Graph Neural Network

# -----------------------------

class GCN(nn.Module):
def **init**(self, in_dim, hidden_dim, out_dim):
super(GCN, self).**init**()
self.W1 = nn.Linear(in_dim, hidden_dim)
self.W2 = nn.Linear(hidden_dim, out_dim)

```
def forward(self, X, A):
    # Layer 1
    H = torch.matmul(A, X)
    H = self.W1(H)
    H = torch.relu(H)

    # Layer 2
    H = torch.matmul(A, H)
    H = self.W2(H)

    return H
```

# -----------------------------

# 3. Model Setup

# -----------------------------

model = GCN(in_dim=d, hidden_dim=8, out_dim=2)

optimizer = optim.Adam(model.parameters(), lr=0.01)

# Target (dummy "domain" output)

target = torch.randn(n, 2)

# -----------------------------

# 4. Training (Learning Dynamics)

# -----------------------------

epochs = 100

for epoch in range(epochs):
optimizer.zero_grad()

```
output = model(X, A)

loss = ((output - target)**2).mean()

loss.backward()
optimizer.step()

if epoch % 20 == 0:
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

# -----------------------------

# 5. Neural Activation → Action

# -----------------------------

def action_mapping(activation):
# Simple mapping to action space
return torch.tanh(activation)

with torch.no_grad():
activations = model(X, A)
actions = action_mapping(activations)

print("\nFinal Activations:")
print(activations)

print("\nActions (Observable Output):")
print(actions)

# -----------------------------

# 6. Interpretation

# -----------------------------

print("\nInterpretation:")
print("Graph → Learning → Activation → Action")
print("This demonstrates:")
print("- Domain as graph")
print("- Learning over structure")
print("- Action as observable neural activation")
