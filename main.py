# ==========================================

# AI META-FIELD DEMO (FINAL CLEAN VERSION)

# ==========================================

import torch
import torch.nn as nn
import torch.optim as optim

def build_graph():
"""
Creates adjacency matrix and normalized adjacency
"""
n = 5

```
A = torch.tensor([
    [1, 1, 0, 0, 0],
    [1, 1, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1]
], dtype=torch.float32)

# Add self loops
A = A + torch.eye(n)

# Degree matrix
D = torch.diag(torch.sum(A, dim=1))

# D^{-1/2}
D_inv_sqrt = torch.linalg.inv(torch.sqrt(D))

# Normalized adjacency
A_norm = D_inv_sqrt @ A @ D_inv_sqrt

return A_norm, n
```

class GCN(nn.Module):
def **init**(self, in_dim, hidden_dim, out_dim):
super(GCN, self).**init**()
self.fc1 = nn.Linear(in_dim, hidden_dim)
self.fc2 = nn.Linear(hidden_dim, out_dim)

```
def forward(self, X, A):
    H = A @ X
    H = self.fc1(H)
    H = torch.relu(H)

    H = A @ H
    H = self.fc2(H)

    return H
```

def action_mapping(activations):
"""
Maps neural activations to observable actions
"""
return torch.tanh(activations)

def main():
torch.manual_seed(42)

```
# Build graph
A_norm, n = build_graph()

# Feature dimension
d = 4

# Node features
X = torch.randn(n, d)

# Model
model = GCN(in_dim=d, hidden_dim=8, out_dim=3)

optimizer = optim.Adam(model.parameters(), lr=0.01)

# Dummy target (domain representation)
target = torch.randn(n, 3)

# Training
epochs = 120

for epoch in range(epochs):
    optimizer.zero_grad()

    output = model(X, A_norm)

    loss = torch.mean((output - target) ** 2)

    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")

# Final output
with torch.no_grad():
    activations = model(X, A_norm)
    actions = action_mapping(activations)

print("\n=== FINAL RESULTS ===")

print("\nNode Activations (Internal State):")
print(activations)

print("\nActions (Observable Output):")
print(actions)

print("\n=== INTERPRETATION ===")
print("Graph → Learning → Activation → Action")
print("This validates the AI meta-field concept.")
```

if **name** == "**main**":
main()


