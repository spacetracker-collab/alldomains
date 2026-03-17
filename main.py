# AI META-FIELD DEMO (SAFE VERSION)

import torch
import torch.nn as nn
import torch.optim as optim

def build_graph():
n = 5

```
A = torch.zeros((n, n), dtype=torch.float32)

A[0, 0] = 1
A[0, 1] = 1
A[1, 0] = 1
A[1, 1] = 1
A[1, 2] = 1
A[2, 1] = 1
A[2, 2] = 1
A[2, 3] = 1
A[3, 2] = 1
A[3, 3] = 1
A[3, 4] = 1
A[4, 3] = 1
A[4, 4] = 1

A = A + torch.eye(n)

D = torch.diag(torch.sum(A, dim=1))
D_inv_sqrt = torch.linalg.inv(torch.sqrt(D))
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

def action_mapping(x):
return torch.tanh(x)

def main():
torch.manual_seed(42)

```
A_norm, n = build_graph()

d = 4
X = torch.randn(n, d)

model = GCN(d, 8, 3)
optimizer = optim.Adam(model.parameters(), lr=0.01)

target = torch.randn(n, 3)

for epoch in range(120):
    optimizer.zero_grad()

    output = model(X, A_norm)
    loss = ((output - target) ** 2).mean()

    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print("Epoch", epoch, "Loss:", float(loss))

with torch.no_grad():
    activations = model(X, A_norm)
    actions = action_mapping(activations)

print("\nACTIVATIONS:")
print(activations)

print("\nACTIONS:")
print(actions)

print("\nGraph -> Learning -> Activation -> Action")
```

if **name** == "**main**":
main()
