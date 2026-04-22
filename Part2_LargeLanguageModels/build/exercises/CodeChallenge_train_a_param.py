import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import matplotlib.pyplot as plt
from torch.optim import optimizer

class PiModel(nn.Module):
    
    def __init__(self):
        super(PiModel, self).__init__()

        self.layer = nn.Linear(1, 1)

    def forward(self, x):
        return self.layer(x)

def test_optimizer(model, optim, label, losses, weights, epochs):
    
    target = torch.tensor([math.pi], dtype=torch.float32)

    for i in range(epochs):
        pred = model(torch.tensor([1], dtype=torch.float32))    
        optim.zero_grad() # Zero out the gradients
        # loss = F.mse_loss(target, pred)
        loss = (target - pred)**2
        loss.backward() # Calculate the gradients
        optim.step() # Actually take the step

        losses[label][i] = loss.item()
        weights[label][i] = model.layer.weight.item()



epochs = 50
lr = 0.05

losses = {
    "SGD": torch.zeros(epochs),
    "Adam": torch.zeros(epochs),
    "AdamW": torch.zeros(epochs)
}

weights = {
    "SGD": torch.zeros(epochs),
    "Adam": torch.zeros(epochs),
    "AdamW": torch.zeros(epochs)
}

# Test SGD
pimodel = PiModel()
optim = torch.optim.SGD(pimodel.parameters(), lr=lr)
test_optimizer(pimodel, optim, "SGD", losses, weights, epochs)

# Test Adam
pimodel = PiModel()
optim = torch.optim.Adam(pimodel.parameters(), lr=lr)
test_optimizer(pimodel, optim, "Adam", losses, weights, epochs)

# Test AdamW
pimodel = PiModel()
optim = torch.optim.AdamW(pimodel.parameters(), lr=lr)
test_optimizer(pimodel, optim, "AdamW", losses, weights, epochs)


    # if(i % 10 == 0):
    #     print(f"Predicted value of Pi is {pred.item():1f} v.s the real value of Pi - {target}")

fig, ax = plt.subplots(1, 2, figsize=(15,4))

for label in losses.keys():
    ax[0].plot(torch.arange(0, len(losses[label])), losses[label], label=label)
    ax[1].plot(torch.arange(0, len(weights[label])), weights[label], label=label)

ax[0].set_title("Loss Values")
ax[1].set_title("Weight Values")
ax[0].legend()
ax[1].legend()

plt.show()

print(losses)
    



   


    




