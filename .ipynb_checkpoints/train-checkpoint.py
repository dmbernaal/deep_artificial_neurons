# EXAMPLE
import torch
import torch.optim as optim
import torch.nn as nn
from dan import DAN

def train(epochs, model, opt, loss_fn, device='cuda:0'):
    device = torch.device(device)
    model = model.to(device)
    dummy_data = [(torch.randn(64, 100), torch.randn(64, 1)) for i in range(100)]
    for i in range(epochs):
        for dd in dummy_data:
            x, y = dd
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            
        print(f'Epoch: {i+1}/{epochs}, loss: {loss.item()}')
        
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
if __name__ == '__main__':
    seed_everything(42)

    model = DAN(100, 1, act=nn.ReLU())
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    train(100, model, optimizer, loss_fn)