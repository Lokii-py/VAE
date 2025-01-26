from datetime import datetime
import torch
from model import VAE
from torch.utils.tensorboard import SummaryWriter

# Configuration parameters
learning_rate = 1e-3
weight_decay = 1e-2
num_epochs = 50
latent_dim = 10  
hidden_dim = 1024
input_dim = 32 * 32 * 3  # CIFAR-10 images are 32x32 with 3 color channels

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model and optimizer setup
model = VAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# TensorBoard writer
writer = SummaryWriter(f'runs/cifar10/vae_{datetime.now().strftime("%Y%m%d-%H%M%S")}')