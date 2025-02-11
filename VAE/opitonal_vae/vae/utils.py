""" This file will contains utilites liek the loss function of VAE, post-training analysis, 
and a validation method for evaluating the VAE during training"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from vae import config

def vae_gaussian_kl_loss(mu, logvar):
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return KLD.mean()

def reconstruction_loss(x_reconstructed, x):
    bce_loss = nn.BCELoss()
    return bce_loss(x_reconstructed, x)
    
def vae_loss(y_pred, y_true):
    mu, logvar, recon_x = y_pred
    recon_loss = reconstruction_loss(recon_x, y_true)
    kld_loss = vae_gaussian_kl_loss(mu, logvar)
    return 500 * recon_loss + kld_loss

def validate(model, test_loader):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(config.DEVICE)
            pred = model(data)
            loss = vae_loss(pred, data)
            running_loss += loss.item()
    return running_loss / len(test_loader)


