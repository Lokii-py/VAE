# import the necessary packages
from vae import config, network, utils
from torchvision import datasets, transforms
import torch.optim as optim
import torch
import os
import matplotlib

# change the backend based on the non-gui backend available
matplotlib.use("agg")

# define the transformation to be applied to the data
transform = transforms.Compose(
    [transforms.ToTensor()]
)
# load the cifar10 training data and create a dataloader
trainset = datasets.CIFAR10(
    "data", train=True, download=True, transform=transform
)
train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=config.BATCH_SIZE, shuffle=True
)
# load the FashionMNIST test data and create a dataloader
testset = datasets.CIFAR10(
    "data", train=False, download=True, transform=transform
)
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=config.BATCH_SIZE, shuffle=True
)

# instantiate the encoder and decoder models
encoder = network.Encoder(config.IMAGE_SIZE, config.EMBEDDING_DIM).to(
    config.DEVICE
)
decoder = network.Decoder(
    config.EMBEDDING_DIM, config.SHAPE_BEFORE_FLATTENING
).to(config.DEVICE)
# pass the encoder and decoder to VAE class
vae = network.VAE(encoder, decoder)

# instantiate optimizer and scheduler
optimizer = optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()), lr=config.LR
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.1, patience=config.PATIENCE, verbose=True
)

# initialize the best validation loss as infinity
best_val_loss = float("inf")
# start training by looping over the number of epochs
for epoch in range(config.EPOCHS):
    # set the vae model to train mode
    # and move it to CPU/GPU
    vae.train()
    vae.to(config.DEVICE)
    running_loss = 0.0
    # loop over the batches of the training dataset
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(config.DEVICE)
        optimizer.zero_grad()

        # forward pass through the VAE
        pred = vae(data)

        print(f"Epoch: {epoch}, Batch: {batch_idx}, Data shape: {data.shape}")

        # compute the VAE loss
        loss = utils.vae_loss(pred, data)

        # backward pass and optimizer step
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
        # compute average loss for the epoch
    train_loss = running_loss / len(train_loader)

    # compute validation loss for the epoch
    val_loss = utils.validate(vae, test_loader)

    # Log every epoch
    print(f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # print training and validation loss at every 20 epochs
    if epoch % 10 == 0 or (epoch+1) == config.EPOCHS:
        print(
            f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )
    # save best vae model weights based on validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(
            {"vae": vae.state_dict()},
            config.MODEL_WEIGHTS_PATH,
        )
    # adjust learning rate based on the validation loss
    scheduler.step(val_loss)
