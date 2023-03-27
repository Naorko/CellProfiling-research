from argparse import Namespace

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics.functional import pearson_corrcoef
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch import nn


# Holds the model architecture configuration
class TabularAE(pl.LightningModule):

    # fixed bug for pl loading model_checkpoint according to
    # https://github.com/PyTorchLightning/pytorch-lightning/issues/2909
    def __init__(self, *args, **kwargs):
        super(TabularAE, self).__init__()

        if isinstance(kwargs, dict):
            hparams = Namespace(**kwargs)

        self.save_hyperparameters(kwargs)
        self.input_size = hparams.input_size
        self.target_size = hparams.target_size
        self.latent_space_dim = hparams.latent_space_dim
        self.bilinear = True

        # Create the Encoder architecture according to the input and latent size
        print(f'Model size configurations:')
        print(f'\tInput size: {self.input_size}')
        enc_dims = [2 ** i for i in range(1, 16)
                    if self.latent_space_dim <= 2 ** i <= self.input_size+1]
        print(f'\tEncode size: {enc_dims}')
        enc_layers = [[nn.Linear(self.input_size, enc_dims[-1]), nn.ReLU(inplace=True)]]
        enc_layers += [[nn.Linear(enc_dims[len(enc_dims) - i], enc_dims[len(enc_dims) - i - 1]),
                        nn.ReLU(inplace=True)]
                       for i in range(1, len(enc_dims))]
        enc_layers = sum(enc_layers, [])
        self.encoder = nn.Sequential(*enc_layers)

        # Create the Decoder architecture according to the latent and target size
        dec_dims = [2 ** i for i in range(16, 1, -1)
                    if self.latent_space_dim <= 2 ** i <= self.target_size+1]
        print(f'\tDecode size: {dec_dims}')
        dec_layers = [[nn.Linear(dec_dims[len(dec_dims) - i], dec_dims[len(dec_dims) - i - 1]),
                       nn.ReLU(inplace=True)]
                      for i in range(1, len(dec_dims))]
        dec_layers += [[nn.Linear(dec_dims[0], self.target_size)]]
        dec_layers = sum(dec_layers, [])
        self.decoder = nn.Sequential(*dec_layers)
        print(f'\tTarget size: {self.target_size}')

    # Set the forward process as chaining the encoder and the decoder
    def forward(self, x):
        z = self.encoder(x)
        y_hat = self.decoder(z)
        return y_hat

    # Set the training step - define the loss as MSE
    def training_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        tensorboard_logs = {'train_loss': loss.detach()}
        self.log('train_loss', loss.detach(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'log': tensorboard_logs}

    # Set the validation step - define the loss as MSE and logging the PCC (using unify_test_function)
    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        _, loss, pcc = unify_test_function(self, batch)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_pcc', pcc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'val_loss': loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss.detach()}
        self.log('avg_val_loss', avg_loss.detach(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    # Configure the Adam optimizer
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=1e-8)

    # Configure the callbacks, early stopping and model checkpoint
    def configure_callbacks(self):
        early_stop = EarlyStopping(monitor="val_loss", mode="min", min_delta=0.002, patience=4, verbose=True)
        checkpoint = ModelCheckpoint(monitor="val_loss")
        return [early_stop, checkpoint]


# Test a batch during testing or validating
#   mse_reduction:
#       - set to 'mean' if validating will output one number for the batch
#       - set to 'none' if testing will output mse for each entry
def unify_test_function(model, batch, mse_reduction='mean'):
    if len(batch) == 3:
        x, y, _ = batch
    else:
        x, y = batch

    x, y = x.to(model.device), y.to(model.device)
    y_hat = model.forward(x)
    loss = F.mse_loss(y_hat.detach(), y, reduction=mse_reduction)
    pcc = pearson_corrcoef(y_hat.reshape(-1), y.reshape(-1))
    return y_hat.detach(), loss.detach(), pcc.detach()
