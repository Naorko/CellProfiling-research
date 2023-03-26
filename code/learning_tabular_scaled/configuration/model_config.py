from enum import Enum

from model_layer.TabularAE import TabularAE

default_exp = (418, 70, 1.5e-4, 25, 8)
exp_values = default_exp
input_size, target_size, lr, epochs, latent_space_dim = exp_values


class Model_Config(Enum):
    TAE = ('autoencoder', TabularAE,
           {'input_size': input_size,
            'target_size': target_size,
            'lr': lr,
            'epochs': epochs,
            'latent_space_dim': latent_space_dim})

    def __init__(self, model_name, model_class, params):
        self.model_name = model_name
        self.model_class = model_class
        self.params = params

    def update_custom_params(self, params):
        for k in params.keys():
            if k in self.params:
                self.params[k] = params[k]
