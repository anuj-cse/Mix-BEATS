import torch
from torch import nn
from torch.nn import functional as F

class TSMixer(nn.Module):
    def __init__(self, patch_size, num_patches, patch_hidden_dim, feature_hidden_dim):
        super(TSMixer, self).__init__()

        self.patch_size = patch_size
        self.num_patches = num_patches
        
        self.patch_mix = nn.Sequential(
            nn.LayerNorm(num_patches),
            nn.Linear(num_patches, patch_hidden_dim),
            nn.GELU(),
            nn.Linear(patch_hidden_dim, num_patches)
        )
        
        self.feature_mix = nn.Sequential(
            nn.LayerNorm(patch_size),
            nn.Linear(patch_size, feature_hidden_dim),
            nn.GELU(),
            nn.Linear(feature_hidden_dim, patch_size)
        )
        
    def forward(self, x):
        batch_size, context_length = x.shape
        x = x.view(batch_size, self.num_patches, self.patch_size)

        x = x + self.patch_mix(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.feature_mix(x)

        x = x.reshape(batch_size, self.num_patches * self.patch_size)
        return x


class GenericBlock(nn.Module):
    def __init__(self, hidden_dim, thetas_dim, device, backcast_length=10, forecast_length=5, patch_size=8, num_patches=21):
        super(GenericBlock, self).__init__()
        self.hidden_dim = hidden_dim
        self.thetas_dim = thetas_dim
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.device = device
        
        self.TTM = TSMixer(self.patch_size, self.num_patches, self.hidden_dim, self.hidden_dim)
        
        self.theta_b_fc = nn.Linear(backcast_length, thetas_dim, bias=False)
        self.theta_f_fc = nn.Linear(backcast_length, thetas_dim, bias=False)

        self.backcast_fc = nn.Linear(thetas_dim, backcast_length)
        self.forecast_fc = nn.Linear(thetas_dim, forecast_length)

    def forward(self, x):

        x = self.TTM(x)


        theta_b = self.theta_b_fc(x)
        theta_f = self.theta_f_fc(x)

        backcast = self.backcast_fc(theta_b)
        forecast = self.forecast_fc(theta_f)

        return backcast, forecast
    

class NBeatsNet(nn.Module):
    def __init__(
            self,
            device=torch.device('cpu'),
            nb_blocks_per_stack=3,
            forecast_length=24,
            backcast_length=128,
            patch_size=8,
            num_patches=21,
            thetas_dim=8,
            hidden_dim=256,
            share_weights_in_stack=False
    ):
        super(NBeatsNet, self).__init__()
        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.hidden_dim = hidden_dim
        self.nb_blocks_per_stack = nb_blocks_per_stack
        self.share_weights_in_stack = share_weights_in_stack
        self.thetas_dim = thetas_dim
        self.device = device
        self.stack_type = ['generic', 'generic', 'generic']

        # Using nn.ModuleList to hold blocks so that parameters are registered properly
        self.parameters = []

        self.stacks = [self.create_stack(type) for type in self.stack_type]
        self.parameters = nn.ParameterList(self.parameters)

        self.to(self.device)
        self._loss = None
        self._opt = None

    def create_stack(self, type):
        if type == 'generic':
            blocks = []
            for _ in range(self.nb_blocks_per_stack):
                block = GenericBlock(
                    self.hidden_dim, self.thetas_dim,
                    self.device, self.backcast_length, self.forecast_length, 
                    self.patch_size, self.num_patches,
                )
                self.parameters.extend(block.parameters())
                blocks.append(block)
            return blocks

    def compile(self, loss='mae', optimizer='adam'):
        if loss == 'mae':
            loss_ = F.l1_loss
        elif loss == 'mse':
            loss_ = F.mse_loss
        else:
            raise ValueError(f'Unknown loss name: {loss}.')
        
        if optimizer == 'adam':
            # Now self.parameters() will correctly return all parameters
            opt_ = torch.optim.Adam(self.parameters(), lr=1e-4)
        else:
            raise ValueError(f'Unknown optimizer: {optimizer}.')
        
        self._opt = opt_
        self._loss = loss_



    def fit(self, x_train, y_train, epochs=10, batch_size=32):
        for epoch in range(epochs):
            self.train()
            train_loss = []
            for batch_idx in range(0, len(x_train), batch_size):
                batch_x = torch.tensor(x_train[batch_idx:batch_idx+batch_size], dtype=torch.float).to(self.device)
                batch_y = torch.tensor(y_train[batch_idx:batch_idx+batch_size], dtype=torch.float).to(self.device)
                
                self._opt.zero_grad()
                _, forecast = self(batch_x)
                loss = self._loss(forecast, batch_y)
                train_loss.append(loss.item())
                loss.backward()
                self._opt.step()

            print(f'Epoch {epoch + 1}/{epochs}, Loss: {torch.mean(torch.tensor(train_loss)):.4f}')

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            forecast = self(torch.tensor(x, dtype=torch.float).to(self.device))[1]
        return forecast.cpu().numpy()

    def forward(self, backcast):
        forecast = torch.zeros(backcast.size(0), self.forecast_length).to(self.device)
        for stack in self.stacks:
            for block in stack:
                backcast_block, forecast_block = block(backcast)
                backcast = backcast - backcast_block  
                forecast += forecast_block  
        return backcast, forecast


