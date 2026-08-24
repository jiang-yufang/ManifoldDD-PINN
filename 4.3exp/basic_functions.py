import torch
import torch.nn as nn

class PINN(nn.Module):
    """Physics-informed neural network for second-order elliptic equations on spheres."""

    def __init__(self, input_dim=2, hidden_dim=50, layers=4, dtype=torch.float32, act_function = nn.Tanh):
        """
        Initialize the PINN.

        Args:
            input_dim: Dimension of the input coordinates.
            hidden_dim: Width of the hidden layers.
            layers: Number of hidden layers.
            dtype: Data type of the parameters.
            act_function: Activation function class.
        """
        super(PINN, self).__init__()
        self.dtype = dtype

        # nn.ModuleList keeps every layer registered so that its parameters
        # can be accessed and optimized directly.
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim).to(dtype=dtype))
        self.layers.append(act_function())

        for _ in range(layers-1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim).to(dtype=dtype))
            self.layers.append(act_function())

        self.layers.append(nn.Linear(hidden_dim, 1).to(dtype=dtype))

    def forward(self, x):
        """Forward pass; casts the input to the network's dtype."""
        x = x.to(self.dtype)
        for layer in self.layers:
            x = layer(x)
        return x

class SimpleFCResidualBlock(nn.Module):
    """Fully-connected residual block: FC -> activation -> FC + shortcut -> activation."""

    def __init__(self, in_features, out_features, act_function):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.act = act_function()

        # A linear projection is used when the input and output dimensions differ.
        self.shortcut = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.act(self.fc1(x))
        out = self.fc2(out)
        out += identity
        out = self.act(out)
        return out

class Res_PINN(nn.Module):
    """Residual physics-informed neural network for second-order elliptic equations on spheres."""

    def __init__(self, input_dim=2, hidden_dim=50, layers=4, dtype=torch.float32, act_function = nn.Tanh):
        """
        Initialize the residual PINN.

        Args:
            input_dim: Dimension of the input coordinates.
            hidden_dim: Width of the hidden layers.
            layers: Number of residual blocks (hidden layers).
            dtype: Data type of the parameters.
            act_function: Activation function class.
        """
        super(Res_PINN, self).__init__()
        self.dtype = dtype

        self.layers = nn.ModuleList()
        self.layers.append(SimpleFCResidualBlock(input_dim, hidden_dim, act_function).to(dtype=dtype))

        for _ in range(layers-1):
            self.layers.append(SimpleFCResidualBlock(hidden_dim, hidden_dim, act_function).to(dtype=dtype))

        self.layers.append(nn.Linear(hidden_dim, 1).to(dtype=dtype))

    def forward(self, x):
        """Forward pass; casts the input to the network's dtype."""
        x = x.to(self.dtype)
        for layer in self.layers:
            x = layer(x)
        return x

class SimpleSimpleFCResidualBlock(nn.Module):
    """Fully-connected residual block with a single linear layer (pre-activation style)."""

    def __init__(self, in_features, out_features, act_function):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc1 = nn.Linear(in_features, out_features)
        self.act = act_function()

        # A linear projection is used when the input and output dimensions differ.
        self.shortcut = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.act(self.fc1(x))
        out = out + identity
        return out

class Simple_PreAct_Res_PINN(nn.Module):
    """Pre-activated residual physics-informed neural network for elliptic equations on spheres."""

    def __init__(self, input_dim=2, hidden_dim=50, layers=4, dtype=torch.float32, act_function = nn.Tanh):
        """
        Initialize the pre-activated residual PINN.

        Args:
            input_dim: Dimension of the input coordinates.
            hidden_dim: Width of the hidden layers.
            layers: Number of residual blocks (hidden layers).
            dtype: Data type of the parameters.
            act_function: Activation function class.
        """
        super(Simple_PreAct_Res_PINN, self).__init__()
        self.dtype = dtype

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim).to(dtype=dtype))
        self.layers.append(act_function())

        for _ in range(layers-1):
            self.layers.append(SimpleSimpleFCResidualBlock(hidden_dim, hidden_dim, act_function).to(dtype=dtype))

        self.layers.append(nn.Linear(hidden_dim, 1).to(dtype=dtype))

    def forward(self, x):
        """Forward pass; casts the input to the network's dtype."""
        x = x.to(self.dtype)
        for layer in self.layers:
            x = layer(x)
        return x



def UniformBall(N, n, device=None, dtype=torch.float32):
    '''
    Sample N uniformly distributed points in the n-dimensional unit ball.
    '''
    sample = torch.randn(N, n, device=device, dtype=dtype)
    norm = torch.rand(N, 1, device=device, dtype=dtype)
    # Points are uniform on every concentric sphere, and the probability of
    # falling between two spheres equals the ratio of their areas/volumes.
    sample = sample / torch.norm(sample, dim=1, keepdim=True) * norm**(1/n)
    return sample

def UniformSphericalSurface(N, n, device=None, dtype=torch.float32):
    """
    Sample N uniformly distributed points on the unit n-sphere in R^{n+1}.
    """
    # Gaussian vectors are directionally uniform; normalize them onto the sphere.
    x = torch.randn(N, n+1, device=device, dtype=dtype)
    norm_x = torch.norm(x, dim=1, keepdim=True)
    x = x / norm_x
    return x
