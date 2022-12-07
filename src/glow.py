import torch
from nflows import distributions, transforms, flows
from src.transforms import ConvNet, ReshapeTransform
from .utils import create_mid_split_binary_mask, nats_to_bits_per_dim
from torch.utils.data import DataLoader
from torch import optim


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_transform_step(num_channels, level_hidden_channels, actnorm=True):
    step_transforms = []
    mask = create_mid_split_binary_mask(num_channels)
    def create_convnet(in_channels, out_channels):
        return ConvNet(in_channels, level_hidden_channels, out_channels)
    if actnorm:
        step_transforms.append(transforms.ActNorm(num_channels))

    step_transforms.extend([
        transforms.OneByOneConvolution(num_channels),
        transforms.AffineCouplingTransform(mask=mask, transform_net_create_fn=create_convnet)
    ])

    return transforms.CompositeTransform(step_transforms)

def create_transform(c, h, w, levels=3, hidden_channels=256, steps_per_level=10, num_bits=8, multi_scale=True):
    hidden_channels = [hidden_channels] * levels
    if multi_scale:
        mct = transforms.MultiscaleCompositeTransform(num_transforms=levels)
        for level, level_hidden_channels in zip(range(levels), hidden_channels):
            squeeze_transform = transforms.SqueezeTransform()
            c, h, w = squeeze_transform.get_output_shape(c, h, w)

            transform_level = transforms.CompositeTransform(
                [squeeze_transform]
                + [create_transform_step(c, level_hidden_channels) for _ in range(steps_per_level)]
                + [transforms.OneByOneConvolution(c)] # End each level with a linear transformation.
            )

            new_shape = mct.add_transform(transform_level, (c, h, w))
            if new_shape:  # If not last layer
                c, h, w = new_shape
    else:
        all_transforms = []

        for level, level_hidden_channels in zip(range(levels), hidden_channels):
            squeeze_transform = transforms.SqueezeTransform()
            c, h, w = squeeze_transform.get_output_shape(c, h, w)

            transform_level = transforms.CompositeTransform(
                [squeeze_transform]
                + [create_transform_step(c, level_hidden_channels) for _ in range(steps_per_level)]
                + [transforms.OneByOneConvolution(c)] # End each level with a linear transformation.
            )
            all_transforms.append(transform_level)

        all_transforms.append(ReshapeTransform(
            input_shape=(c,h,w),
            output_shape=(c*h*w,)
        ))
        mct = transforms.CompositeTransform(all_transforms)

    preprocess_transform = transforms.AffineScalarTransform(scale=(1. / 2 ** num_bits), shift=-0.5)
    return transforms.CompositeTransform([preprocess_transform, mct])

def create_flow(c, h, w, levels=3, hidden_channels=256, steps_per_level=10, num_bits=8, multi_scale=True):
    distribution = distributions.StandardNormal((c * h * w,)).to(device)
    transform = create_transform(
        c, h, w,
        levels=levels,
        hidden_channels=hidden_channels,
        steps_per_level=steps_per_level,
        num_bits=num_bits,
        multi_scale=multi_scale
    )
    flow = flows.Flow(transform, distribution)
    return flow

class Foo:
    def __init__(self, bar="aaa"):
        self = (bar)

class Glow:
    def __init__(self, c, h, w, levels=3, hidden_channels=256, steps_per_level=10, num_bits=8, multi_scale=True):
        self.shape = (c, h, w)
        distribution = distributions.StandardNormal((c * h * w,)).to(device)
        transform = create_transform(
            c, h, w,
            levels=levels,
            hidden_channels=hidden_channels,
            steps_per_level=steps_per_level,
            num_bits=num_bits,
            multi_scale=multi_scale
        )
        self.flow = flows.Flow(transform, distribution)
    
    def train(self, train_set, batch_size=128, max_epochs=50, log_interval=10, learning_rate=5e-4, weight_decay=1e-2):
        c, h, w = self.shape
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(self.flow.parameters(), lr=learning_rate, weight_decay=weight_decay)
        loss_list = []
        for epoch in range(1, max_epochs+1):
            loss_all = torch.tensor(0, dtype=torch.float, device=device)
            for data_batch, labels in train_loader:
                self.flow.train()
                optimizer.zero_grad()
                log_density = self.flow.log_prob(inputs=data_batch)
                loss = -nats_to_bits_per_dim(torch.mean(log_density), c, h, w)
                loss.backward()
                optimizer.step()
                loss_all += loss.sum()
            loss_list.append(loss_all.item()/len(train_loader.dataset))
            if epoch % log_interval == 0:
                print(f" - Epoch {epoch:3d}: {loss.item():.3f}")
        return loss_list
    
    def sample(self, n_samples=100):
        return self.flow.sample_and_log_prob(n_samples)