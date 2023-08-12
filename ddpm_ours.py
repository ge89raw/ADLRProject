import argparse
import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import numpy as np

from grasp_object_dataset import graspDataset
from positional_embeddings import PositionalEmbedding


class Block(nn.Module):
    def __init__(self, size: int):
        super().__init__()

        self.ff = nn.Linear(size, size)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor):
        return x + self.act(self.ff(x))


class MLP(nn.Module):
    def __init__(self, hidden_size: int = 128, hidden_layers: int = 3, emb_size: int = 128, in_grasp_classes: int = 3, out_classes: int = 28, distance_emb_size: int=512,
                 time_emb: str = "sinusoidal", input_emb: str = "sinusoidal"):
        super().__init__()

        self.emb_size = emb_size

        self.time_mlp = PositionalEmbedding(emb_size, time_emb)
        self.input_mlp = PositionalEmbedding(emb_size, input_emb, scale=2500.0) 
        

        self.conv_layers = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(64*4*4*4, distance_emb_size) # emb size for distances = 512
        )
        
    
        concat_size = len(self.time_mlp.layer) + \
            len(self.input_mlp.layer)* out_classes + in_grasp_classes + distance_emb_size #28 is the number of joint angles + 3 of the one_hot vector + 512 for the distance matrix
        layers = [nn.Linear(concat_size, hidden_size), nn.GELU()]
        for _ in range(hidden_layers):
            layers.append(Block(hidden_size))
        layers.append(nn.Linear(hidden_size, out_classes)) # This is the dimension ( it was 2 in the example model) it is 28 in our case (joint angles+rotation+translation)
        self.joint_mlp = nn.Sequential(*layers)


    def forward(self, x, t, label, object): 
        # embedding of the time vector:
        t_emb = self.time_mlp(t)

        # embedding of the x vector (our joint angles):
        x_emb = self.input_mlp(x) # 4x28x128
        
        x_emb = x_emb.reshape(x.size(0),-1)
        
        # padding our one hot vector (label):
        # extend the one hot vector: original one hot vector dim = 3, we want one hot vector dim = emb_size
        label_tensor= torch.tensor(label, dtype=torch.float32)
        #zeros_tensor = torch.zeros(label_tensor.size(0),self.emb_size-label_tensor.size(1), dtype=torch.float32)
        #one_hot_vector = torch.cat((label_tensor, zeros_tensor), dim=-1)

        # passing the distance matrix through 3D conv network
        distance_mesh = self.conv_layers(object.unsqueeze(1).to(torch.float32))

        # concatenating our data: time + joint angles + one hot vector
        x = torch.cat((x_emb, t_emb, label_tensor, distance_mesh), dim=-1).to(torch.float32)
        
        x = self.joint_mlp(x)
        return x


class NoiseScheduler():
    def __init__(self,
                 num_timesteps=1000,
                 beta_start=0.0001,
                 beta_end=0.02,
                 beta_schedule="linear"):

        self.num_timesteps = num_timesteps
        if beta_schedule == "linear":
            self.betas = torch.linspace(
                beta_start, beta_end, num_timesteps, dtype=torch.float32)
        elif beta_schedule == "quadratic":
            self.betas = torch.linspace(
                beta_start ** 0.5, beta_end ** 0.5, num_timesteps, dtype=torch.float32) ** 2

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.)

        # required for self.add_noise
        self.sqrt_alphas_cumprod = self.alphas_cumprod ** 0.5
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod) ** 0.5

        # required for reconstruct_x0
        self.sqrt_inv_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod)
        self.sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(
            1 / self.alphas_cumprod - 1)

        # required for q_posterior
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)

    def reconstruct_x0(self, x_t, t, noise):
        s1 = self.sqrt_inv_alphas_cumprod[t]
        s2 = self.sqrt_inv_alphas_cumprod_minus_one[t]
        s1 = s1.reshape(-1, 1).cuda()
        s2 = s2.reshape(-1, 1).cuda() 

        return s1 * x_t - s2 * noise

    def q_posterior(self, x_0, x_t, t):
        s1 = self.posterior_mean_coef1[t]
        s2 = self.posterior_mean_coef2[t]
        s1 = s1.reshape(-1, 1).cuda()
        s2 = s2.reshape(-1, 1).cuda()
        
        mu = s1 * x_0 + s2 * x_t
        return mu

    def get_variance(self, t):
        if t == 0:
            return 0

        variance = self.betas[t] * (1. - self.alphas_cumprod_prev[t]) / (1. - self.alphas_cumprod[t])
        variance = variance.clip(1e-20)
        return variance

    def step(self, model_output, timestep, sample):
        t = timestep
        pred_original_sample = self.reconstruct_x0(sample, t, model_output)
        pred_prev_sample = self.q_posterior(pred_original_sample, sample, t)

        variance = 0
        if t > 0:
            noise = torch.randn_like(model_output)
            variance = (self.get_variance(t) ** 0.5) * noise

        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample

    def add_noise(self, x_start, x_noise, timesteps):
        s1 = self.sqrt_alphas_cumprod[timesteps]
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps]

        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)

        return s1 * x_start + s2 * x_noise

    def __len__(self):
        return self.num_timesteps

# Function to obtain mean, std, max, and min of given dataset
def get_mean_std(main_dir, object_dir, dataset):

    # Allocate variables
    full_joints = []

    # Load all samples - only joint values!
    for sample in tqdm(dataset):
        joints = sample[0]
        full_joints.append(joints)

    mean = np.mean(full_joints, axis = 0)
    std = np.std(full_joints, axis = 0)
    max = np.max(full_joints, axis = 0)
    min = np.min(full_joints, axis = 0)
    

    return mean, std, max, min

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="base")
    parser.add_argument("--main_dir", type=str, default="./dataset_grasps/")
    parser.add_argument("--object_dir", type=str, default="./dataset_objects/")
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_timesteps", type=int, default=50)
    parser.add_argument("--beta_schedule", type=str, default="linear", choices=["linear", "quadratic"])
    parser.add_argument("--embedding_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--hidden_layers", type=int, default=3)
    parser.add_argument("--time_embedding", type=str, default="sinusoidal", choices=["sinusoidal", "learnable", "linear", "zero"])
    parser.add_argument("--input_embedding", type=str, default="sinusoidal", choices=["sinusoidal", "learnable", "linear", "identity"])
    parser.add_argument("--out_classes", type=int, default=28)
    parser.add_argument("--input_grasp_classes", type=int, default=3)
    parser.add_argument("--dist_emb_size", type=int, default=512)

    config = parser.parse_args()

    # Generate dataset with all dataset samples
    main_dataset = graspDataset(config.main_dir, config.object_dir, mode = 'train', split = {'train': 1, 'val': 0, 'test': 0}, normalization=None, transform_joint = None, transform_object = None)
    # Get the mean and std for normalization
    mean_std_max_min = list(get_mean_std(config.main_dir, config.object_dir, main_dataset))
    # Generate train dataset and dataloader
    train_dataset = graspDataset(config.main_dir, config.object_dir, mode = 'train', split = {'train': 1, 'val': 0, 'test': 0}, normalization=mean_std_max_min)
    train_dataloader = DataLoader(train_dataset , batch_size=config.train_batch_size, shuffle=True, num_workers=2, drop_last=False)

    model = MLP(
        hidden_size=config.hidden_size,
        hidden_layers=config.hidden_layers,
        emb_size=config.embedding_size,
        time_emb=config.time_embedding,
        input_emb=config.input_embedding,
        in_grasp_classes=config.input_grasp_classes,
        out_classes=config.out_classes,
        distance_emb_size=config.dist_emb_size)

    noise_scheduler = NoiseScheduler(
        num_timesteps=config.num_timesteps,
        beta_schedule=config.beta_schedule)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
    )

    global_step = 0
    frames = []
    losses = []
    print("Training model...")

    for epoch in range(config.num_epochs):

        model.train()
        progress_bar = tqdm(total=len(train_dataloader))
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(train_dataloader):
            # Variables
            joint_angles_batch = batch[0] 
            label_one_hot = batch[1]
            mat_distances = batch[2] 

            noise = torch.randn(joint_angles_batch.shape)
            timesteps = torch.randint(
                0, noise_scheduler.num_timesteps, (joint_angles_batch.shape[0],)
            ).long()

            noisy = noise_scheduler.add_noise(joint_angles_batch, noise, timesteps)
            noise_pred = model(noisy, timesteps, label_one_hot, mat_distances)
            loss = F.mse_loss(noise_pred, noise)
            loss.backward(loss)

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "step": global_step}
            losses.append(loss.detach().item())
            progress_bar.set_postfix(**logs)
            global_step += 1
        progress_bar.close()


    print("Saving model...")
    outdir = f"results/{config.experiment_name}"
    os.makedirs(outdir, exist_ok=True)
    torch.save(model.state_dict(), f"{outdir}/model.pth")

    print("Saving loss as numpy array...")
    np.save(f"{outdir}/loss.npy", np.array(losses))
