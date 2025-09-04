# pip install torch matplotlib numpy
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import pandas as pd

# -----------------------------
# 1) Prepare dataset
# -----------------------------
class TimeStrainColors(Dataset):
    """
    Expects:
      - times: 1D array-like of length N (shared across strains)
      - data_dict: {strain_name: torch.Tensor of shape (N, 36, 3)}
    Outputs:
      - X = (t_norm: float32, strain_id: long)
      - Y = (36*3 normalized RGB flattened to 108)
    """
    def __init__(self, times, data_dict, normalize=True):
        self.strain_names = sorted(list(data_dict.keys()))
        self.strain_to_id = {s:i for i,s in enumerate(self.strain_names)}
        self.times = torch.as_tensor(times, dtype=torch.float32)  # (N,)
        self.N = self.times.shape[0]

        # Stack all strains along first dim of samples: total samples = N * S
        tensors = []
        for s in self.strain_names:
            T = torch.as_tensor(data_dict[s], dtype=torch.float32)  # (N,36,3)
            tensors.append(T)
        # shape (S, N, 36, 3)
        self.all = torch.stack(tensors, dim=0)
        self.S = self.all.shape[0]

        # Build min/max for normalization (across all strains, times, channels)
        self.normalize = normalize
        if normalize:
            self.y_min = self.all.amin(dim=(0,1,2,3))
            self.y_max = self.all.amax(dim=(0,1,2,3))
            # Fall back if flat:
            if torch.isclose(self.y_min, self.y_max):
                self.y_min, self.y_max = torch.tensor(0.0), torch.tensor(1.0)
        else:
            self.y_min, self.y_max = torch.tensor(0.0), torch.tensor(1.0)

        # Time normalization to [0,1]
        tmin, tmax = self.times.min(), self.times.max()
        self.tmin = float(tmin)
        self.tmax = float(tmax) if float(tmax) != float(tmin) else float(tmin + 1.0)

    def __len__(self):
        return self.N * self.S

    def __getitem__(self, idx):
        # Map flat idx to (strain_id, time_idx)
        strain_id = idx // self.N
        t_idx     = idx % self.N

        t_raw = self.times[t_idx]
        t_norm = (t_raw - self.tmin) / (self.tmax - self.tmin)

        y = self.all[strain_id, t_idx]  # (36,3)
        if self.normalize:
            y = (y - self.y_min) / (self.y_max - self.y_min + 1e-8)
            y = torch.clamp(y, 0.0, 1.0)

        y = y.reshape(-1)  # (108,)
        return torch.tensor([t_norm], dtype=torch.float32), torch.tensor(strain_id, dtype=torch.long), y

    def denorm(self, y):  # y shape (..., 36, 3)
        if not self.normalize:
            return y
        return y * (self.y_max - self.y_min) + self.y_min


# -----------------------------
# 2) Model: (time, strain_id) -> 108
# -----------------------------
class ColorPredictor(nn.Module):
    def __init__(self, num_strains, time_in_dim=1, emb_dim=16, hidden=256, out_dim=108):
        super().__init__()
        self.emb = nn.Embedding(num_strains, emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(time_in_dim + emb_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
        # We'll squash to [0,1] with sigmoid if we trained on normalized RGB
        self.final = nn.Sigmoid()

    def forward(self, t_norm, strain_id):
        """
        t_norm: (B,1) float
        strain_id: (B,) long
        returns: (B,108) in [0,1]
        """
        e = self.emb(strain_id)          # (B, emb_dim)
        x = torch.cat([t_norm, e], dim=1)
        y = self.mlp(x)
        y = self.final(y)
        return y


# -----------------------------
# 3) Training utilities
# -----------------------------
def train_model(dataset, epochs=50, batch_size=128, lr=1e-3, val_ratio=0.2, seed=42, device="cpu"):
    torch.manual_seed(seed)
    N = len(dataset)
    n_val = int(math.ceil(N * val_ratio))
    n_train = N - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False)

    model = ColorPredictor(num_strains=len(dataset.strain_names)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None

    for ep in range(1, epochs+1):
        model.train()
        tr_loss = 0.0
        for t_norm, sid, y in train_loader:
            t_norm, sid, y = t_norm.to(device), sid.to(device), y.to(device)
            y_hat = model(t_norm, sid)
            loss = loss_fn(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item() * t_norm.size(0)
        tr_loss /= len(train_set)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for t_norm, sid, y in val_loader:
                t_norm, sid, y = t_norm.to(device), sid.to(device), y.to(device)
                y_hat = model(t_norm, sid)
                loss = loss_fn(y_hat, y)
                val_loss += loss.item() * t_norm.size(0)
        val_loss /= len(val_set)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"Epoch {ep:03d} | train MSE: {tr_loss:.6f} | val MSE: {val_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# -----------------------------
# 4) Rendering helpers
# -----------------------------
def render_6x6(colors_36x3, title=None, square_size=30):
    """
    colors_36x3: np.array or torch.tensor of shape (36,3) in [0,1]
    Displays a 6x6 grid where each cell is filled with its RGB.
    """
    if isinstance(colors_36x3, torch.Tensor):
        colors = colors_36x3.detach().cpu().numpy()
    else:
        colors = np.asarray(colors_36x3)

    # Build an image by tiling each color into a small square
    H = W = square_size
    img_rows = []
    for r in range(6):
        row_blocks = []
        for c in range(6):
            idx = r * 6 + c
            block = np.ones((H, W, 3), dtype=float) * colors[idx][None,None,:]
            row_blocks.append(block)
        img_rows.append(np.concatenate(row_blocks, axis=1))
    img = np.concatenate(img_rows, axis=0)

    plt.figure(figsize=(4,4))
    plt.imshow(img, aspect='equal')
    plt.axis('off')
    if title:
        plt.title(title, fontsize=10)
    plt.show()


def predict_colors(model, dataset, time_value, strain_name, device="cpu", denorm=True):
    """
    time_value: scalar in original time units (will be normalized internally)
    strain_name: one of dataset.strain_names
    Returns colors as (36,3) in [0,1] if dataset was normalized; otherwise denorm to raw scale.
    """
    model.eval()
    # normalize time consistently with dataset
    t_norm = (float(time_value) - dataset.tmin) / (dataset.tmax - dataset.tmin)
    t_norm = torch.tensor([[t_norm]], dtype=torch.float32, device=device)

    sid = torch.tensor([dataset.strain_to_id[strain_name]], dtype=torch.long, device=device)
    with torch.no_grad():
        y_hat = model(t_norm, sid)  # (1,108)
    colors = y_hat.view(36, 3).cpu()

    # If you need raw scale (unlikely for display), denormalize
    if denorm and dataset.normalize:
        colors = torch.clamp(colors, 0, 1)  # keep in [0,1] for display

    return colors


# -----------------------------
# 5) Example usage (replace with YOUR data)
# -----------------------------
if __name__ == "__main__":
    # ======= YOUR DATA HERE =======
    # times of length N (shared across strains)
    # data_dict: mapping from strain name -> tensor (N,36,3)
    # Below we synthesize dummy data for illustration.
    df = pd.read_csv('Cleaned_Table.csv')
    df.dropna(inplace=True)  # Ensure no NaNs
    rng = np.random.default_rng(0)
    times = list(range(120, 601, 30))
    mean_control = np.zeros((len(times),36, 3)) # [times, dye, rgb]
    mean_MRSA = np.zeros((len(times), 36, 3))
    # mean_b = np.zeros((len(times), 36, 3))
    for j in range(len(times)):
        for i in range(36):
            a_r = []
            b_r = []
            a_g = []
            b_g = []
            a_b = []
            b_b = []
            for k in range(14):
                a_r.append(df[f'Control {k+1}'][108*j + 3*i]) # Red
                a_g.append(df[f'Control {k+1}'][108*j + 3*i + 1]) # Green
                a_b.append(df[f'Control {k+1}'][108*j + 3*i + 2]) # Blue
            for k in range(10):
                b_r.append(df[f'MRSA {k+1}'][108*j + 3*i]) # Red
                b_g.append(df[f'MRSA {k+1}'][108*j + 3*i + 1]) # Green
                b_b.append(df[f'MRSA {k+1}'][108*j + 3*i + 2]) # Blue
            mean_control[j,i,0] = np.mean(a_r)
            mean_control[j,i,1] = np.mean(a_g)
            mean_control[j,i,2] = np.mean(a_b)

            mean_MRSA[j,i,0] = np.mean(b_r)
            mean_MRSA[j,i,1] = np.mean(b_g)
            mean_MRSA[j,i,2] = np.mean(b_b)
    # N = 200
    # times = np.linspace(0, 100, N)
    # strains = ["control", "MRSA", "Ecoli"]

    # def synth_colors(N, offset=0.0):
    #     # (N,36,3) smooth color changes over time per dye
    #     t = np.linspace(0, 4*np.pi, N)[:, None]  # (N,1)
    #     dyes = np.linspace(0, 2*np.pi, 36)[None, :]  # (1,36)
    #     base = np.sin(t + dyes)  # (N,36)
    #     R = 0.5 + 0.5*np.sin(t + dyes + offset)
    #     G = 0.5 + 0.5*np.sin(t + dyes + 1.0 + offset)
    #     B = 0.5 + 0.5*np.sin(t + dyes + 2.0 + offset)
    #     arr = np.stack([R, G, B], axis=-1)  # (N,36,3) in [0,1]
    #     # pretend there are deviations (stretch/shift), maybe outside [0,1]
    #     arr = (arr - 0.5) * (1.5 + 0.2*offset) + 0.2*offset
    #     return torch.tensor(arr, dtype=torch.float32)

    data_dict = {
        "control": mean_control,#synth_colors(N, offset=0.0),
        "MRSA":    mean_MRSA,#synth_colors(N, offset=0.6),
        # "Ecoli":   synth_colors(N, offset=1.1),
    }

    dataset = TimeStrainColors(times, data_dict, normalize=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = train_model(dataset, epochs=40, batch_size=256, lr=1e-3, val_ratio=0.2, device=device)

    # Predict for an arbitrary time and strain, render as 6x6 image
    t_query = 37.5
    for strain in dataset.strain_names:
        colors = predict_colors(model, dataset, time_value=t_query, strain_name=strain, device=device)
        render_6x6(colors, title=f"{strain} @ t={t_query}")
