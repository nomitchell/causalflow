# visualize_disentanglement.py
# PURPOSE: To validate the effectiveness of the Stage 1 CausalEncoder by
# visualizing the latent spaces of the causal factor 's' and the style factor 'z'.
# This script replicates the t-SNE visualization from Figure 4 of the CausalDiff paper.
#
# HOW TO RUN:
# python visualize_disentanglement.py --config configs/cifar10_causalflow.yml --encoder_checkpoint ./checkpoints/causal_encoder_best.pt --num_samples 5000 --output_file ./latent_space_visualization.png

import argparse
import yaml
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm

# --- Model and Data Imports ---
from models.encoder import CausalEncoder
from data.cifar10 import CIFAR10

def get_config_and_setup(args):
    """Load configuration from YAML and set up device."""
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    config_obj = type('Config', (), {})()
    for key, value in config.items():
        setattr(config_obj, key, value)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config_obj.device = device
    
    print(f"--- Latent Space Visualization ---")
    print(f"Using device: {device}")
    
    return config_obj

def load_encoder(config, checkpoint_path):
    """Loads the pre-trained and frozen CausalEncoder from Stage 1."""
    print(f"Loading CausalEncoder from {checkpoint_path}")
    encoder = CausalEncoder(s_dim=config.s_dim, z_dim=config.z_dim).to(config.device)
    
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    
    encoder.eval()
    print("CausalEncoder loaded.")
    return encoder

def main():
    parser = argparse.ArgumentParser(description="Visualize Causal Encoder Latent Space")
    parser.add_argument('--config', type=str, default='configs/cifar10.yml', help='Path to the config file.')
    parser.add_argument('--encoder_checkpoint', type=str, default='checkpoints/causal_encoder_best.pt', help='Path to the trained CausalEncoder checkpoint.')
    parser.add_argument('--num_samples', type=int, default=5000, help='Number of test samples to visualize.')
    parser.add_argument('--output_file', type=str, default='latent_space_visualization.png', help='Path to save the output plot.')
    args = parser.parse_args()
    
    config = get_config_and_setup(args)
    
    # --- Load Model and Data ---
    encoder = load_encoder(config, args.encoder_checkpoint)
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=CIFAR10.get_test_transform())
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)

    # --- Generate Latent Vectors ---
    print(f"Generating latent vectors for {args.num_samples} samples...")
    all_s = []
    all_z = []
    all_labels = []
    
    with torch.no_grad():
        for x_batch, y_batch in tqdm(test_loader, desc="Encoding samples"):
            if len(all_labels) >= args.num_samples:
                break
            x_batch = x_batch.to(config.device)
            s, z, _, _ = encoder(x_batch)
            
            all_s.append(s.cpu().numpy())
            all_z.append(z.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())

    all_s = np.concatenate(all_s, axis=0)[:args.num_samples]
    all_z = np.concatenate(all_z, axis=0)[:args.num_samples]
    all_sz = np.concatenate([all_s, all_z], axis=1)
    all_labels = np.concatenate(all_labels, axis=0)[:args.num_samples]

    print("Latent vectors generated. Running t-SNE...")

    # --- Run t-SNE ---
    # It's recommended to run t-SNE on each latent space separately
    tsne = TSNE(n_components=2, verbose=1, perplexity=40)
    
    s_2d = tsne.fit_transform(all_s)
    print("t-SNE for 's' complete.")
    
    tsne = TSNE(n_components=2, verbose=1, perplexity=40)
    z_2d = tsne.fit_transform(all_z)
    print("t-SNE for 'z' complete.")

    tsne = TSNE(n_components=2, verbose=1, perplexity=40)
    sz_2d = tsne.fit_transform(all_sz)
    print("t-SNE for '[s,z]' complete.")

    # --- Plotting ---
    print("Generating plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))
    
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # Use a distinct color map
    colors = list(mcolors.TABLEAU_COLORS.values())

    # Plot 1: Causal Factor 's'
    for i in range(config.num_classes):
        indices = all_labels == i
        ax1.scatter(s_2d[indices, 0], s_2d[indices, 1], c=colors[i], label=cifar10_classes[i], s=10, alpha=0.7)
    ax1.set_title("Label-Causative Factor (s)", fontsize=16)
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Plot 2: Style Factor 'z'
    for i in range(config.num_classes):
        indices = all_labels == i
        ax2.scatter(z_2d[indices, 0], z_2d[indices, 1], c=colors[i], s=10, alpha=0.7)
    ax2.set_title("Label-Non-Causative Factor (z)", fontsize=16)
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Plot 3: Concatenation '[s, z]'
    for i in range(config.num_classes):
        indices = all_labels == i
        ax3.scatter(sz_2d[indices, 0], sz_2d[indices, 1], c=colors[i], s=10, alpha=0.7)
    ax3.set_title("Concatenated Latent Space [s, z]", fontsize=16)
    ax3.set_xticks([])
    ax3.set_yticks([])

    # Create a single legend for the figure
    fig.legend(handles=ax1.get_legend_handles_labels()[0], 
               labels=cifar10_classes, 
               loc='upper center', 
               ncol=config.num_classes, 
               bbox_to_anchor=(0.5, 1.02),
               fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for the legend
    plt.savefig(args.output_file, dpi=300, bbox_inches='tight')
    
    print(f"\nPlot saved successfully to {args.output_file}")
    print("Check the plot: 's' should show class-based clusters, 'z' should look random.")

if __name__ == '__main__':
    main()
