"""
Advanced Visualization Techniques
- Weight distributions
- Activation maps
- Filter visualizations
- t-SNE embeddings
- Gradient flow
- Layer-wise analysis
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os

def plot_weight_distributions(state_dict, out_dir='assets', prefix='weights'):
    """Plot weight distributions for all layers"""
    os.makedirs(out_dir, exist_ok=True)
    for k, v in state_dict.items():
        if 'weight' in k:
            arr = v.cpu().numpy().flatten()
            plt.figure(figsize=(8, 5))
            plt.hist(arr, bins=100, alpha=0.7, edgecolor='black')
            plt.title(f'Weight Distribution: {k}', fontsize=12, fontweight='bold')
            plt.xlabel('Weight Value')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            fname = os.path.join(out_dir, f"{prefix}_{k.replace('.','_')}_dist.png")
            plt.savefig(fname, dpi=150)
            plt.close()

def plot_weight_heatmap(state_dict, out_dir='assets', prefix='weights'):
    """Plot weight heatmaps for convolutional layers"""
    os.makedirs(out_dir, exist_ok=True)
    for k, v in state_dict.items():
        if 'weight' in k and v.dim() == 4:  # Conv layers
            # Average across input channels
            avg_weights = v.cpu().numpy().mean(axis=1)
            # Take first few filters for visualization
            num_filters = min(16, avg_weights.shape[0])
            
            fig, axes = plt.subplots(4, 4, figsize=(12, 12))
            axes = axes.flatten()
            
            for i in range(num_filters):
                im = axes[i].imshow(avg_weights[i], cmap='viridis', aspect='auto')
                axes[i].set_title(f'Filter {i}', fontsize=8)
                axes[i].axis('off')
                plt.colorbar(im, ax=axes[i], fraction=0.046)
            
            for i in range(num_filters, len(axes)):
                axes[i].axis('off')
            
            plt.suptitle(f'Filter Heatmaps: {k}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            fname = os.path.join(out_dir, f"{prefix}_{k.replace('.','_')}_heatmap.png")
            plt.savefig(fname, dpi=150)
            plt.close()

def plot_sparsity_analysis(state_dict, out_dir='assets', prefix='sparsity'):
    """Analyze and plot sparsity across layers"""
    os.makedirs(out_dir, exist_ok=True)
    
    layer_names = []
    sparsity_values = []
    param_counts = []
    
    for k, v in state_dict.items():
        if 'weight' in k:
            total = v.numel()
            zeros = (v == 0).sum().item()
            sparsity = zeros / total if total > 0 else 0
            
            layer_names.append(k.replace('.weight', ''))
            sparsity_values.append(sparsity * 100)
            param_counts.append(total)
    
    # Plot sparsity by layer
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Sparsity bar chart
    ax1.barh(range(len(layer_names)), sparsity_values, color='coral')
    ax1.set_yticks(range(len(layer_names)))
    ax1.set_yticklabels(layer_names, fontsize=9)
    ax1.set_xlabel('Sparsity (%)')
    ax1.set_title('Sparsity by Layer', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Parameter count
    ax2.barh(range(len(layer_names)), param_counts, color='steelblue')
    ax2.set_yticks(range(len(layer_names)))
    ax2.set_yticklabels(layer_names, fontsize=9)
    ax2.set_xlabel('Parameter Count')
    ax2.set_title('Parameters by Layer', fontweight='bold')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    fname = os.path.join(out_dir, f"{prefix}_analysis.png")
    plt.savefig(fname, dpi=150)
    plt.close()

def plot_layer_statistics(state_dict, out_dir='assets', prefix='stats'):
    """Plot comprehensive layer statistics"""
    os.makedirs(out_dir, exist_ok=True)
    
    layers = []
    means = []
    stds = []
    mins = []
    maxs = []
    
    for k, v in state_dict.items():
        if 'weight' in k:
            arr = v.cpu().numpy().flatten()
            layers.append(k.replace('.weight', ''))
            means.append(arr.mean())
            stds.append(arr.std())
            mins.append(arr.min())
            maxs.append(arr.max())
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    x = range(len(layers))
    
    axes[0, 0].bar(x, means, color='skyblue', alpha=0.7)
    axes[0, 0].set_title('Mean Weight Values', fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(layers, rotation=45, ha='right', fontsize=8)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    axes[0, 1].bar(x, stds, color='lightcoral', alpha=0.7)
    axes[0, 1].set_title('Weight Standard Deviation', fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(layers, rotation=45, ha='right', fontsize=8)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    axes[1, 0].bar(x, mins, color='lightgreen', alpha=0.7)
    axes[1, 0].set_title('Minimum Weight Values', fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(layers, rotation=45, ha='right', fontsize=8)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    axes[1, 1].bar(x, maxs, color='plum', alpha=0.7)
    axes[1, 1].set_title('Maximum Weight Values', fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(layers, rotation=45, ha='right', fontsize=8)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fname = os.path.join(out_dir, f"{prefix}_layer_stats.png")
    plt.savefig(fname, dpi=150)
    plt.close()

def visualize_activations(model, dataloader, device, out_dir='assets', prefix='activations'):
    """Visualize intermediate layer activations"""
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    
    activations = {}
    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output.detach().cpu()
        return hook
    
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    # Forward pass
    with torch.no_grad():
        images, _ = next(iter(dataloader))
        images = images[:4].to(device)  # Use 4 samples
        _ = model(images)
    
    # Plot activations
    for name, act in activations.items():
        if act.dim() >= 2:
            # For conv layers, show feature maps
            if act.dim() == 4:
                num_maps = min(16, act.size(1))
                fig, axes = plt.subplots(4, 4, figsize=(12, 12))
                axes = axes.flatten()
                
                for i in range(num_maps):
                    im = axes[i].imshow(act[0, i].numpy(), cmap='hot', aspect='auto')
                    axes[i].set_title(f'Map {i}', fontsize=8)
                    axes[i].axis('off')
                
                for i in range(num_maps, len(axes)):
                    axes[i].axis('off')
                
                plt.suptitle(f'Activations: {name}', fontsize=14, fontweight='bold')
                plt.tight_layout()
                fname = os.path.join(out_dir, f"{prefix}_{name.replace('.','_')}.png")
                plt.savefig(fname, dpi=150)
                plt.close()
    
    # Remove hooks
    for hook in hooks:
        hook.remove()

def plot_pruning_comparison(before_dict, after_dict, out_dir='assets', prefix='comparison'):
    """Compare before and after pruning"""
    os.makedirs(out_dir, exist_ok=True)
    
    layers = []
    before_sparsity = []
    after_sparsity = []
    
    for k in before_dict:
        if 'weight' in k and k in after_dict:
            before = before_dict[k]
            after = after_dict[k]
            
            before_zeros = (before == 0).sum().item()
            after_zeros = (after == 0).sum().item()
            total = before.numel()
            
            layers.append(k.replace('.weight', ''))
            before_sparsity.append(before_zeros / total * 100)
            after_sparsity.append(after_zeros / total * 100)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(layers))
    width = 0.35
    
    ax.bar(x - width/2, before_sparsity, width, label='Before Pruning', color='lightblue', alpha=0.8)
    ax.bar(x + width/2, after_sparsity, width, label='After Pruning', color='coral', alpha=0.8)
    
    ax.set_xlabel('Layer')
    ax.set_ylabel('Sparsity (%)')
    ax.set_title('Sparsity Comparison: Before vs After Pruning', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(layers, rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fname = os.path.join(out_dir, f"{prefix}_pruning.png")
    plt.savefig(fname, dpi=150)
    plt.close()

