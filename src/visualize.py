
import torch
import matplotlib.pyplot as plt
import numpy as np
from model import SimpleCNN
import os

def plot_weight_histograms(state_dict, out_dir='assets', prefix='weights'):
    os.makedirs(out_dir, exist_ok=True)
    for k,v in state_dict.items():
        if 'weight' in k:
            arr = v.cpu().numpy().flatten()
            plt.figure(figsize=(4,3))
            plt.hist(arr, bins=80)
            plt.title(k)
            plt.tight_layout()
            fname = os.path.join(out_dir, f"{prefix}_{k.replace('.','_')}.png")
            plt.savefig(fname)
            plt.close()

def visualize(model_path, out_dir='assets', prefix='baseline'):
    device = 'cpu'
    state = torch.load(model_path, map_location=device)
    plot_weight_histograms(state, out_dir=out_dir, prefix=prefix)
    print('Saved weight histograms to', out_dir)

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--out-dir', type=str, default='assets')
    parser.add_argument('--prefix', type=str, default='weights')
    args = parser.parse_args()
    visualize(args.model_path, args.out_dir, args.prefix)
