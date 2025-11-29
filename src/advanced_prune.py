"""
Advanced Pruning Techniques
- Magnitude-based pruning
- L1/L2 regularization pruning
- Structured pruning (channel/filter)
- Gradient-based pruning
"""

import torch
import torch.nn as nn
import numpy as np

def magnitude_prune(state_dict, amount):
    """Standard magnitude-based pruning"""
    all_weights = []
    for k, v in state_dict.items():
        if 'weight' in k and v.dim() > 1:
            all_weights.append(v.abs().flatten())
    if len(all_weights) == 0:
        return state_dict
    
    all_weights = torch.cat(all_weights)
    cutoff = torch.quantile(all_weights, amount)
    
    new_state = {}
    for k, v in state_dict.items():
        if 'weight' in k and v.dim() > 1:
            mask = (v.abs() > cutoff).float()
            new_state[k] = v * mask
        else:
            new_state[k] = v
    return new_state

def l1_prune(state_dict, amount):
    """L1 norm-based pruning"""
    all_l1 = []
    for k, v in state_dict.items():
        if 'weight' in k and v.dim() > 1:
            all_l1.append(v.abs().sum(dim=tuple(range(1, v.dim()))).flatten())
    
    if len(all_l1) == 0:
        return state_dict
    
    all_l1 = torch.cat(all_l1)
    cutoff = torch.quantile(all_l1, amount)
    
    new_state = {}
    for k, v in state_dict.items():
        if 'weight' in k and v.dim() > 1:
            l1_norms = v.abs().sum(dim=tuple(range(1, v.dim())))
            mask = (l1_norms > cutoff).float()
            if v.dim() == 2:  # FC layer
                mask = mask.unsqueeze(1).expand_as(v)
            elif v.dim() == 4:  # Conv layer
                mask = mask.view(-1, 1, 1, 1).expand_as(v)
            new_state[k] = v * mask
        else:
            new_state[k] = v
    return new_state

def structured_channel_prune(state_dict, amount, model):
    """Structured pruning - remove entire channels"""
    device = next(iter(state_dict.values())).device
    model.load_state_dict(state_dict)
    model.eval()
    
    # Calculate channel importance
    channel_importance = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Use L1 norm of filters as importance
            importance = module.weight.data.abs().sum(dim=(1, 2, 3))
            channel_importance[name] = importance
    
    # Prune channels
    new_state = state_dict.copy()
    for name, importance in channel_importance.items():
        num_channels = len(importance)
        num_prune = int(num_channels * amount)
        if num_prune > 0:
            _, indices = torch.topk(importance, num_channels - num_prune)
            # This is simplified - full implementation would require model restructuring
            # For now, zero out pruned channels
            for k, v in new_state.items():
                if name in k and 'weight' in k and v.dim() == 4:
                    mask = torch.zeros(v.size(0), dtype=torch.float32, device=device)
                    mask[indices] = 1.0
                    mask = mask.view(-1, 1, 1, 1)
                    new_state[k] = v * mask
    
    return new_state

def gradient_based_prune(state_dict, model, dataloader, amount, device):
    """Gradient-based importance pruning"""
    model.load_state_dict(state_dict)
    model.train()
    
    # Compute gradients
    gradients = {}
    criterion = nn.CrossEntropyLoss()
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.grad is not None and 'weight' in name:
                if name not in gradients:
                    gradients[name] = param.grad.abs()
                else:
                    gradients[name] += param.grad.abs()
        
        model.zero_grad()
        break  # Use one batch for speed
    
    # Normalize and prune
    all_grads = torch.cat([g.flatten() for g in gradients.values()])
    cutoff = torch.quantile(all_grads, amount)
    
    new_state = {}
    for k, v in state_dict.items():
        if k in gradients and v.dim() > 1:
            mask = (gradients[k] > cutoff).float()
            new_state[k] = v * mask
        else:
            new_state[k] = v
    
    return new_state

def random_prune(state_dict, amount):
    """Random pruning (baseline)"""
    new_state = {}
    for k, v in state_dict.items():
        if 'weight' in k and v.dim() > 1:
            mask = torch.rand_like(v) > amount
            new_state[k] = v * mask.float()
        else:
            new_state[k] = v
    return new_state

