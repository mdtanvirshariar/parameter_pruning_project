"""
Advanced Model Analysis
- FLOPs calculation
- Model size analysis
- Inference time measurement
- Memory usage
- Architecture visualization
"""

import torch
import torch.nn as nn
import time
import numpy as np
from model import SimpleCNN

def calculate_flops(model, input_size=(1, 3, 32, 32)):
    """Calculate FLOPs (Floating Point Operations)"""
    try:
        flops = 0
        model.eval()
        device = next(model.parameters()).device
        
        def conv_flop_count(layer, x):
            try:
                batch_size = x.size(0)
                output_dims = x.size()[2:]
                kernel_dims = layer.kernel_size
                in_channels = layer.in_channels
                out_channels = layer.out_channels
                groups = layer.groups
                
                filters_per_channel = out_channels // groups
                conv_per_position_flops = int(np.prod(kernel_dims)) * in_channels * filters_per_channel
                active_elements_count = batch_size * int(np.prod(output_dims))
                overall_conv_flops = conv_per_position_flops * active_elements_count
                
                bias_flops = 0
                if layer.bias is not None:
                    bias_flops = out_channels * active_elements_count
                
                overall_flops = overall_conv_flops + bias_flops
                return overall_flops
            except Exception as e:
                return 0
        
        def linear_flop_count(layer, x):
            try:
                return x.size(0) * layer.in_features * layer.out_features
            except:
                return 0
        
        x = torch.randn(input_size).to(device)
        
        with torch.no_grad():
            for name, module in model.named_modules():
                try:
                    if isinstance(module, nn.Conv2d):
                        flops += conv_flop_count(module, x)
                        # Update x for next layer
                        x = module(x)
                    elif isinstance(module, nn.Linear):
                        if x.dim() > 2:
                            x = x.view(x.size(0), -1)
                        flops += linear_flop_count(module, x)
                        x = module(x)
                    elif isinstance(module, (nn.ReLU, nn.MaxPool2d, nn.Flatten)):
                        x = module(x)
                except Exception:
                    # Skip modules that cause errors
                    continue
        
        return flops
    except Exception as e:
        # Return a default value if calculation fails
        raise Exception(f"FLOPs calculation failed: {str(e)}")

def measure_inference_time(model, input_size=(1, 3, 32, 32), num_runs=100, device='cpu'):
    """Measure average inference time"""
    try:
        model.eval()
        device_obj = torch.device(device)
        model.to(device_obj)
        x = torch.randn(input_size).to(device_obj)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                try:
                    _ = model(x)
                except Exception:
                    break
        
        # Measure
        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.synchronize()
        
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                try:
                    if device == 'cuda' and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    start = time.time()
                    _ = model(x)
                    if device == 'cuda' and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    times.append(time.time() - start)
                except Exception:
                    break
        
        if len(times) == 0:
            return {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0
            }
        
        return {
            'mean': np.mean(times) * 1000,  # ms
            'std': np.std(times) * 1000,
            'min': np.min(times) * 1000,
            'max': np.max(times) * 1000
        }
    except Exception as e:
        # Return default values if measurement fails
        return {
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0
        }

def get_model_size_mb(model):
    """Get model size in MB"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

def analyze_model_architecture(model):
    """Analyze model architecture in detail"""
    architecture = {
        'layers': [],
        'total_params': 0,
        'trainable_params': 0,
        'non_trainable_params': 0
    }
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf module
            num_params = sum(p.numel() for p in module.parameters())
            if num_params > 0:
                architecture['layers'].append({
                    'name': name,
                    'type': type(module).__name__,
                    'params': num_params,
                    'shape': [list(p.shape) for p in module.parameters() if p.requires_grad]
                })
    
    architecture['total_params'] = sum(p.numel() for p in model.parameters())
    architecture['trainable_params'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    architecture['non_trainable_params'] = architecture['total_params'] - architecture['trainable_params']
    
    return architecture

def compare_model_complexity(model1_state, model2_state, model_class=SimpleCNN):
    """Compare complexity of two models"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model1 = model_class().to(device)
    model2 = model_class().to(device)
    model1.load_state_dict(model1_state)
    model2.load_state_dict(model2_state)
    
    results = {
        'model1': {},
        'model2': {}
    }
    
    for model, name in [(model1, 'model1'), (model2, 'model2')]:
        results[name]['flops'] = calculate_flops(model)
        results[name]['size_mb'] = get_model_size_mb(model)
        results[name]['params'] = sum(p.numel() for p in model.parameters())
        results[name]['inference_time'] = measure_inference_time(model, device=device)
    
    return results

