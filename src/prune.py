
import torch, argparse, os
import numpy as np
from model import SimpleCNN

def magnitude_prune_state_dict(state_dict, amount):
    # amount: fraction to prune overall (0..1)
    # collect absolute values
    all_weights = []
    for k,v in state_dict.items():
        if 'weight' in k and v.dim()>1:
            all_weights.append(v.abs().flatten())
    if len(all_weights)==0:
        return state_dict
    all_weights = torch.cat(all_weights)
    cutoff = torch.quantile(all_weights, amount)
    new_state = {}
    for k,v in state_dict.items():
        if 'weight' in k and v.dim()>1:
            mask = (v.abs() > cutoff).float()
            new_state[k] = v * mask
        else:
            new_state[k] = v
    return new_state

def evaluate(model, testloader, device):
    model.eval()
    correct=0; total=0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _,pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
    return 100.*correct/total

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--prune-percent', type=float, default=0.5, help='fraction to prune (0..1)')
    parser.add_argument('--save-dir', type=str, default='saved')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN().to(device)

    # load testset
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    # Use 0 workers on Windows to avoid issues
    import platform
    num_workers = 0 if platform.system() == 'Windows' else 2
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=num_workers)

    state = torch.load(args.model_path, map_location=device)
    pruned_state = magnitude_prune_state_dict(state, args.prune_percent)
    model.load_state_dict(pruned_state)
    acc = evaluate(model, testloader, device)
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(pruned_state, os.path.join(args.save_dir, f'pruned_{int(args.prune_percent*100)}.pth'))
    print(f'Pruned model saved. Test accuracy after pruning: {acc:.2f}%')
