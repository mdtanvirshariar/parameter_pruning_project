import os
import argparse
import platform
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import SimpleCNN
from tqdm import tqdm

def train(args):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # Use subset for faster training if quick mode
    if args.quick_mode:
        # Use only 5% of training data for ultra-fast training
        from torch.utils.data import Subset
        subset_size = len(trainset) // 20  # 5% instead of 10%
        indices = torch.randperm(len(trainset))[:subset_size]
        trainset = Subset(trainset, indices)
        print(f"Quick mode: Using {subset_size} samples (5%) for ultra-fast training")
    
    # Optimize data loading
    num_workers = 0 if platform.system() == 'Windows' else 2
    pin_memory = torch.cuda.is_available()  # Faster GPU transfer
    
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, 
                            num_workers=num_workers, pin_memory=pin_memory, persistent_workers=num_workers>0,
                            prefetch_factor=2 if num_workers > 0 else None)
    # Use smaller test set for faster evaluation
    if args.quick_mode:
        from torch.utils.data import Subset
        test_subset_size = len(testset) // 10  # Use 10% of test set
        test_indices = torch.randperm(len(testset))[:test_subset_size]
        testset = Subset(testset, test_indices)
    testloader = DataLoader(testset, batch_size=512, shuffle=False, 
                           num_workers=num_workers, pin_memory=pin_memory)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN().to(device)
    
    # Use mixed precision for faster training (if GPU available)
    use_amp = torch.cuda.is_available() and hasattr(torch.cuda, 'amp')
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        print("Using mixed precision training for speed boost!")
    
    # Try to compile model for faster execution (PyTorch 2.0+)
    try:
        if hasattr(torch, 'compile'):
            model = torch.compile(model, mode='reduce-overhead')
            print("Model compiled for faster execution!")
    except:
        pass
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Early stopping for quick training (more aggressive)
    best_acc = 0.0
    patience = 1  # Stop after 1 epoch without improvement
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for images, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}", leave=False):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            
            # Use mixed precision if available
            if use_amp and scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            running += loss.item()
        print(f"Epoch {epoch+1}/{args.epochs} loss: {running/len(trainloader):.4f}")
        
        # Evaluate only at the end for maximum speed (skip during training)
        if (epoch + 1) == args.epochs:
            model.eval()
            correct = 0; total = 0
            with torch.no_grad():
                for images, labels in testloader:
                    images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    outputs = model(images)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            acc = 100.*correct/total
            print(f"Test accuracy after epoch {epoch+1}: {acc:.2f}%")
            
            # Early stopping (more aggressive in quick mode)
            if acc > best_acc:
                best_acc = acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience and args.quick_mode:
                    print(f"Early stopping at epoch {epoch+1} (no improvement)")
                    break
                # Also stop early if we have decent accuracy in quick mode
                if args.quick_mode and acc > 40.0 and epoch >= 1:
                    print(f"Early stopping at epoch {epoch+1} (good enough accuracy: {acc:.2f}%)")
                    break
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'baseline.pth'))
    print('Saved model to', os.path.join(args.save_dir, 'baseline.pth'))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=2, help='Number of training epochs (reduced for faster training)')
    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size (larger = faster but more memory)')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate for optimizer')
    parser.add_argument('--save-dir', type=str, default='saved')
    parser.add_argument('--quick-mode', action='store_true', help='Use subset of data for faster training')
    args = parser.parse_args()
    train(args)
