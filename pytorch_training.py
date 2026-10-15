import ssl

ssl._create_default_https_context = ssl._create_unverified_context
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18

def get_cifar10_resnet18():
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model

def main():
    if not torch.cuda.is_available():
        raise SystemError("CUDA GPU is not available. Please check your drivers/PyTorch build.")
    
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"Device Count: {torch.cuda.device_count()}")

    torch.backends.cudnn.benchmark = True

    # 2. Data Loading & Augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # FIX 1: Set drop_last=True so train batches are always exactly 128
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True
        ,drop_last=True
    )

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True
    )

    # 3. Model Initialization
    base_model = get_cifar10_resnet18().to(device)

    if torch.cuda.device_count() > 1:
        print(f"Utilizing {torch.cuda.device_count()} GPUs with DataParallel!")
        base_model = nn.DataParallel(base_model)

    # Compile with cudagraphs
    try:
        model = torch.compile(base_model, backend="cudagraphs")
        print("Model compiled with torch.compile(backend='cudagraphs')")
    except Exception as e:
        model = base_model
        print(f"torch.compile skipped: {e}")

    # 4. Optimization & AMP Scaler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    scaler = torch.amp.GradScaler('cuda')
    epochs = 200

    # 5. Training Loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in trainloader:
            # FIX 2: Clear pending graph states at step boundaries
            torch.compiler.cudagraph_mark_step_begin()

            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        scheduler.step()

        # 6. Evaluation Loop
        model.eval()
        test_correct = 0
        test_total = 0

        # FIX 3: Evaluate on base_model / model._orig_mod so dynamic test batch sizes
        # (256, and remainder 16) don't clash with the 128-batch CUDA graph
        eval_model = getattr(model, "_orig_mod", model)

        with torch.no_grad():
            for inputs, labels in testloader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.amp.autocast('cuda'):
                    outputs = eval_model(inputs)

                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

        train_acc = 100. * correct / total
        test_acc = 100. * test_correct / test_total

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch [{epoch+1:03d}/{epochs}] | "
                  f"Train Loss: {running_loss/total:.4f} | "
                  f"Train Acc: {train_acc:.2f}% | "
                  f"Test Acc: {test_acc:.2f}%")

if __name__ == '__main__':
    main()