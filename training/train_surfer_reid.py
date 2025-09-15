import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from reid_backbones import ReIDExtractor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a ReID backbone on a surfers dataset"
    )
    parser.add_argument("--data", required=True, help="Path to dataset root containing train/ folder")
    parser.add_argument("--backbone", default="osnet", help="Backbone name from reid_backbones")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto", help="Device for training")
    return parser.parse_args()


def main():
    args = parse_args()

    extractor = ReIDExtractor(backend=args.backbone, device=args.device)
    model = extractor.model
    if model is None:
        raise RuntimeError("Failed to load backbone")

    device = extractor.device
    model.train()

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize(extractor.input_size[::-1]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    train_root = os.path.join(args.data, "train")
    dataset = datasets.ImageFolder(train_root, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # simple classification head for training
    with torch.no_grad():
        dummy = torch.zeros(
            1, 3, extractor.input_size[0], extractor.input_size[1], device=device
        )
        feat_dim = model(dummy).shape[1]
    classifier = torch.nn.Linear(feat_dim, len(dataset.classes)).to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(classifier.parameters()), lr=args.lr
    )
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            feats = model(imgs)
            logits = classifier(feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

    weights_path = os.getenv("REID_WEIGHTS", "reid_surfer.pth")
    torch.save(model.state_dict(), weights_path)
    print(f"Saved weights to {weights_path}")


if __name__ == "__main__":
    main()
