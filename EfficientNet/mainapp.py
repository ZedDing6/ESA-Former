import os
import math
import argparse
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import confusion_matrix
from omodel import efficientnet_b0 as create_model
from spec_decoder import TIFImageFolder
from utils import train_one_epoch, validate_one_epoch, evaluate_full, set_seed, collate_fn
from log import get_logger
from flops import count_flops_and_params


def run_once(run_id, args):
    set_seed(42 + run_id)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Run {run_id} using {device} device.")

    tb_writer = SummaryWriter(log_dir=f"./runs/run_{run_id}")
    os.makedirs("./weights", exist_ok=True)
    logger = get_logger(f'/mnt/data/home/wangjialuo/code/Efficient/effoapp_run{run_id}.log')

    img_size = {"B0": 200, "B1": 240, "B2": 260, "B3": 300, "B4": 380, "B5": 456, "B6": 528, "B7": 600}
    num_model = "B0"

    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(img_size[num_model]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]),
        "val": transforms.Compose([
            transforms.Resize(img_size[num_model]),
            transforms.CenterCrop(img_size[num_model]),
            transforms.ToTensor(),
        ])
    }

    # data_root = "/home/wangjialuo/"
    # image_path = os.path.join(data_root, "TIFdata", "data")
    data_root = "/mnt/data/home/wangjialuo/dataset"
    image_path = os.path.join(data_root, "appdata", "data")

    train_dataset = TIFImageFolder(
        root=os.path.join(image_path, "train"),
        transform=data_transform["train"]
    )
    val_dataset = TIFImageFolder(
        root=os.path.join(image_path, "val"),
        transform=data_transform["val"]
    )

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=nw,
        pin_memory=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
        collate_fn=collate_fn
    )

    model = create_model(num_classes=args.num_classes).to(device)

    flops, params = count_flops_and_params(
        model,
        img_size=200,
        in_channels=4,
        device="cuda"
    )
    print(f"Params: {params / 1e6:.2f} M")
    print(f"FLOPs:  {flops / 1e9:.2f} G")

    if args.weights != "":
        if os.path.exists(args.weights):
            weights_dict = torch.load(args.weights, map_location=device)
            load_weights_dict = {k: v for k, v in weights_dict.items()
                                 if model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(load_weights_dict, strict=False)
        else:
            raise FileNotFoundError(f"not found weights file: {args.weights}")

    if args.freeze_layers:
        for name, para in model.named_parameters():
            if ("features.top" not in name) and ("classifier" not in name):
                para.requires_grad_(False)
            else:
                print(f"training {name}")

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=1E-4)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    train_losses, val_losses = [], []

    best_val_accuracy = 0.0
    best_val_metrics = None
    best_confusion_matrix = None

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model, optimizer, train_loader, device, epoch
        )
        scheduler.step()

        val_loss, val_acc = validate_one_epoch(
            model, val_loader, device, epoch
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        tb_writer.add_scalar("Train/Loss", train_loss, epoch)
        tb_writer.add_scalar("Train/Acc", train_acc, epoch)
        tb_writer.add_scalar("Val/Loss", val_loss, epoch)
        tb_writer.add_scalar("Val/Acc", val_acc, epoch)

        train_acc_full, train_pre, train_rec, train_f1, _, _ = evaluate_full(
            model, train_loader, device
        )
        val_acc_full, val_pre, val_rec, val_f1, val_preds, val_targets = evaluate_full(
            model, val_loader, device
        )

        logger.info(
            f"[Run {run_id} | Epoch {epoch}] Train Acc: {train_acc_full:.4f}, Precision: {train_pre:.4f}, Recall: {train_rec:.4f}, F1: {train_f1:.4f}")
        logger.info(
            f"[Run {run_id} | Epoch {epoch}] Val Acc: {val_acc_full:.4f}, Precision: {val_pre:.4f}, Recall: {val_rec:.4f}, F1: {val_f1:.4f}")

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_val_metrics = (val_acc_full, val_pre, val_rec, val_f1)
            best_confusion_matrix = confusion_matrix(val_targets, val_preds)
            torch.save(model.state_dict(), f"./weights/effoappmodel_run{run_id}.pth")

    if best_confusion_matrix is not None:
        plt.figure(figsize=(8, 6))
        plt.imshow(best_confusion_matrix, interpolation='nearest', cmap='Blues')
        plt.title(f'Confusion Matrix (Run {run_id})')
        plt.colorbar()

        tick_marks = np.arange(args.num_classes)
        plt.xticks(tick_marks, range(args.num_classes))
        plt.yticks(tick_marks, range(args.num_classes))

        thresh = best_confusion_matrix.max() / 2.
        for i in range(args.num_classes):
            for j in range(args.num_classes):
                plt.text(j, i, format(best_confusion_matrix[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if best_confusion_matrix[i, j] > thresh else "black")

        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.tight_layout()
        plt.savefig(f'effoapp_cm_run{run_id}.png')
        plt.close()

    plt.figure()
    plt.plot(range(1, args.epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, args.epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'effoapp_loss_run{run_id}.png')
    plt.close()

    tb_writer.close()

    if best_val_metrics is not None:
        return best_val_metrics
    else:
        return (0, 0, 0, 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=7)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.0001)
    parser.add_argument('--weights', type=str, default='')
    parser.add_argument('--freeze_layers', action='store_true', default=False)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    results = []
    for run_id in range(3):
        metrics = run_once(run_id, args)
        results.append(metrics)

    results = np.array(results)
    mean_metrics = results.mean(axis=0)
    std_metrics = results.std(axis=0)

    print(f"\nFinal Average Results (3 runs):")
    print(f"Accuracy : {mean_metrics[0]:.4f} ± {std_metrics[0]:.4f}")
    print(f"Precision: {mean_metrics[1]:.4f} ± {std_metrics[1]:.4f}")
    print(f"Recall   : {mean_metrics[2]:.4f} ± {std_metrics[2]:.4f}")
    print(f"F1-score : {mean_metrics[3]:.4f} ± {std_metrics[3]:.4f}")