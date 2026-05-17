import argparse
import os

import math
import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from log import get_logger
from spec_decoder import TIFImageFolder
from utils import train_one_epoch, evaluate
from model import vit_base_patch16_224_in21k as create_model
from flops import count_flops_and_params

import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_metrics(predictions, targets):
    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, average='macro', zero_division=0)
    recall = recall_score(targets, predictions, average='macro', zero_division=0)
    f1 = f1_score(targets, predictions, average='macro', zero_division=0)
    return accuracy, precision, recall, f1


def collate_fn(batch):
    images,imgs, labels = zip(*batch)
    transformed_images = []
    transformed_imgs = []
    for image in images:
        transformed_images.append(transforms.Resize((200, 200))(image))
    for img in imgs:
        transformed_imgs.append(transforms.Resize((200, 200))(img))
    transformed_images = torch.stack(transformed_images)
    labels = [label[0] for label in labels]
    transformed_imgs = torch.stack(transformed_imgs, dim=0)
    labels = torch.tensor(labels)
    return transformed_images,transformed_imgs, labels
def evaluate1(model, data_loader, device):
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    accuracy, precision, recall, f1 = calculate_metrics(predictions, targets)

    return accuracy, precision, recall, f1, predictions, targets
def run_once(args, run_id):
    seed = 42 + run_id
    set_seed(seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    os.makedirs("./weights", exist_ok=True)

    tb_writer = SummaryWriter(log_dir=f"./runs/run_{run_id}")

    logger = get_logger(
        f'/mnt/data/home/wangjialuo/code/Vision-Trans/ablat/esaformer_run{run_id}.log'
    )

    # ================= 数据处理（不动） =================
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop((200, 200)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize((200, 200)),
            transforms.CenterCrop((200, 200)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    }

    data_root = "/home/wangjialuo/"
    image_path = os.path.join(data_root, "TIFdata", "data")
    # data_root = "/mnt/data/home/wangjialuo/dataset"
    # image_path = os.path.join(data_root, "appdata", "data")
    train_dataset = TIFImageFolder(
        root=os.path.join(image_path, "train"),
        transform=data_transform["train"]
    )
    val_dataset = TIFImageFolder(
        root=os.path.join(image_path, "val"),
        transform=data_transform["val"]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn
    )

    model = create_model(num_classes=args.num_classes, has_logits=False).to(device)
    # print(model)

    flops, params = count_flops_and_params(
        model,
        img_size=200,
        in_channels=5,
        device="cuda"
    )

    print(f"Params: {params / 1e6:.2f} M")
    print(f"FLOPs:  {flops / 1e9:.2f} G")

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=5e-5
    )

    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    best_f1 = 0.0
    best_metrics = None
    best_matrix = None

    train_losses, val_losses = [], []

    # ================= 训练 =================
    for epoch in range(args.epochs):
        train_loss, _ = train_one_epoch(
            model, optimizer, train_loader, device, epoch
        )
        scheduler.step()

        acc_tr, pre_tr, rec_tr, f1_tr, _, _ = evaluate1(
            model, train_loader, device
        )
        acc_va, pre_va, rec_va, f1_va, preds, tars = evaluate1(
            model, val_loader, device
        )

        train_losses.append(train_loss)

        val_loss, _ = evaluate(
            model, val_loader, device, epoch
        )
        val_losses.append(val_loss)

        logger.info(
            f"[Run {run_id} | Epoch {epoch}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {acc_tr:.4f} | Train Precision: {pre_tr:.4f} | "
            f"Train Recall: {rec_tr:.4f} | Train F1: {f1_tr:.4f}"
        )
        logger.info(
            f"[Run {run_id} | Epoch {epoch}] "
            f"Val Loss: {val_loss:.4f} | Val Acc: {acc_va:.4f} | Val Precision: {pre_va:.4f} | "
            f"Val Recall: {rec_va:.4f} | Val F1: {f1_va:.4f}"
        )

        if f1_va > best_f1:
            best_f1 = f1_va
            best_metrics = (acc_va, pre_va, rec_va, f1_va)
            best_matrix = confusion_matrix(tars, preds)

            torch.save(
                model.state_dict(),
                f"./weights/esaformerbest_run{run_id}.pth"
            )

    # ================= 画图 =================
    plt.figure()
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.legend()
    plt.savefig(f"esaformerloss_run{run_id}.png")
    plt.close()

    if best_matrix is not None:
        plt.figure(figsize=(8, 6))
        plt.imshow(best_matrix, interpolation='nearest', cmap='Blues')
        plt.title(f'Confusion Matrix (Run {run_id})')
        plt.colorbar()

        tick_marks = np.arange(args.num_classes)
        plt.xticks(tick_marks, tick_marks, rotation=45)
        plt.yticks(tick_marks, tick_marks)

        thresh = best_matrix.max() / 2.0
        for i in range(best_matrix.shape[0]):
            for j in range(best_matrix.shape[1]):
                plt.text(
                    j, i, int(best_matrix[i, j]),
                    ha="center", va="center",
                    color="white" if best_matrix[i, j] > thresh else "black"
                )

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(f"esaformer_cm_run{run_id}.png")
        plt.close()

    return best_metrics
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.0001)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    results = []

    for run_id in range(3):
        metrics = run_once(args, run_id)
        results.append(metrics)

    results = np.array(results)
    mean_metrics = results.mean(axis=0)

    print(
        f"\nFinal Average Results (3 runs):\n"
        f"Accuracy : {mean_metrics[0]:.4f}\n"
        f"Precision: {mean_metrics[1]:.4f}\n"
        f"Recall   : {mean_metrics[2]:.4f}\n"
        f"F1-score : {mean_metrics[3]:.4f}\n"
    )
