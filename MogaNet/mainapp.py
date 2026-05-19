import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import transforms
from SpecAPP import TIFImageFolder
from tqdm import tqdm
from modelapp import moganet_base
from torch.utils.data import DataLoader
from log import get_logger
from torch.optim import lr_scheduler
import math
import random
from flops import count_flops_and_params

# ===================== 随机种子函数 =====================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ===================== 数据整理 =====================
def collate_fn(batch):
    images, labels = zip(*batch)
    transformed_images = []
    for image in images:
        transformed_images.append(transforms.Resize((200, 200))(image))
    transformed_images = torch.stack(transformed_images)
    labels = [label[0] for label in labels]
    labels = torch.tensor(labels)
    return transformed_images, labels

# ===================== 指标计算 =====================
def calculate_metrics(predictions, targets):
    report = classification_report(
        targets, predictions,
        output_dict=True,
        zero_division=1
    )
    acc = np.mean(predictions == targets)
    precision = report['macro avg']['precision']
    recall = report['macro avg']['recall']
    f1 = report['macro avg']['f1-score']
    return acc, precision, recall, f1, report

# ===================== 单次训练 =====================
def run_once(run_id, args):
    seed = 42 + run_id
    set_seed(seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Run {run_id} using {device} device.")

    os.makedirs("./weights", exist_ok=True)
    logger = get_logger(f'/mnt/data/home/wangjialuo/code/moganet/appmoga_run{run_id}.log')

    # 数据增强
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    }

    # image_path = "/home/wangjialuo/TIFdata/data"
    # assert os.path.exists(image_path), f"{image_path} path does not exist."
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
    print(f'Using {nw} dataloader workers per process')

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=nw, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=nw, collate_fn=collate_fn
    )

    # ===================== 模型 =====================
    net = moganet_base()
    flops, params = count_flops_and_params(
        net, img_size=200, in_channels=4, device="cuda"
    )
    print(f"Params: {params / 1e6:.2f} M")
    print(f"FLOPs:  {flops / 1e9:.2f} G")

    in_channel = net.head.in_features
    net.head = nn.Linear(in_channel, args.num_classes)
    net.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        net.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=5e-5
    )

    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    best_val_f1 = -1.0
    best_metrics = None
    best_matrix = None

    train_losses, val_losses = [], []
    save_path = f'./weights/appmoga_run{run_id}.pth'

    # ===================== 训练 =====================
    for epoch in range(args.epochs):
        net.train()
        running_loss = 0.0
        train_preds, train_targets = [], []

        for imgs, labels in tqdm(train_loader, file=sys.stdout):
            optimizer.zero_grad()
            outputs = net(imgs.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_preds.extend(outputs.argmax(1).cpu().numpy())
            train_targets.extend(labels.numpy())

        train_preds = np.array(train_preds)
        train_targets = np.array(train_targets)
        train_acc, train_p, train_r, train_f1, _ = calculate_metrics(train_preds, train_targets)
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        scheduler.step()

        # ---------- 验证 ----------
        net.eval()
        val_preds, val_targets = [], []
        val_loss_sum = 0.0

        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, file=sys.stdout):
                outputs = net(imgs.to(device))
                loss = loss_function(outputs, labels.to(device))
                val_loss_sum += loss.item()

                val_preds.extend(outputs.argmax(1).cpu().numpy())
                val_targets.extend(labels.numpy())

        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        val_acc, val_p, val_r, val_f1, _ = calculate_metrics(val_preds, val_targets)
        val_loss = val_loss_sum / len(val_loader)
        val_losses.append(val_loss)

        logger.info(
            f'[Run {run_id} | Epoch {epoch+1}] '
            f'Train Loss {train_loss:.3f}, Acc {train_acc:.3f}, '
            f'Precision {train_p:.3f}, Recall {train_r:.3f}, F1 {train_f1:.3f}'
        )
        logger.info(
            f'[Run {run_id} | Epoch {epoch+1}] '
            f'Val Loss {val_loss:.3f}, Acc {val_acc:.3f}, '
            f'Precision {val_p:.3f}, Recall {val_r:.3f}, F1 {val_f1:.3f}'
        )

        # ===================== ⭐ 仅改这里：以 Val F1 为最优 =====================
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_metrics = (
                train_acc, train_p, train_r, train_f1,
                val_acc, val_p, val_r, val_f1
            )
            best_matrix = confusion_matrix(val_targets, val_preds)
            torch.save(net.state_dict(), save_path)

    # ===================== 绘图（完全保留） =====================
    plt.figure(figsize=(8, 6))
    plt.imshow(best_matrix, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.title(f'Confusion Matrix (Run {run_id})')
    plt.colorbar()
    tick_marks = np.arange(len(val_dataset.classes))
    plt.xticks(tick_marks, val_dataset.classes, rotation=45)
    plt.yticks(tick_marks, val_dataset.classes)
    thresh = best_matrix.max() / 2.
    for i in range(len(val_dataset.classes)):
        for j in range(len(val_dataset.classes)):
            plt.text(j, i, format(best_matrix[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if best_matrix[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(f'appmoga_cm_run{run_id}.png')
    plt.close()

    plt.figure()
    plt.plot(range(1, args.epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, args.epochs + 1), val_losses, label='Validation Loss')
    plt.legend()
    plt.savefig(f'appmoga_loss_run{run_id}.png')
    plt.close()

    return best_metrics

# ===================== 主函数 =====================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=7)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.0001)
    parser.add_argument('--device', default='cuda:2')
    args = parser.parse_args()

    results = []
    for run_id in range(3):
        results.append(run_once(run_id, args))

    results = np.array(results)
    mean_metrics = results.mean(axis=0)

    final_msg = (
        "\n========== FINAL AVERAGE (3 RUNS, BEST Val-F1) ==========\n"
        f"Train | Acc {mean_metrics[0]:.4f}, P {mean_metrics[1]:.4f}, "
        f"R {mean_metrics[2]:.4f}, F1 {mean_metrics[3]:.4f}\n"
        f"Val   | Acc {mean_metrics[4]:.4f}, P {mean_metrics[5]:.4f}, "
        f"R {mean_metrics[6]:.4f}, F1 {mean_metrics[7]:.4f}\n"
    )

    print(final_msg)

    # ✅ 写入第三轮 log
    final_logger = get_logger(
        '/mnt/data/home/wangjialuo/code/moganet/appmoga_run3.log'
    )
    final_logger.info(final_msg)
