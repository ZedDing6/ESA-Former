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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from idea_model import efficientnet_b0 as create_model
from spec_decoder import TIFImageFolder
from utils import train_one_epoch
from log import get_logger

def collate_fn(batch):
    images, labels = zip(*batch)
    transformed_images = []
    for image in images:
        transformed_images.append(transforms.Resize((200, 200))(image))
    transformed_images = torch.stack(transformed_images)
    labels = [label[0] for label in labels]
    labels = torch.tensor(labels)
    return transformed_images, labels
def calculate_metrics(predictions, targets):
    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, average='macro', zero_division=0)
    recall = recall_score(targets, predictions, average='macro', zero_division=0)
    f1 = f1_score(targets, predictions, average='macro', zero_division=0)
    return accuracy, precision, recall, f1


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

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter()
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")
    img_size = {"B0": 200,
                "B1": 240,
                "B2": 260,
                "B3": 300,
                "B4": 380,
                "B5": 456,
                "B6": 528,
                "B7": 600}
    num_model = "B0"
    data_transform = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size[num_model]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]

        ),
        "val": transforms.Compose(
            [
                transforms.Resize(img_size[num_model]),
                transforms.CenterCrop(img_size[num_model]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
    }
    data_root = "/home/wangjialuo/"
    image_path = os.path.join(data_root, "TIFdata", "data")
    train_dataset = TIFImageFolder(root=os.path.join(image_path, "train"), transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = TIFImageFolder(root=os.path.join(image_path, "val"), transform=data_transform["val"])
    logger = get_logger('/mnt/data/home/wangjialuo/code/Efficient/Efficient.log')
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=collate_fn)

    # 如果存在预训练权重则载入
    model = create_model(num_classes=args.num_classes).to(device)
    if args.weights != "":
        if os.path.exists(args.weights):
            weights_dict = torch.load(args.weights, map_location=device)
            load_weights_dict = {k: v for k, v in weights_dict.items()
                                 if model.model.state_dict()[k].numel() == v.numel()}
            print(model.model.load_state_dict(load_weights_dict, strict=False))
        else:
            raise FileNotFoundError("not found weights file: {}".format(args.weights))

    # 是否冻结权重
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除最后一个卷积层和全连接层外，其他权重全部冻结
            if ("features.top" not in name) and ("classifier" not in name):
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(pg, lr=args.lr, weight_decay=5E-2)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    val_losses = []
    train_losses = []
    best_accuracy=0
    best_confusion_matrix = None
    for epoch in range(args.epochs):
        # train
        train_loss = train_one_epoch(model=model,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch)

        accuracy, precision, recall, f1,_,_= evaluate1(model=model,
                                                   data_loader=train_loader,
                                                   device=device)
        print("[epoch {}] accuracy: {:.3f}, precision: {:.3f}, recall: {:.3f}, F1: {:.3f}".format(epoch, accuracy,precision, recall,f1))
        logger.info("[epoch {}] accuracy: {:.3f}, precision: {:.3f}, recall: {:.3f}, F1: {:.3f}".format(epoch, accuracy,
                                                                                                  precision, recall,
                                                                                                  f1))
        tags = ["loss", "accuracy", "precision", "recall", "F1", "learning_rate"]

        test_accuracy, test_precision, test_recall, test_f1, predictions, targets= evaluate1(model=model,
                                                                        data_loader=val_loader,
                                                                        device=device)
        val_loss=train_one_epoch(model=model,
                                    optimizer=optimizer,
                                    data_loader=val_loader,
                                    device=device,
                                    epoch=epoch)
        val_preds =np.array(predictions)
        val_targets =np.array(targets)

        print("[epoch {}] Test accuracy: {:.3f}, precision: {:.3f}, recall: {:.3f}, F1: {:.3f}".format(epoch,
                                                                                                       test_accuracy,
                                                                                                       test_precision,
                                                                                                       test_recall,
                                                                                                       test_f1))
        logger.info("[epoch {}] Test accuracy: {:.3f}, precision: {:.3f}, recall: {:.3f}, F1: {:.3f}".format(epoch,
                                                                                                       test_accuracy,
                                                                                                       test_precision,
                                                                                                       test_recall,
                                                                                                       test_f1))

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_confusion_matrix = confusion_matrix(val_targets, val_preds)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], accuracy, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)
        tb_writer.add_scalar(tags[3], recall, epoch)
        tb_writer.add_scalar(tags[4], f1, epoch)
        tb_writer.add_scalar(tags[5], optimizer.param_groups[0]["lr"], epoch)
        torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))

    plt.figure(figsize=(8, 6))
    plt.imshow(best_confusion_matrix, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.title('Confusion Matrix')
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
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('Effi.png')
    plt.show()
    #loss
    plt.figure()
    plt.plot(range(1, args.epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, args.epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(bottom=0, top=max(max(train_losses), max(val_losses)) + 0.5)
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig('Effiloss.png')  # 保存loss图为图片文件
    plt.show()  # 显示loss图

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0001)

    parser.add_argument('--data-path', type=str,
                        default="/home/wangjialuo/TIFdata/data")

    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:1', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
