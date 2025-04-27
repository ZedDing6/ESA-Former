import os
import math
import argparse

import numpy as np
from matplotlib import pyplot as plt

from log import get_logger
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from spec_decoder import TIFImageFolder
from vit_model import vit_base_patch16_224_in21k as create_model
from utils import  train_one_epoch, evaluate


def calculate_metrics(predictions, targets):
    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, average='macro', zero_division=0)
    recall = recall_score(targets, predictions, average='macro', zero_division=0)
    f1 = f1_score(targets, predictions, average='macro', zero_division=0)
    return accuracy, precision, recall, f1


def collate_fn(batch):
    images, labels = zip(*batch)
    transformed_images = []
    for image in images:
        transformed_images.append(transforms.Resize((200, 200))(image))
    transformed_images = torch.stack(transformed_images)
    labels = [label[0] for label in labels]
    labels = torch.tensor(labels)
    return transformed_images, labels
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

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()
    logger = get_logger('/mnt/data/home/wangjialuo/code/Vision-Trans/vit.log')

    data_transform = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop((200,200)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose(
            [
                transforms.Resize((200,200)),
                transforms.CenterCrop((200,200)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    data_root = "/home/wangjialuo/"
    image_path = os.path.join(data_root, "TIFdata", "data")
    train_dataset = TIFImageFolder(root=os.path.join(image_path, "train"), transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = TIFImageFolder(root=os.path.join(image_path, "val"), transform=data_transform["val"])

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

    model = create_model(num_classes=args.num_classes, has_logits=False).to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        # 删除不需要的权重
        del_keys = ['head.weight', 'head.bias'] if model.has_logits \
            else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
        for k in del_keys:
            del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head, pre_logits外，其他权重全部冻结
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    best_acc=0
    train_losses = []
    val_losses = []
    best_Matrix = None
    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)
        train_losses.append(train_loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scheduler.step()
        accuracy_train, precision_train, recall_train, f1_train,_,_ = evaluate1(model=model,
                                                    data_loader=train_loader,
                                                    device=device)
        print("[epoch {}] accuracy: {:.3f}, precision: {:.3f}, recall: {:.3f}, F1: {:.3f}".format(epoch, accuracy_train,
                                                                                                  precision_train, recall_train,
                                                                                                  f1_train))
        logger.info("[epoch {}] accuracy: {:.3f}, precision: {:.3f}, recall: {:.3f}, F1: {:.3f}".format(epoch, accuracy_train,
                                                                                                  precision_train,
                                                                                                  recall_train,
                                                                                                  f1_train))
        tags = ["loss", "accuracy", "precision", "recall", "F1", "learning_rate"]
        accuracy_val, precision_val, recall_val, f1_val,predictions, targets = evaluate1(model=model,
                                                                        data_loader=val_loader,
                                                                        device=device)
        print("[epoch {}] Test accuracy: {:.3f}, precision: {:.3f}, recall: {:.3f}, F1: {:.3f}".format(epoch,
                                                                                                       accuracy_val,
                                                                                                       precision_val,
                                                                                                       recall_val,
                                                                                                       f1_val))
        logger.info("[epoch {}] Test accuracy: {:.3f}, precision: {:.3f}, recall: {:.3f}, F1: {:.3f}".format(epoch,
                                                                                                       accuracy_val,
                                                                                                       precision_val,
                                                                                                       recall_val,
                                                                                                       f1_val))
        # validate

        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)
        val_losses.append(val_loss)
        #
        # tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar("loss", train_loss, epoch)
        tb_writer.add_scalar("train_accuracy", accuracy_train, epoch)
        tb_writer.add_scalar("train_precision", precision_train, epoch)
        tb_writer.add_scalar("train_recall", recall_train, epoch)
        tb_writer.add_scalar("train_f1", f1_train, epoch)

        tb_writer.add_scalar("val_accuracy", accuracy_val, epoch)
        tb_writer.add_scalar("val_precision", precision_val, epoch)
        tb_writer.add_scalar("val_recall", recall_val, epoch)
        tb_writer.add_scalar("val_f1", f1_val, epoch)
        tb_writer.add_scalar(tags[0], optimizer.param_groups[0]["lr"], epoch)


        if accuracy_val > best_acc:
            best_acc = accuracy_val
            best_Matrix = confusion_matrix(targets, predictions)
            torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))
    plt.figure(figsize=(8, 6))
    plt.imshow(best_Matrix, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.title('Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(args.num_classes)
    plt.xticks(tick_marks, range(args.num_classes), rotation=45)
    plt.yticks(tick_marks, range(args.num_classes))

    thresh = best_Matrix.max() / 2.
    for i in range(args.num_classes):
        for j in range(args.num_classes):
            plt.text(j, i, format(best_Matrix[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if best_Matrix[i, j] > thresh else "black")

    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.tight_layout()
    plt.savefig('Vit.png')  # 保存混淆矩阵图像
    plt.show()
    plt.figure()
    plt.plot(range(1, args.epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, args.epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(bottom=0, top=max(max(train_losses), max(val_losses)) + 0.5)
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig('Vitloss.png')  # 保存loss图为图片文件
    plt.show()  # 显示loss图

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.0001)

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="/home/wangjialuo/TIFdata/data")
    parser.add_argument('--model-name', default='', help='create model name')

    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
