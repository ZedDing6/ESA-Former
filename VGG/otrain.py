import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from torchvision import transforms
import torch.optim as optim
from tqdm import tqdm
from spec_decoder import TIFImageFolder
from omodel import vgg
from torch.utils.data import DataLoader
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
def main():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    logger = get_logger('/mnt/data/home/wangjialuo/code/vgg/ovgg.log')

    data_transform = {
        "train": transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }

    data_root = "/home/wangjialuo/"
    image_path = os.path.join(data_root, "TIFdata", "data")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    train_dataset = TIFImageFolder(root=os.path.join(image_path, "train"),
                                   transform=data_transform["train"])
    train_num = len(train_dataset)
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 8
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw,collate_fn=collate_fn)

    validate_dataset = TIFImageFolder(root=os.path.join(image_path, "val"),
                                      transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw,collate_fn=collate_fn)
    print("using {} images for training, {} images for validation.".format(train_num, val_num))

    model_name="vgg16"
    net = vgg(model_name=model_name, num_classes=4, init_weights=True)
    net = net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    epochs = 100
    best_acc = 0.0
    save_path = './{}Net.pth'.format("ovgg")
    train_steps = len(train_loader)
    train_losses = []
    val_losses = []
    best_matix=None

    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        correct = 0
        total = 0

        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels.to(device)).sum().item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

        train_loss = running_loss / train_steps
        train_acc = correct / total

        # validate
        net.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        predicted_labels = []
        true_labels = []

        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                loss = loss_function(outputs, val_labels.to(device))
                val_loss += loss.item()

                _, predicted = outputs.max(1)
                total += val_labels.size(0)
                correct += predicted.eq(val_labels.to(device)).sum().item()

                predicted_labels += predicted.cpu().tolist()
                true_labels += val_labels.tolist()

        predicted_labels = np.eye(4)[predicted_labels]
        true_labels = np.eye(4)[true_labels]
        train_losses.append(train_loss)
        val_loss /= len(validate_loader)
        val_losses.append(val_loss)
        val_acc = correct / total
        val_precision = precision_score(true_labels, predicted_labels, average='macro', zero_division=1)
        val_recall = recall_score(true_labels, predicted_labels, average='macro')
        val_f1 = f1_score(true_labels, predicted_labels, average='macro')
        val_auc = roc_auc_score(true_labels, predicted_labels, multi_class='ovr')

        logger.info(
            '[epoch %d] train_loss: %.3f  train_acc: %.3f  val_loss: %.3f  val_acc: %.3f  val_precision: %.3f  val_recall: %.3f  val_f1: %.3f  val_auc: %.3f' %
            (epoch + 1, train_loss, train_acc, val_loss, val_acc, val_precision, val_recall, val_f1, val_auc))
        print(
            '[epoch %d] train_loss: %.3f  train_acc: %.3f  val_loss: %.3f  val_acc: %.3f  val_precision: %.3f  val_recall: %.3f  val_f1: %.3f  val_auc: %.3f' %
            (epoch + 1, train_loss, train_acc, val_loss, val_acc, val_precision, val_recall, val_f1, val_auc))

        if val_acc > best_acc:
            best_acc = val_acc
            # torch.save(net.state_dict(), save_path)
            torch.save(net, save_path)  # 保存整个模型，包括结构和权重
            best_matix=confusion_matrix(np.argmax(true_labels, axis=-1), np.argmax(predicted_labels, axis=-1))
    plt.figure(figsize=(8, 6))
    plt.imshow(best_matix, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.title('Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(len(validate_dataset.classes))
    plt.xticks(tick_marks, validate_dataset.classes, rotation=45)
    plt.yticks(tick_marks, validate_dataset.classes)

    thresh = best_matix.max() / 2.
    for i in range(len(validate_dataset.classes)):
        for j in range(len(validate_dataset.classes)):
            plt.text(j, i, format(best_matix[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if best_matix[i, j] > thresh else "black")

    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.tight_layout()
    plt.savefig('ovgg.png')  # 保存混淆矩阵图像
    plt.show()
    # 输出并绘制loss图
    plt.figure()
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(bottom=0, top=max(max(train_losses), max(val_losses)) + 0.5)
    plt.legend()

    plt.title('Training and Validation Loss')
    plt.savefig('ovggloss.png')  # 保存loss图为图片文件
    plt.show()  # 显示loss图


if __name__ == '__main__':
    main()