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
from spec_decoder import TIFImageFolder
from tqdm import tqdm
from origin_model import resnet50
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

    logger = get_logger('/home/wangjialuo/pyproject/resnet/oresnet.log')

    data_transform = {
        "train": transforms.Compose([
            # transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            # transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    data_root="/home/wangjialuo/"
    # data_root = os.path.abspath(os.path.join(os.getcwd(), "../..")) # get data root path
    image_path = os.path.join(data_root, "TIFdata", "data") # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = TIFImageFolder(root=os.path.join(image_path, "train"),
                                   transform=data_transform["train"])
    train_num = len(train_dataset)

    flower_list = train_dataset.class_to_idx
    cla_dict = {val: key for key, val in flower_list.items()}
    with open('class_indices.json', 'w') as json_file:
        json.dump(cla_dict, json_file, indent=4)

    batch_size = 8
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8]) # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw,collate_fn=collate_fn)

    validate_dataset = TIFImageFolder(root=os.path.join(image_path, "val"), transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False, num_workers=nw,collate_fn=collate_fn)

    print("using {} images for training, {} images for validation.".format(train_num, val_num))
    net = resnet50()


    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 4)
    net.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0005)
    epochs = 300
    best_acc = 0.0
    save_path = './oresnet.pth'
    train_losses = []
    val_losses = []
    best_Matrix=None
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        train_preds, train_targets = [], []

        for train_images, train_labels in tqdm(train_loader, file=sys.stdout):
            optimizer.zero_grad()
            outputs = net(train_images.to(device))
            loss = loss_function(outputs, train_labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            predict_y = torch.max(outputs, dim=1)[1]
            train_preds.extend(predict_y.tolist())
            train_targets.extend(train_labels.tolist())

        train_preds = np.array(train_preds)
        train_targets = np.array(train_targets)
        train_accuracy = np.mean(train_preds == train_targets)

        net.eval()
        val_preds, val_targets = [], []

        with torch.no_grad():
            for val_images, val_labels in tqdm(validate_loader, file=sys.stdout):
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                val_preds.extend(predict_y.tolist())
                val_targets.extend(val_labels.tolist())

        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        val_accuracy = np.mean(val_preds == val_targets)

        train_report = classification_report(train_targets, train_preds, output_dict=True,zero_division=1)
        val_report = classification_report(val_targets, val_preds, output_dict=True,zero_division=1)
        train_loss=running_loss / len(train_loader)
        logger.info('[epoch %d] train_loss: %.3f train_accuracy: %.3f' % (
            epoch + 1, train_loss, train_accuracy))
        train_losses.append(train_loss)
        val_loss=running_loss / len(validate_loader)
        logger.info('[epoch %d] val_loss: %.3f val_accuracy: %.3f' % (
            epoch + 1, running_loss / len(validate_loader), val_accuracy))
        val_losses.append(val_loss)
        logger.info('Training Precision: {:.3f}'.format(train_report['macro avg']['precision']))
        logger.info('Training Recall: {:.3f}'.format(train_report['macro avg']['recall']))
        logger.info('Training F1-score: {:.3f}'.format(train_report['macro avg']['f1-score']))
        logger.info('Training Accuracy: {:.3f}'.format(train_report['accuracy']))
        logger.info('Validation Precision: {:.3f}'.format(val_report['macro avg']['precision']))
        logger.info('Validation Recall: {:.3f}'.format(val_report['macro avg']['recall']))
        logger.info('Validation F1-score: {:.3f}'.format(val_report['macro avg']['f1-score']))
        logger.info('Validation Accuracy: {:.3f}'.format(val_report['accuracy']))

        if val_accuracy > best_acc:
            best_acc = val_accuracy
            torch.save(net.state_dict(), save_path)
            best_Matrix=confusion_matrix(val_targets, val_preds)
        # 可视化混淆矩阵
    plt.figure(figsize=(8, 6))
    plt.imshow(best_Matrix, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.title('Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(len(validate_dataset.classes))
    plt.xticks(tick_marks, validate_dataset.classes, rotation=45)
    plt.yticks(tick_marks, validate_dataset.classes)

    thresh = best_Matrix.max() / 2.
    for i in range(len(validate_dataset.classes)):
        for j in range(len(validate_dataset.classes)):
            plt.text(j, i, format(best_Matrix[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if best_Matrix[i, j] > thresh else "black")

    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.tight_layout()
    plt.savefig('oResNet.png')  # 保存混淆矩阵图像
    plt.show()
    plt.figure()
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(bottom=0, top=max(max(train_losses), max(val_losses)) + 0.5)
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig('oResNet.png')  # 保存loss图为图片文件
    plt.show()  # 显示loss图
    print('Finished Training')

if __name__ == '__main__':
    main()
