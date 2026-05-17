import os
import sys
import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def set_seed(seed):
    import random
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


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    mean_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    sample_num = 0

    optimizer.zero_grad()
    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = loss_function(outputs, labels)

        pred_classes = torch.max(outputs, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels).sum()
        sample_num += images.shape[0]

        loss.backward()

        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)

        current_acc = accu_num.item() / sample_num
        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(
            epoch, mean_loss.item(), current_acc
        )

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return mean_loss.item(), accu_num.item() / sample_num


@torch.no_grad()
def validate_one_epoch(model, data_loader, device, epoch):
    model.eval()
    loss_function = torch.nn.CrossEntropyLoss()
    mean_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    sample_num = 0

    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = loss_function(outputs, labels)

        pred_classes = torch.max(outputs, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels).sum()
        sample_num += images.shape[0]

        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)

        current_acc = accu_num.item() / sample_num
        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(
            epoch, mean_loss.item(), current_acc
        )

    return mean_loss.item(), accu_num.item() / sample_num


@torch.no_grad()
def evaluate_full(model, data_loader, device):
    model.eval()
    all_predictions = []
    all_targets = []

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_predictions.extend(preds.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())

    accuracy, precision, recall, f1 = calculate_metrics(all_predictions, all_targets)

    return accuracy, precision, recall, f1, all_predictions, all_targets


def collate_fn(batch):
    images, labels = zip(*batch)
    from torchvision import transforms
    transformed_images = [transforms.Resize((200, 200))(image) for image in images]
    transformed_images = torch.stack(transformed_images)
    if isinstance(labels[0], torch.Tensor):
        labels = torch.stack(labels)
    else:
        labels = torch.tensor(labels)
    return transformed_images, labels