import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from osgeo import gdal
from torchvision import transforms
from spec_decoder import TIFImageFolder
from model import resnet50

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 数据转换预处理
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 批量加载图像进行预测
    root_path = "/home/wangjialuo/TIFdata/data/val/"
    dataset = TIFImageFolder(root=root_path, transform=data_transform)
    #假设要预测的图像在该路径下，名为0_11_0.tif
    target_name = "0_11_0.tif"
    sample_idx = next(i for i, (path, _) in enumerate(dataset.samples)
                      if os.path.basename(path) == target_name)
    img_tensor, label = dataset[sample_idx]
    img = torch.unsqueeze(img_tensor, dim=0)  # 扩展批次维度


    # 读取类索引
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' does not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # 创建模型
    model = resnet50(num_classes=4).to(device)

    # 加载模型权重
    weights_path = "/mnt/data/home/wangjialuo/code/ResNet/resnet.pth"
    assert os.path.exists(weights_path), "file: '{}' does not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # 进行预测
    model.eval()
    with torch.no_grad():
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "Class: {}   Prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())


    # 显示图像
    rgb_tensor = img_tensor[:3]  # 提取RGB通道
    rgb_tensor = rgb_tensor * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) \
                 + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)  # 逆归一化
    display_img = (rgb_tensor.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype('uint8')
    plt.imshow(display_img)
    plt.axis('off')  # 关闭坐标轴

    # 在图像上方显示分类标签
    title_fontsize = 12
    plt.title(print_res, fontsize=title_fontsize, pad=5)  # 设置标题和上边距

    # 添加经纬度信息，适当下移
    text_position_y = 0.5  # 初始文本垂直位置
    line_height = 0.4  # 行高度，控制行间距
    font_props = {'fontsize': title_fontsize, 'ha': 'left', 'va': 'bottom'}

    plt.gca().text(1.04, text_position_y - line_height, f"Latitude: {label[1]}",
                   rotation=90, transform=plt.gca().transAxes, **font_props)  # 纬度

    plt.gca().text(1.04, text_position_y, f"Longitude: {label[2]}",
                   rotation=90, transform=plt.gca().transAxes, **font_props)  # 经度

    # 打印每个类的概率
    for i in range(len(predict)):
        print("Class: {:10}   Prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))

    plt.show()

if __name__ == '__main__':
    main()