import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from Spec_decoder import TIFImageFolder
from model import swin_tiny_patch4_window7_224 as create_model


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img_size = 200
    data_transform = transforms.Compose(
        [transforms.Resize(int(img_size * 1.14)),
         transforms.CenterCrop(img_size),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # 批量加载图像进行预测
    root_path = "/home/wangjialuo/TIFdata/data/val/"
    dataset = TIFImageFolder(root=root_path, transform=data_transform)
    # 假设要预测的图像在该路径下，名为0_11_0.tif
    target_name = "0_11_0.tif"
    sample_idx = next(i for i, (path, _) in enumerate(dataset.samples)
                      if os.path.basename(path) == target_name)
    img_tensor, label = dataset[sample_idx]
    img = torch.unsqueeze(img_tensor, dim=0)  # 扩展批次维度

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = create_model(num_classes=5).to(device)
    # load model weights
    model_weight_path = "/home/wangjialuo/pyproject/swin/weights/swin.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()


if __name__ == '__main__':
    main()
