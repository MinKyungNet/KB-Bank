from CNN import EfficientNet
from PIL import Image
from torch.autograd import Variable
from selenium import webdriver

import torch
import torchvision
import cv2
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms

from utils.model_util import return_cam
from utils.config import Config


def test():
    config = Config().params
    classes = ['거북이', '노트북', 'TG-M660G', '드라이어', '마우스', '면도기', '모니터', '백팩', '볼캡', '볼펜', '스마트폰',
               '시계', '안경', '운동화', '자전거', '초밥', '코카콜라', '키보드', '펩시', 'Hp Pavilion 800']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load and transform test image
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    image = Image.open(config["test_image"])
    image_transformed = transform(image)
    image_unsqueezed = image_transformed.unsqueeze(0)
    test_image = Variable(image_unsqueezed).to(device)

    # load model
    model = EfficientNet.from_name('efficientnet-b0')
    model.load_state_dict(torch.load(config["model_path"]))
    model.to(device)
    model.eval()

    # when model work feature_blobs get output of layer
    feature_blobs = []
    def hook_feature(module, input, output):
        feature_blobs.append(output.cpu().data.numpy())
    finalconv_name = '_conv_head'
    model._modules.get(finalconv_name).register_forward_hook(hook_feature)

    # model forward
    output = model(test_image)
    softmax_output = F.softmax(output, dim=1)

    # The three most likely items
    h_x = softmax_output.data.squeeze()
    probs, idx = h_x.sort(0, True)
    for i in range(3):
        print(classes[idx[i]], probs[i].item())
        config["top3"].append([classes[idx[i]], str(probs[i].item()*100)[:5]+"%"])

    # get only weight from last layer(linear)
    params = list(model.parameters())
    weight_softmax = np.squeeze(params[-2].cpu().data.numpy())

    # save CAM image
    image_PIL = transforms.ToPILImage()(image_transformed[0])
    image_PIL.save('result/test.jpg')
    img = cv2.imread('result/test.jpg')
    height, weight, _ = img.shape
    CAMs = return_cam(feature_blobs[0], weight_softmax, [idx[0].item()])
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (weight, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.8 + img * 0.5
    cv2.imwrite('result/CAM.jpg', result)

    # show search result
    path = config["driver_path"]
    browser = webdriver.Chrome(path)
    browser.get('https://www.naver.com')
    search_box = browser.find_element_by_name("query")
    search_box.send_keys(classes[idx[0]])
    search_box.submit()
    shopping_button = browser.find_element_by_class_name("lnb12")
    shopping_button.click()


if __name__ == '__main__':
    test()
