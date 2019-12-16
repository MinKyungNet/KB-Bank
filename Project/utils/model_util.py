from torch.autograd import Variable

import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np


# validation test
def validation(model, vad_loader, criterion, device):
    vad_loss = 0
    vad_acc = 0
    for i, (images, labels) in enumerate(vad_loader):
        images = Variable(images).to(device)
        labels = Variable(labels).to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        vad_loss += loss.item()
        outputs = torch.argmax(outputs, dim=1)
        vad_acc += (outputs == labels).float().mean()
    vad_loss = vad_loss / len(vad_loader)
    vad_acc = vad_acc / len(vad_loader)
    return vad_loss, vad_acc


# draw acc graph
def draw_plot(train_accs, vad_accs, file_name):
    plt.plot(train_accs, label='Train Acc')
    plt.plot(vad_accs, label='Vad Acc')
    plt.legend(frameon=False)
    plt.savefig(file_name + '.png')
    plt.show()


# get class activation map
def return_cam(feature_conv, weight_softmax, class_idx):
    size_upsample = (128, 128)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam/np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam
