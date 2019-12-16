from CNN import EfficientNet
from datetime import datetime
from torch.autograd import Variable

import torch

from utils.image_loader import load_data
from utils.config import Config
from utils.model_util import validation
from utils.model_util import draw_plot


def train():
    config = Config().params
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load data and set model for fine-tuning
    train_loader, vad_loader = load_data()
    model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=config["class_num"])
    for name, param in model.named_parameters():
        if '_fc' in name:
            continue
        param.requires_grad = False
    model = model.to(device)

    # set loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    train_accs, vad_accs = [], []

    for epoch in range(config["epoch"]):
        train_loss = 0
        train_acc = 0
        for step, (images, labels) in enumerate(train_loader, 0):
            # set data and predicted
            images = Variable(images).to(device)
            labels = Variable(labels).to(device)
            outputs = model(images)

            # train model by difference
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # calculate accuracy
            outputs = torch.argmax(outputs, dim=1)
            train_acc += (outputs == labels).float().mean()
            train_loss += loss.item()

            # development test and print loss, acc
            if step == len(train_loader)-1:
                model.eval()
                with torch.no_grad():
                    train_loss = train_loss / len(train_loader)
                    train_acc = train_acc / len(train_loader)
                    vad_loss, vad_acc = validation(model, vad_loader, criterion, device)
                print("Epoch: {}/{}..".format(epoch+1, config["epoch"]),
                      "Train Loss: {:.6f}..".format(train_loss),
                      "Train Acc: {:.6f}..".format(train_acc),
                      "Vad Loss: {:.6f}..".format(vad_loss),
                      "Vad Acc: {:.6f}".format(vad_acc))
                train_accs.append(train_acc)
                vad_accs.append(vad_acc)
                model.train()

    # after train save model and acc graph
    now = datetime.now()
    model_acc = sum(vad_accs[-5:]) * 100 / 5
    name = '{:%mM %dD %IH %MM} {:.1f}%'.format(now, model_acc)
    torch.save(model.state_dict(), name + ".tar")
    draw_plot(train_accs, vad_accs, name)


if __name__ == '__main__':
    train()
