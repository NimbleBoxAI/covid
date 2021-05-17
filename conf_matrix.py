import torch, time, copy
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, models, transforms 
import os 
from PIL import Image
from efficientnet_pytorch import EfficientNet
from tqdm import tqdm
from sklearn.metrics import f1_score

class EffNet(nn.Module):
    def __init__(self, img_size):
        super(EffNet, self).__init__()
        self.eff_net = EfficientNet.from_name('efficientnet-b5', in_channels=1, image_size = img_size, num_classes=3)
        self.eff_net.set_swish(memory_efficient=False)
    def forward(self, x):
        x = self.eff_net(x)
        x = torch.nn.functional.softmax(x, dim=1)
        return x

image_size = (456, 456)
img_mean, img_std = [0.459], [0.347]
data_transforms={"train":transforms.Compose([transforms.Resize(image_size),
                                             transforms.Grayscale(num_output_channels=1),
                                             transforms.ToTensor(),
                                             transforms.Normalize(img_mean, img_std)
                                           ]),
                "test":transforms.Compose([transforms.Resize(image_size),
                                          transforms.Grayscale(num_output_channels=1),
                                          transforms.ToTensor(),
                                          transforms.Normalize(img_mean, img_std)
                                           ])
                }
 
data_dir=r"../covid-dataset"
 
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir,x),data_transforms[x]) for x in ["train","test"]}

dataloaders={}
dataloaders["train"]=torch.utils.data.DataLoader(image_datasets["train"], batch_size=1, shuffle=True) 
dataloaders["test"]=torch.utils.data.DataLoader(image_datasets["test"], batch_size=40, shuffle=True) 

dataset_sizes={x: len(image_datasets[x]) for x in ["train","test"]}
 
class_names=image_datasets["train"].classes

nb_classes=len(class_names)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_ft = nn.DataParallel(EffNet(image_size))
model_ft.load_state_dict(torch.load("./models/efnet-b5.pth", map_location=device))
model_ft = model_ft.to(device)
model_ft.eval()

confusion_matrix = torch.zeros(nb_classes, nb_classes)
f1 = np.zeros(3)
count = 0
with torch.no_grad():
    for i, (inputs, classes) in enumerate(tqdm(dataloaders['test'])):
        inputs = inputs.to(device)
        classes = classes.to(device)
        outputs = model_ft(inputs)
        _, preds = torch.max(outputs, 1)
        for t, p in zip(classes.view(-1), preds.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1
        classes = classes.cpu()
        preds = preds.cpu()
        mid_f1 = np.array(f1_score(classes, preds, average=None))
        f1 += np.add(f1, mid_f1)
        count += 1


print("Per class Accuracy: ", confusion_matrix.diag()/confusion_matrix.sum(1))
confusion_matrix = confusion_matrix.cpu().detach().numpy()
print(class_names)
print("Confusion matrix: \n", confusion_matrix - confusion_matrix.min() / (confusion_matrix.max() - confusion_matrix.min()))
print("F1 Score: ", f1/count)
