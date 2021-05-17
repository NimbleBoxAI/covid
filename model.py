import torch, time
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, models, transforms 
import os
from PIL import Image
from efficientnet_pytorch import EfficientNet
from tqdm import tqdm
from sklearn.metrics import f1_score

image_size = (456, 456)
img_mean, img_std = [0.459], [0.347]

data_transforms={"train":transforms.Compose([transforms.Resize(image_size),
                                             transforms.Grayscale(num_output_channels=1),
                                             transforms.RandomRotation(degrees=90),
                                             transforms.ToTensor(),
                                             transforms.Normalize(img_mean, img_std)
                                           ]),
                "val":transforms.Compose([transforms.Resize(image_size),
                                          transforms.Grayscale(num_output_channels=1),
                                          transforms.ToTensor(),
                                          transforms.Normalize(img_mean, img_std)
                                           ])
                }
 
data_dir=r"../covid-dataset"
 
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir,x),data_transforms[x]) for x in ["train","val"]}

dataloaders={}
dataloaders["train"]=torch.utils.data.DataLoader(image_datasets["train"], batch_size=40, shuffle=True, num_workers=8) 
dataloaders["val"]=torch.utils.data.DataLoader(image_datasets["val"], batch_size=40, shuffle=True, num_workers=8) 

dataset_sizes={x: len(image_datasets[x]) for x in ["train","val"]}

class_names=image_datasets["train"].classes
print(class_names)

num_classes=len(class_names)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
val={"loss":[],"acc":[]}
train={"loss":[],"acc":[]}
 
def train_model(model, criterion, optimizer , num_epochs=15):
    start_time=time.time()
 
    best_f1 = 0.0
 
    for epoch in range(num_epochs):
        print("epoch{}/{}".format(epoch, num_epochs-1))
        print("-"*10)
 
        for phase in ["train", "val"]:
            if phase =="train":
                model.train()
            else:
                model.eval()
 
            running_loss=0.0
            running_corrects=0.0
 
            for inputs,labels in tqdm(dataloaders[phase]):
                inputs=inputs.to(device)
                labels=labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=="train"):
                    outputs=model(inputs)
                    _,preds=torch.max(outputs,1)
                    loss=criterion(outputs,labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects +=  torch.sum(preds == labels.data)
 
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            labels = labels.cpu()
            preds = preds.cpu()
            epoch_f1 = f1_score(labels, preds, average="micro")

            if phase == "train":
              train["loss"].append(epoch_loss)
              train["acc"].append(epoch_acc.item())
            else:
              val["loss"].append(epoch_loss)
              val["acc"].append(epoch_acc.item())
 
            print("{} Loss: {:.4f} Acc: {:.4f} f1_score: {}".format(phase, epoch_loss, epoch_acc, epoch_f1))

            if phase == "val" and epoch_f1 > best_f1:
                torch.save(model.state_dict(),"./models/efnet-b5.pth")
                best_f1 = epoch_f1
 
    time_elapsed = time.time() - start_time
    print("training completed in {:.0f}m {:.0f}s".format(time_elapsed//60,time_elapsed%60))
    print("best f1 score: {:.4f}".format(best_f1))
 
    return model
 
class EffNet(nn.Module):
    def __init__(self, img_size):
        super(EffNet, self).__init__()
        self.eff_net = EfficientNet.from_name('efficientnet-b5', in_channels=1, image_size = img_size, num_classes=3)
        self.eff_net.set_swish(memory_efficient=False)
    def forward(self, x):
        x = self.eff_net(x)
        x = torch.nn.functional.softmax(x, dim=1)
        return x

model = EffNet(image_size)
model = nn.DataParallel(model)        
model.to(device)

nSamples = [82286, 35996, 25496]
normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
normedWeights = torch.FloatTensor(normedWeights).to(device)

criterion=nn.CrossEntropyLoss(weight=normedWeights)
optimizer=optim.AdamW(model.parameters(),lr=1e-3)
model_ft=train_model(model,criterion,optimizer,num_epochs=25)
 
torch.save(model_ft.save_dict(),"./models/efnet-b5-last.pth")
