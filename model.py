import torch, time, copy
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, models, transforms 
import os 
from PIL import Image
from efficientnet_pytorch import EfficientNet
from tqdm import tqdm

image_size = (224,224)
data_transforms={"train":transforms.Compose([transforms.Resize(image_size),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                           ]),
                "val":transforms.Compose([transforms.Resize(image_size),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                           ])
                }
 
data_dir=r"covid"
 
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir,x),data_transforms[x]) for x in ["train","val"]}

dataloaders={}
dataloaders["train"]=torch.utils.data.DataLoader(image_datasets["train"], batch_size=64, shuffle=True, num_workers=4) 
dataloaders["val"]=torch.utils.data.DataLoader(image_datasets["val"], batch_size=64, shuffle=False, num_workers=4) 

dataset_sizes={x: len(image_datasets[x]) for x in ["train","val"]}
 
class_names=image_datasets["train"].classes
print(class_names)
"""
num_classes=len(class_names)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
val={"loss":[],"acc":[]}
train={"loss":[],"acc":[]}
 
def train_model(model, criterion, optimizer , num_epochs=50):
    start_time=time.time()
 
    best_acc= 0.0
 
    for epoch in range(num_epochs):
        print("epoch{}/{}".format(epoch,num_epochs-1))
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
 
                running_loss+=loss.item() * inputs.size(0)
                running_corrects+=  torch.sum(preds==labels.data)
 
            '''if phase == "train":
              scheduler.step()'''
 
            epoch_loss=running_loss/dataset_sizes[phase]
            epoch_acc=running_corrects.double()/dataset_sizes[phase]
 
            if phase == "train":
              train["loss"].append(epoch_loss)
              train["acc"].append(epoch_acc.item())
            else:
              val["loss"].append(epoch_loss)
              val["acc"].append(epoch_acc.item())
 
            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase,epoch_loss,epoch_acc))
 
            if phase == "val" and epoch_acc>best_acc:
                torch.save(model,"./models/efnet-b3-best.pth")
                best_acc=epoch_acc
 
    time_elapsed=time.time()- start_time
    print("training completed in {:.0f}m {:.0f}s".format(time_elapsed//60,time_elapsed%60))
    print("best val accuracy: {:4f}".format(best_acc))
 
    return model
 
model = EfficientNet.from_pretrained('efficientnet-b3')
model._fc = nn.Linear(1536, num_classes)
model.to(device)

criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=1e-3)
#exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
model_ft=train_model(model,criterion,optimizer,num_epochs=15)
 
torch.save(model_ft,"./models/efnet-b3-last.pth")
"""