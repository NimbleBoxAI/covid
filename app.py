import torch 
import streamlit as st
from PIL import Image
from torchvision import transforms
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

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
labels = ['Covid', 'Normal', 'Pneumonia']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EffNet(image_size)
model = torch.load("./models/efnet-b5-last.pth", map_location=device).module
model.eval()

tfms = transforms.Compose([transforms.Resize(image_size),
                           transforms.Grayscale(num_output_channels=1),
                           transforms.ToTensor(),
                           transforms.Normalize(img_mean, img_std)])

def inference(model, tfms, img):
  img = tfms(img)
  img = torch.unsqueeze(img, 0)
  return model(img)

st.title("Covid Identifier")

img_list = st.file_uploader("""Upload your CT scans for prediction
                    You can also uplaod multiple CT scans to improve end results""",
                  accept_multiple_files=True)

if len(img_list) != 0:
  res = 0
  bar = st.progress(0)
  for prog, st_img in enumerate(img_list):
    img = Image.open(st_img)
    res += inference(model, tfms, img.convert('RGB'))
    bar.progress(int(prog * 100/len(img_list)) + int(100/len(img_list)))
   
  res = res/len(img_list)
  st.text("Predicted Class: " + labels[torch.argmax(res)])
else:
  st.text("Please Upload an image")
  
