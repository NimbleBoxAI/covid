import torch 
import torch.nn as nn
from PIL import Image
import streamlit as st
from torchvision import transforms
from efficientnet_pytorch import EfficientNet

labels = ['Covid', 'Normal', 'Pneumonia']
img_mean, img_std = [0.459], [0.347]
image_size = (456, 456)

class EffNet(nn.Module):
    def __init__(self, img_size):
        super(EffNet, self).__init__()
        self.eff_net = EfficientNet.from_name('efficientnet-b5', in_channels=1,
            image_size = img_size, num_classes=3)
        self.eff_net.set_swish(memory_efficient=False)
    def forward(self, x):
        x = self.eff_net(x)
        x = torch.nn.functional.softmax(x, dim=1)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = nn.DataParallel(EffNet(image_size))
model.load_state_dict(torch.load("./models/efnet-b5-best.pth",
    map_location=device))
model = model.module
model = model.to(device)
model.eval()

tfms = transforms.Compose([transforms.Resize(image_size),
                           transforms.Grayscale(num_output_channels=1),
                           transforms.ToTensor(),
                           transforms.Normalize(img_mean, img_std)])

def process(img, tfms, model):
  img = tfms(img)
  img = torch.unsqueeze(img, 0).to(device)
  return model(img)

st.title("Covid Identifier")
st.write("""Upload your CT scans for prediction you can also upload multiple
            CT scans to predict multiple results or combine them to improve
            results.""")
st.write("Use the below checkbox for that selection before uploading images")

combine = st.checkbox("Combine images for the result")
img_list = st.file_uploader("Upload files here", accept_multiple_files=True)

if len(img_list) != 0:
  res = 0
  bar = st.progress(0)
  for prog, st_img in enumerate(img_list):
    img = Image.open(st_img)
    if combine:
      res += process(img, tfms, model)
      bar.progress(int(prog * 100/len(img_list)) + int(100/len(img_list)))
    else:
      res += process(img, tfms, model)
      bar.progress(int(prog * 100/len(img_list)) + int(100/len(img_list)))
      st.text(st_img.name + ": " + labels[torch.argmax(res)])
  if combine:
    res /= len(img_list) 
    st.text("Predicted Class: " + labels[torch.argmax(res)])
else:
  st.text("Please Upload an image")
