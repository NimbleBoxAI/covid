import torch 
import torch.nn as nn
from PIL import Image
import streamlit as st
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from gradcam import GradCam,GuidedBackpropReLUModel,show_cams,show_gbs,preprocess_image
import numpy as np

labels = ['Covid', 'Normal', 'Pneumonia']
img_mean, img_std = [0.459], [0.347]
image_size = (456, 456)
target_index = 0

device = torch.device("cpu")
model = EfficientNet.from_name('efficientnet-b5', in_channels=1, image_size = image_size, num_classes=3)
model.load_state_dict(torch.load("./models/model.pth", map_location=device))
model = model.to(device)
model.eval()
use_cuda = True if device == "cuda" else False 
grad_cam = GradCam(model=model, blob_name = '_blocks', target_layer_names=['25'], use_cuda=use_cuda)
gb_model = GuidedBackpropReLUModel(model=model, activation_layer_name = 'MemoryEfficientSwish', use_cuda=use_cuda)

tfms = transforms.Compose([transforms.Resize(image_size),
                           transforms.Grayscale(num_output_channels=1),
                           transforms.ToTensor(),
                           transforms.Normalize(img_mean, img_std)])

def process(tfms, img):
  img = tfms(img)
  img = torch.unsqueeze(img, 0).to(device)
  return img

st.title("Covid Detector")
st.write("""Covid detection network trained on axial views of CT scan images.
            Disclaimer : Only use axial views when predicting.""")

st_img = st.file_uploader("Upload files here")
if st_img != None:
  res = 0
  bar = st.progress(0)
  img = Image.open(st_img).resize((456, 456))
  img.save('img.jpg')
  img_tf = process(tfms, img.convert('RGB'))
  res+= model(img_tf)
  target_index = torch.argmax(res[-1])
  img = np.array(img) / 255
  img_tf = img_tf.requires_grad_(True)
  mask_dic = grad_cam(img_tf, target_index)
  show_cams(img, mask_dic)
  show_gbs(img_tf, gb_model, target_index, mask_dic)
  bar.progress(int(prog * 100/len(img_list)) + int(100/len(img_list)))
  st.image(img)
  st.image('cam25.jpg')    
  res = res/len(img_list)
  st.text("Predicted Class: " + labels[torch.argmax(res)])
else:
  st.text("Please Upload an image")