import torch 
import streamlit as st
from PIL import Image
from torchvision import transforms
from gradcam import GradCam,GuidedBackpropReLUModel,show_cams,show_gbs,preprocess_image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy as np

target_index = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("./models/efnet-b3-best.pth", map_location=device)
model.eval()
grad_cam = GradCam(model=model, blob_name = '_blocks', target_layer_names=['25'], use_cuda=True)


labels = ['Covid', 'Normal', 'Pneumonia']

tfms = transforms.Compose([transforms.Resize((224, 224)),
                           transforms.ToTensor(),
                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def process(tfms, img):
  img = tfms(img)
  img = torch.unsqueeze(img, 0).to(device)
  return img

st.title("Covid Identifier")

img_list = st.file_uploader("""Upload your CT scans for prediction
                    You can also uplaod multiple CT scans to improve end results""",
                  accept_multiple_files=True)

if len(img_list) != 0:
  res = 0
  bar = st.progress(0)
  for prog, st_img in enumerate(img_list):
    img = Image.open(st_img).resize((224,224))
    img.save('img.jpg')
    img_tf = process(tfms, img.convert('RGB'))
    res+= model(img_tf)
    target_index = torch.argmax(res[-1])
    img = np.array(img) / 255
    img_tf = img_tf.requires_grad_(True)
    mask_dic = grad_cam(img_tf, target_index)
    show_cams(img, mask_dic)
    gb_model = GuidedBackpropReLUModel(model=model, activation_layer_name = 'MemoryEfficientSwish', use_cuda=True)
    show_gbs(img_tf, gb_model, target_index, mask_dic)
    bar.progress(int(prog * 100/len(img_list)) + int(100/len(img_list)))
    st.image(img)
    st.image('cam25.jpg')    
  res = res/len(img_list)
  st.text("Predicted Class: " + labels[torch.argmax(res)])
else:
  st.text("Please Upload an image")
  
