import torch 
import streamlit as st
from PIL import Image
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("./models/efnet-b3-best.pth", map_location=device)
model.eval()

labels = ['Covid', 'Normal', 'Pneumonia']

tfms = transforms.Compose([transforms.Resize((224, 224)),
                           transforms.ToTensor(),
                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

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
  
