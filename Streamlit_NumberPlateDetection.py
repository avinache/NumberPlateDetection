import streamlit as st
import torch
from PIL import Image
import os
import shutil
import pytesseract
import easyocr

model = torch.hub.load(
    '/workspaces/codespaces-blank/yolov5',  # 👈 local repo path (must contain hubconf.py)
    'custom',
    path='/workspaces/codespaces-blank/NumPlateBest.pt',
    source='local'
)

results_dir = "/workspaces/codespaces-blank/output"
crop_dir = "/workspaces/codespaces-blank/cropped"
os.makedirs(crop_dir, exist_ok=True)

# Reset Output Directory
if os.path.exists(results_dir):
    shutil.rmtree(results_dir)

reader = easyocr.Reader(['en'])

#############################################################################################################################################
#############################################################################################################################################
######################################                            ###########################################################################
######################################       Streamlit Test       ###########################################################################
######################################                            ###########################################################################
#############################################################################################################################################
#############################################################################################################################################

st.set_page_config(page_title="Vehicle Number Plate Detection", page_icon="🚗", layout="centered")
SideBarMenu = st.sidebar.radio('Navigation',['Home-Dashboard','Number Plate Detection'])
if SideBarMenu=='Home-Dashboard':
  st.title('🚗 Vehicle Number Plate Detection System')
  st.markdown("### Fast • Accurate • Reliable")
  st.markdown("""
  Manual identification of vehicle plates is **time-consuming, labor-intensive, and prone to errors**.  
  This project automates the process using **computer vision techniques**, improving efficiency and reliability.
  **Applications include:**
  - 🚓 Law Enforcement  
  - 🅿 Parking Management  
  - 🚧 Toll Collection  
  """)
if SideBarMenu=='Number Plate Detection':
  st.title('👉 Upload an image below to test the system!')
  st.title("Upload an Image")
  uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
  if uploaded_file is not None:
    image = Image.open(uploaded_file)
    results = model(image)
    results.save(save_dir=results_dir)
    st.image(results.render()[0], caption="Detected Plates", use_container_width=True) #use_container_width;use_column_width
    for filename in os.listdir(results_dir):
      DetectedImg = os.path.join(results_dir, filename)
      print(f"Image Path : {DetectedImg}")
      boxes = results.xyxy[0].cpu().numpy()
      image = Image.open(DetectedImg)
      for i, box in enumerate(boxes):
         xmin, ymin, xmax, ymax, conf, cls = box
         cropped = image.crop((xmin, ymin, xmax, ymax))
         save_path = os.path.join(crop_dir, f"plate_{i+1}.jpg")
         cropped.save(save_path)
         print(f"Saved cropped plate to {save_path}")
         PlateResult = reader.readtext(save_path)
         detected_texts = []
         for (bbox, text, prob) in PlateResult:
            detected_texts.append(text)
         plate_number = " ".join(detected_texts)
         print(f"Final Plate Text: {plate_number}")
         cols = st.columns(2)
         with cols[0]:
            st.image(cropped, caption=f"Plate {i+1}", use_container_width=True)
         with cols[1]:
            st.markdown(
            f"""
            **Confidence:** {conf:.2f}  
            **Plate Number:** `{plate_number}`
            """)