import streamlit as st 
import torch
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image
import hashlib
from pylab import rcParams
from matplotlib import pyplot as plt
import numpy as np
import os
from pathlib import Path
import re
import pickle
import entohi
import caption
st.set_option('deprecation.showfileUploaderEncoding', False)

html_temp = """
   <div class="" style="background-color:orange;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
   <center><p style="font-size:30px;color:white;margin-top:10px;">Final Year Deep Learning Project</p></center> 
   </div>
   </div>
   </div>
   """
st.markdown(html_temp,unsafe_allow_html=True)
  
st.title("""
        Image Captioning and Persons Counter Application
         """
         )

file = st.camera_input("Take a picture") or st.file_uploader("Please upload image for counting number of persons or Caption a Image", type=("jpg", "png"))

#file= st.file_uploader("Please upload image for counting number of persons or Caption a Image", type=("jpg", "png"))


from  PIL import Image, ImageOps

rcParams['figure.figsize'] = 5, 10

if file is None:
  st.text("Please upload an Image file")
else:
  image=Image.open(file)
  image.save('test.jpg')
  print(file)
  print(file.name)
  print(image)
  #image=np.array(image)
  #file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
  #image = cv2.imdecode(file_bytes, 1)
  st.image(image,caption='Uploaded Image.', use_column_width=True)
def detect_objects(image):
        model = fasterrcnn_resnet50_fpn(pretrained=True)
        model.eval()

        transform = T.Compose([T.ToTensor()])
        image = transform(image)
        image = image.unsqueeze(0)

        with torch.no_grad():
            predictions = model(image)

        return predictions

    # Count persons
def count_persons(predictions, threshold=0.75):
    labels = predictions[0]['labels']
    scores = predictions[0]['scores']

    num_people = sum(1 for label, score in zip(labels, scores) if label == 1 and score > threshold)
    return num_people



if st.button("COUNT The Persons"):
  #a=counter.count('test.jpg', visualize=True)
  predictions = detect_objects(Image.open(file))
  num_people = count_persons(predictions)
  a = count_persons(predictions)
  if a == 1:

    print('{} person detected in image.'.format(a))
    text=str('{} person detected in image.'.format(a))
    st.markdown(f"## Output text in English:")
    st.write(text)
    st.markdown(f"## Your audio in English:")
    enaudio = entohi.englishspeech(text)
    print(enaudio)
    audio_file = open(f"{enaudio}","rb")
    audio_bytes = audio_file.read()
    st.audio(audio_bytes,format="audio/mp3/wav", start_time=0)
    st.markdown(f"## Output text in Hindi")
    hitext=entohi.convert(text)
    print(hitext)
    st.write('{}'.format(hitext))
    hiaudio=entohi.hindispeech(hitext)
    print(hiaudio)
    audio_file = open(f"{hiaudio}", "rb")
    audio_bytes = audio_file.read()
    st.markdown(f"## Your audio in Hindi:")
    st.audio(audio_bytes, format="audio/mp3/wav", start_time=0)
    
   
  else:

    print('{} people detected in image.'.format(a))
    text=str('{} people detected in image.'.format(a))
    st.markdown(f"## Output text in English:")
    st.write(text)
    st.markdown(f"## Your audio in English:")
    enaudio = entohi.englishspeech(text)
    print(enaudio)
    audio_file = open(f"{enaudio}","rb")
    audio_bytes = audio_file.read()
    st.audio(audio_bytes,format="audio/mp3/wav", start_time=0)
    st.markdown(f"## Output text in Hindi")
    hitext=entohi.convert(text)
    print(hitext)
    st.write('{}'.format(hitext))
    hiaudio=entohi.hindispeech(hitext)
    print(hiaudio)
    audio_file = open(f"{hiaudio}", "rb")
    audio_bytes = audio_file.read()
    st.markdown(f"## Your audio in Hindi:")
    st.audio(audio_bytes, format="audio/mp3/wav", start_time=0)
if st.button("Caption the Image"):
  s=caption.generate_captions('test.jpg')
  print(s)
  print(type(s))
  text=s
  
  st.markdown(f"## Caption for the Image")
  st.write(text)
  
  st.markdown(f"## Output text in English:")
  st.write(text)
  st.markdown(f"## Your audio in English:")
  enaudio = entohi.englishspeech(text)
  print(enaudio)
  audio_file = open(f"{enaudio}","rb")
  audio_bytes = audio_file.read()
  st.audio(audio_bytes,format="audio/mp3/wav", start_time=0)
  st.markdown(f"## Output text in Hindi")
  hitext=entohi.convert(text)
  print(hitext)
  st.write('{}'.format(hitext))
  hiaudio=entohi.hindispeech(hitext)
  print(hiaudio)
  audio_file = open(f"{hiaudio}", "rb")
  audio_bytes = audio_file.read()
  st.markdown(f"## Your audio in Hindi:")
  st.audio(audio_bytes, format="audio/mp3/wav", start_time=0)
  
html_temp = """
   <div class="" style="background-color:purple;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:20px;color:white;margin-top:10px;">Final Year Project</p></center> 
   </div>
   </div>
   </div>
   """
st.markdown(html_temp,unsafe_allow_html=True)
