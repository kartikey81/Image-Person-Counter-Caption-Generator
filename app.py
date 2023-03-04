import streamlit as st 
import numpy as np
import os
from pathlib import Path
import re
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

import cv2
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

if st.button("Caption the Image"):
  s=caption.generate_captions('test.jpg')
  print(s)
  print(type(s))
  text=s
  
  st.markdown(f"## Caption for the Image")
  st.write(text)
  
 
  
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
