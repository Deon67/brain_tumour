from keras.models import load_model
import streamlit as st
import numpy as np
from PIL import Image
from keras.preprocessing import image
import cv2

model=load_model('brain1.h5')
st.write('# Brain tumour Detection')
"""
* A cancerous or non-cancerous mass or growth of abnormal cells in the brain.
* Tumours can start in the brain, or cancer elsewhere in the body can spread to the brain.
* Symptoms include new or increasingly strong headaches, blurred vision, loss of balance, confusion and seizures. In some cases, there may be no symptoms.
* Treatments include surgery, radiation and chemotherapy.
* It is very commonCommon
* More than 1 million cases per year (India)
* Treatable by a medical professional
* Requires a medical diagnosis
* Lab tests or imaging always required 
"""
"""
This App will be able to detect whether a person has brain tumour or not.
"""


img=st.file_uploader('The Allowed Formats are',type=['jpeg','jpg','png'])
if img is None:
    st.write('Please Select An Image')
else:
    img2=Image.open(img)
    st.image(img2,use_column_width=True)
    img3=img2.resize((64,64))
    test_img=image.img_to_array(img3)
    try:
        test_img=cv2.cvtColor(test_img,cv2.COLOR_BGR2RGB)
    except:
        st.write('# Please Enter a Better Image')
    else:
        test_img=np.expand_dims(test_img,axis=0)
        result=model.predict(test_img)
        if np.round(result[0][0])==1:
            st.write('# No,this Person is fine')
        else:
            st.write('# Yes,This Person Suffers from brain tumour')
