pip install sklearn

from sklearn.utils import shuffle
from io import BytesIO
import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import pickle
import base64
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow import keras

import keras
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, InputLayer, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pickle


import cv2
import math
import os
from glob import glob
from scipy import stats as s

from moviepy.editor import *   ###VideoFileClip

from streamlit_player import st_player

import io
import os 

Drive = "C:"
## Define root folder
RootFolder = Drive+"/1-GG/CAP4/EventDetection/Dataset/VDO"
modesavedFolder = "C:/1-GG/CAP4/EventDetection/Dataset/SavedModel"

def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Predicted_data', index=False)
    writer.save()
    processed_data = output.getvalue()
    return processed_data

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    val = to_excel(df)
    b64 = base64.b64encode(val)
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="Event_prediction.xlsx">Download csv file</a>'



def VDO_Preprocessing(input_video_file_path):
    image_height  = 224
    image_width   = 224
    #window_size   = 1
    
    base_model = VGG16(weights='imagenet', include_top=False)
    
    # Initialize a Deque Object with a fixed size which will be used to implement moving/rolling average functionality.
    #predicted_labels_probabilities_deque = deque(maxlen = window_size)

    # Reading the Video File using the VideoCapture Object
    video_reader = cv2.VideoCapture(input_video_file_path)

    # Getting the width and height of the video 
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Writing the Overlayed Video Files Using the VideoWriter Object
    #video_writer = cv2.VideoWriter(output_video_file_path, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), 24, (original_video_width, original_video_height))


    prediction_images = []
    while True: 

        # Reading The Frame
        status, frame = video_reader.read() 

        if not status:
            break

        #print("we are here")
        # Resize the Frame to fixed Dimensions
        resized_frame = cv2.resize(frame, (image_height, image_width))
        resized_frame = image.img_to_array(resized_frame)
        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1
        normalized_frame = resized_frame / 255

        
        
        # converting the list to numpy array
        # appending the image to the image list
        prediction_images.append(normalized_frame)
    # converting the list to numpy array
    prediction_images = np.array(prediction_images)
        
    # shape of the array
    prediction_images.shape
    # extracting features for validation frames
    prediction_images = base_model.predict(prediction_images)
    #prediction_images.shape
    
    # converting features in one dimensional array
    prediction_images = prediction_images.reshape(prediction_images.shape[0], 7*7*512)

    max = prediction_images.max()
    prediction_images = prediction_images/max
    
    return prediction_images

def predict_on_live_video(video_file_path,model_class,prediction,output_file_path):

    # Initialize a Deque Object with a fixed size which will be used to implement moving/rolling average functionality.
    #predicted_labels_probabilities_deque = deque(maxlen = window_size)
    
    frame_no = -1

    # Reading the Video File using the VideoCapture Object
    video_reader = cv2.VideoCapture(video_file_path)

    # Getting the width and height of the video 
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Writing the Overlayed Video Files Using the VideoWriter Object
    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), 24, (original_video_width, original_video_height))

    while True: 

        # Reading The Frame
        status, frame = video_reader.read() 

        if not status:
            break

        frame_no = frame_no+1
        
        # some frames with black - skip those
        ## images in openCV (or in your case frames) are represented as a numpy array, 
        ## they can be averaged for low values (which represent black frames).
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  ## Convert to grey image (black and white)
        if np.average(gray) < 20:  ## if it dark screen , skip 
        # skips an iteration, so the frame isn't saved
          continue
        

           
        # Accessing The Class Name using prediction list.
        predicted_class_name = model_class[prediction[frame_no]]
        #print(predicted_class_name)
        
        # Overlaying Class Name Text Ontop of the Frame
        cv2.putText(frame, predicted_class_name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        #cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        #cv2.putText(frame, avg_prob, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        #cv2.putText(frame, 
                    #avg_prob, 
                    #(10, 100),   # bottomLeftCornerOfText
                    #cv2.FONT_HERSHEY_SIMPLEX, 
                    #1, 
                    #(0, 0, 255), 
                    #2)
                       
        # Writing The Frame
        video_writer.write(frame)
    
    # Closing the VideoCapture and VideoWriter objects and releasing all resources held by them. 
    video_reader.release()
    video_writer.release()

def streamlit_interface():
   """
      Function for Streamlit Interface
   """
   st.markdown('<h1 style="background-color:lightblue; text-align:center; font-family:arial;color:white">CAPS Assignment - GROUP-99 </h1>', unsafe_allow_html=True)
   st.markdown('<h2 style="background-color:MediumSeaGreen; text-align:center; font-family:arial;color:white">VDO Analytics &  Event PREDICTION</h2>', unsafe_allow_html=True)
   
   # Sidebars (Left)
   st.sidebar.header("VDO Analytics & Event Prediction")
   st.image('img1.png', width=600)

   # Sidebar -  Upload File for Batch Prediction
   st.sidebar.subheader("Get Batch Prediction")
   uploaded_file        = st.sidebar.file_uploader("Upload Your VDO File", type='mp4', key=None)
   usr_sidebar_model    = st.sidebar.radio('Choose classifiers Model', ('Neural Network Sequential Model', 'CNN_model'))
   
    ## select model
   if usr_sidebar_model == 'Neural Network Sequential Model':
        model_name = 'video classification model_jj.h5'
        model = keras.models.load_model(RootFolder+'/'+model_name)
    
        #modelload = pickle.load(open('C:/GG-16-03/CAP4/CrimeAnalytics/finaldataset/Crime_RF_Model_pickle', 'rb'))
        
   elif usr_sidebar_model == 'CNN_model':
        model_name = 'CNN_modelBP.h5'
        model = keras.models.load_model(modesavedFolder+'/'+model_name)
        
   else:
        print('Choose a classifier model')
     
   
   
   
   
   
   if st.sidebar.button('Submit Batch'):
      if uploaded_file is not None:
         input_video_file_path = uploaded_file
         #batch_data = pd.read_csv(uploaded_file)
         st.write("Input Vdo File" , input_video_file_path )
         
         input_video_file_path = 'C:/1-GG/CAP4/EventDetection/Dataset/combined_video-2vdo.mp4' ## temporary solution 
         st.sidebar.text('Start of Pre-processing')
         preProcessedData = VDO_Preprocessing(input_video_file_path)
         st.write("Shape of the feature set" , preProcessedData.shape )
         st.sidebar.text('End of Pre-processing')
         
         
         st.sidebar.text('Event Prediction ......')
         ## Event Prediction
         #predict = []        
         # predicting class for each array
         prediction = model.predict_classes(preProcessedData)
         #print(prediction)
         model_class = ['Basketball', 'SoccerPenalty'] 
         output_file_path = 'C:/1-GG/CAP4/EventDetection/Dataset/combined_video-2vdo-StreamlitOutput.mp4'
         
         predict_on_live_video(input_video_file_path,model_class,prediction,output_file_path)
         
         
         st.sidebar.text('Prediction Created Sucessfully!')
         #st.header("Sample Output")
         #st.write(batch_pred_df.head(10)) 
         
         #st.sidebar.text(shuffle(batch_pred_df.head()))
         #st.sidebar.header("Download Complete File")
         #st.sidebar.markdown(get_table_download_link(batch_pred_df), unsafe_allow_html=True)
   
 
if __name__ == '__main__':
    streamlit_interface()
