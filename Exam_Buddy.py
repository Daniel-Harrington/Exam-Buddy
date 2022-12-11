#  Exam Buddy
#  Displaying a face tracked to a hand with changing emotino based on how open the hand is.
#  Daniel Harrington's spinoff of the google Media Pipeline Hands Model Example
#  12/10/2022

##############{Enviroment Requirements}###############
#
# Google Mediapipe:
#
# pip install mediapipe
# 
# Open Conputer Vision:
#
# pip install opencv-python
#
# Numpy:
#
# pip install numpy
#
# Tensorflow 2.7.0:
#
# pip install tensorflow==2.7.0
#
######################################################

#################{Note}###################
#
# Moving hands out of frame resets the threshold for happy/recalibrates
# This is not perfect code its just a fun result of playing with Mediapipe hands
#
#################{Note}###################

#  import necessary packages
import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

max_area = 1

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue
    y , x, c = image.shape
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)


    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        index_coords = [int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * x),int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * y)]
        middle_finger_coords = [int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * x),int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * y)]
        ring_finger_coords = [int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * x),int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * y)]
        thumb_coords = [int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * x),int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * y)]
        pinky_finger_coords = [int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * x),int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * y)]
        wrist_coords = [int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * x),int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * y)]
        thumb_base = [int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x * x),int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y * y)]
      
    #Happy Hands

      hand_polygon = np.array([thumb_coords,index_coords,middle_finger_coords,ring_finger_coords,pinky_finger_coords,wrist_coords], np.int32)

      square_matrix_poly = np.array([np.subtract(hand_polygon[0],hand_polygon[2]),np.subtract(hand_polygon[-1],hand_polygon[-2])])
      #print(square_matrix_poly)
      
      area = np.linalg.det(square_matrix_poly)
      #print(area)
      if area > max_area:
        max_area = area

      #print("max")
      # print(max_area)

      #Increase green as more happy
      scaled_matrix = np.array([np.add(hand_polygon[0],hand_polygon[-3])//2,np.add(hand_polygon[2],hand_polygon[-1])//2],np.int32)
        
      happyface = np.array([np.add(hand_polygon[0],hand_polygon[1])/2,thumb_base,np.add(hand_polygon[-2],hand_polygon[-1])/1.8],np.int32)
      sadface = np.array([np.add(hand_polygon[0],hand_polygon[1])/2,np.add(thumb_base,middle_finger_coords)//2,np.add(hand_polygon[-2],hand_polygon[-1])/1.8],np.int32)
      if area > max_area//2:
       # print(f'area" {area}, max area:{max_area}')
         cv2.fillPoly(image,[hand_polygon],(0,255,255))
         cv2.polylines(image,[happyface],False,(0,0,0),5)
      else:
        cv2.fillPoly(image,[hand_polygon],(0,0,255))
        cv2.polylines(image,[sadface],False,(0,0,0),5)
        
      
      #print(scaled_matrix)
      cv2.ellipse(image,scaled_matrix[0],(5,5),0,0,360,(0,0,0),-1)
      

      cv2.ellipse(image,scaled_matrix[1],(5,5),0,0,360,(0,0,0),-1)
    else:
      max_area = 1
    # Flip the image horizontally for a selfie-view display. And Resize
      
    cv2.imshow('Exam Buddy', cv2.flip(cv2.resize(image,(x*2,y*2)), 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()