# import the inference-sdk
from inference_sdk import InferenceHTTPClient

# initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    # Train model with RoboFlow to identify bubbles, fishing bar, moving chalk
    api_key="CENSOR_SENSITIVE_INFO"
)


import pyautogui as pg
import time
import io
import numpy as np
from PIL import ImageGrab, Image
from roboflow import Roboflow
import asyncio

screenWidth, screenHeight = pg.size()  # Get the size of the primary monitor

# Set the target size for the captured image (reduce size to improve performance)
target_width = screenWidth // 2
target_height = screenHeight // 2
hook_height = screenHeight // 4


def rugPull():
    rugPull_error = 0
    
    while True:
        # Capture screen
        img = ImageGrab.grab(bbox=(0, 0, screenWidth, screenHeight))
        img = img.resize((target_width, target_height))  # Resize the image to reduce processing time
        np_img = np.array(img)

        # Get predictions from the Roboflow model
        response = CLIENT.infer(img, model_id="CENSOR_SENSITIVE_INFO")

        # Ensure we access the 'predictions' part of the response
        preds = response.get('predictions', [])

        if not preds:
            rugPull_error += 1

        if rugPull_error > 2:
            return 'failed'

        chalk_center = None
        fishing_bar_center = None

        # Iterate through the predictions list
        for prediction in preds:
            detected_class = prediction['class']
            x1, y1, x2, y2 = int(prediction['x'] - prediction['width'] / 2), int(prediction['y'] - prediction['height'] / 2), int(prediction['x'] + prediction['width'] / 2), int(prediction['y'] + prediction['height'] / 2)

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            if detected_class == 'chalk':
                chalk_center = (center_x, center_y)
            elif detected_class == 'fishing_bar':
                fishing_bar_center = (center_x, center_y)

        # Check the condition and perform the click
        if chalk_center and fishing_bar_center and chalk_center[0] < fishing_bar_center[0]:
            pg.click(x=chalk_center[0], y=chalk_center[1])



async def capture_and_predict():
    pg.click(x = target_width, y = hook_height)
    
    while True:
        
        # Capture screen
        img = ImageGrab.grab(bbox=(0, 0, screenWidth, screenHeight))
        img = img.resize((target_width, target_height))  # Resize the image to reduce processing time
        np_img = np.array(img)


        # Get predictions from the Roboflow model
        predictions = CLIENT.infer(img,model_id="CENSOR_SENSITIVE_INFO")

        object_detected = False

        # Check for detections
        for prediction in predictions['predictions']:
            detected_bubble = prediction['class']
            confidence = prediction['confidence']
            if detected_bubble == 'bubble' and confidence > 0.1:
                object_detected = True
                click_x = int((prediction['x'] - prediction['width'] / 2) * 2)
                click_y = int((prediction['y'] - prediction['height'] / 2) * 2)
                pg.click(click_x, click_y)
                break

        if object_detected:
            rugPull()
            if rugPull() == 'failed':
                continue

        await asyncio.sleep(0.01)  # Add a short delay to reduce CPU usage

# Run the asynchronous function

time.sleep(3)
rugPull()



## To be continue
