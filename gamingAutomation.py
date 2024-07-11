import pyautogui as pg
import time
import io
import numpy as np
from PIL import ImageGrab, Image
from roboflow import Roboflow
import asyncio

screenWidth, screenHeight = pg.size()  # Get the size of the primary monitor


## Previously training models on Roboflow for object detection -> detect bubbles
# Initialize Roboflow model
rf = Roboflow(api_key="YOUR OWN API KEY")
project = rf.workspace().project("PROJECT NAME")
model = project.version("VERSION").model

# Set the target size for the captured image (reduce size to improve performance)
target_width = screenWidth // 2
target_height = screenHeight // 2

async def capture_and_predict():
    while True:
        # Capture screen
        img = ImageGrab.grab(bbox=(0, 0, screenWidth, screenHeight))
        img = img.resize((target_width, target_height))  # Resize the image to reduce processing time
        np_img = np.array(img)


        # Get predictions from the Roboflow model
        predictions = model.predict(np_img).json()

        object_detected = False

        # Check for detections
        for prediction in predictions['predictions']:
            confidence = prediction['confidence']
            if confidence > 0.1:
                object_detected = True
                click_x = int((prediction['x'] - prediction['width'] / 2) * 2)
                click_y = int((prediction['y'] - prediction['height'] / 2) * 2)
                pg.click(click_x, click_y)
                break

        if object_detected:
            break

        await asyncio.sleep(0.01)  # Add a short delay to reduce CPU usage

# Run the asynchronous function
start_time = time.time()
asyncio.run(capture_and_predict())
end_time = time.time()
elapsed = end_time - start_time
print(elapsed)


## To be continue
