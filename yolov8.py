import glob
import os
import sys

#importing ver
try: 
    sys.path.append(glob.glob('C:/Graduation project/CARLA_0.9.10/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name =='nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import random
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
from ultralytics import YOLO
import mediapipe as mp
from scipy.interpolate import RectBivariateSpline
import torch
from ultralytics import YOLO
import json
from pynput.keyboard import Key, Controller
import matplotlib.cm as cm


IM_WIDTH = 640  
IM_HEIGHT = 480


# Load the model
model = YOLO('carla.pt')

# Read the image
#img = cv2.imread('i.jpeg')

# Initialize keyboard controller
keyboard = Controller()




def processImage(image):

    i = np.array(image.raw_data)
    
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    img = i2[:, :, :3]
    img=cv2.resize(img,(640,640))
    results = model.predict(img)
    # Iterate over the results
    for result in results:
        boxes = result.boxes.cpu().numpy()  # Get boxes on CPU in numpy format
    
        for box in boxes: 
             # Iterate over boxes
            
            r = box.xyxy[0].astype(int)  # Get corner points as int
            class_id = int(box.cls[0])  # Get class ID
            score = box.conf.item() 
            class_name = model.names[class_id]  # Get class name using the class ID
       
            print(f"Class: {class_name}, Box: {r},score:{score}")  # Print class name and box coordinates
        
        
            text = f"{class_name}: {score:.2f}" 
            cv2.rectangle(img, (r[0], r[1]), (r[2], r[3]), (0, 255, 0), 2)
            cv2.putText(img, text, (r[0], r[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 1)
    cv2.imshow('RGB',img)
    
    cv2.waitKey(1)




actor_list = []
try:
    # Connect the client and set up bp library and spawn point
    client = carla.Client('localhost', 2000)
    world = client.get_world()  
    blueprint_library = world.get_blueprint_library()
    

    # Add vehicle
    vehicle_bp = world.get_blueprint_library().filter('*mini*') #vehicle mini
    

    spawn_point = random.choice(world.get_map().get_spawn_points())
    
    vehicle = world.spawn_actor(vehicle_bp[0], spawn_point)

    vehicle.apply_control(carla.VehicleControl(throttle=0.4, steer=0.0))
    vehicle.set_autopilot(True)  

    actor_list.append(vehicle)

    # https://carla.readthedocs.io/en/latest/cameras_and_sensors
    # get the blueprint for this sensor
    blueprint = blueprint_library.find('sensor.camera.rgb')
    # change the dimensions of the image
    blueprint.set_attribute('image_size_x', f'{IM_WIDTH}')  
    blueprint.set_attribute('image_size_y', f'{IM_HEIGHT}')
    blueprint.set_attribute('fov', '90')

    # Adjust sensor relative to vehicle
    
    spawn_point = carla.Transform(carla.Location(x=0.5, z=1.9)) 

    # spawn the sensor and attach to vehicle.
    sensor = world.spawn_actor(blueprint, spawn_point, attach_to=vehicle)

    # add sensor to list of actors
    actor_list.append(sensor)
   
    sensor.listen(lambda data:processImage(data))

    bp_lib = world.get_blueprint_library() 
    spawn_points = world.get_map().get_spawn_points() 
    for i in range(100): 
        vehicle_bp = random.choice(bp_lib.filter('vehicle')) 
        npc = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))    
    for v in world.get_actors().filter('*vehicle*'): 
        v.set_autopilot(True) 
  

    
    time.sleep(120)

finally:
    print('destroying actors')
    for actor in actor_list:
        actor.destroy()
    print('done.')