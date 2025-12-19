
import glob
import os
import sys
from Planning.Interface.driverless_agent import DriverlessAgent

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
from enum import Enum
import matplotlib.pyplot as plt
import numpy as np
import cv2
# from agents.navigation.behavior_agent import BehaviorAgent  # pylint: disable=import-error
import cv2
from ultralytics import YOLO
import glob
import os
import sys
import random
import time
import numpy as np
import cv2 
import math

from Utils.tool import get_ob_box, save_fig, DRAW_ALL_SPEED_FIG
from Utils.spawn_npc_fun import spawn_npc

IM_WIDTH = 640
IM_HEIGHT = 480
SHOW_CAM = False
DRAW_ROAD_LINE = False
DRAW_OB = False

TEST_ID = 6   # 测试条件
FPS = 20 # 50慢 # 20快

class ObState(Enum):
    VOID = -1
    ASSIGN = 1
    RANDOM = 2

import random
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import math

IM_WIDTH = 640  
IM_HEIGHT = 480


model = YOLO('carla.pt')

def color_filter(image):
    #convert to HLS to mask based on HLS
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)   # convert RGB into Hls
    lower = np.array([0,190,0])   
    upper = np.array([255,255,255])
    #To get the white lane lines, we'll be getting rid of any pixels that have a Lightness value that's less than 190.
    
    yellower = np.array([10,0,90])    
    yelupper = np.array([50,255,255])
    #To get the yellow lane lines, 
    #we'll be getting rid of any pixels with a Hue value outside of 10 and 50 and a high Saturation value. 
    
    yellowmask = cv2.inRange(hls, yellower, yelupper)      
    whitemask = cv2.inRange(hls, lower, upper)

    
    mask = cv2.bitwise_or(yellowmask, whitemask) # bitwise_or works as OR operation
    masked = cv2.bitwise_and(image, image, mask = mask)    
    #add the filtered yellow and white lane lines into a single image.
    
    return masked

def roi(img):
    x = int(img.shape[1])
    y = int(img.shape[0])
    # shape is polynomial shape that define area of interest 
    shape = np.array([[int(0), int(y)], [int(x), int(y)], [int(0.55*x), int(0.6*y)], [int(0.45*x), int(0.6*y)]])

    #define a numpy array with the dimensions of img, but comprised of zeros
    mask = np.zeros_like(img)

    #Uses 3 channels or 1 channel for color depending on input image
    if len(img.shape) > 2: 
        channel_count = img.shape[2]  #no of channels
        ignore_mask_color = (255,) * channel_count 
    else:
        ignore_mask_color = 255

    #creates a polygon with the mask color
    cv2.fillPoly(mask, np.int32([shape]), ignore_mask_color)

    #returns the image only where the mask pixels are not zero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)   #convert into gray 

def canny(img):
    return cv2.Canny(grayscale(img), 50, 120)
    #Edges detection in images refer to the  transitions in pixel intensity values
    #that occur between different regions.
    #minval=50
    #maxval=120




rightSlope, leftSlope, rightIntercept, leftIntercept = [],[],[],[]
def draw_lines(img, lines, thickness=5):
    global rightSlope, leftSlope, rightIntercept, leftIntercept
    rightColor=[0,255,0]
    leftColor=[255,0,0]
    
    #this is used to filter out the outlying lines that can affect the average
    #We then use the slope we determined to find the y-intercept of the filtered lines by solving for b in y=mx+b
    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = (y1-y2)/(x1-x2)
            if slope > 0.3:
                if x1 > 500 :
                    yintercept = y2 - (slope*x2)                    
                    rightSlope.append(slope)
                    rightIntercept.append(yintercept)
                else: None                
            elif slope < -0.3:
                if x1 < 600:
                    yintercept = y2 - (slope*x2)                    
                    leftSlope.append(slope)
                    leftIntercept.append(yintercept)    
                    
                    
    #We use slicing operators and np.mean() to find the averages of the 30 previous frames
    #This makes the lines more stable, and less likely to shift rapidly
    leftavgSlope = np.mean(leftSlope[-30:])
    leftavgIntercept = np.mean(leftIntercept[-30:])
    
    rightavgSlope = np.mean(rightSlope[-30:])
    rightavgIntercept = np.mean(rightIntercept[-30:])
    
    
    #Here we plot the lines and the shape of the lane using the average slope and intercepts
    try:
        left_line_x1 = int((0.65*img.shape[0] - leftavgIntercept)/leftavgSlope)
        left_line_x2 = int((img.shape[0] - leftavgIntercept)/leftavgSlope)
    
        right_line_x1 = int((0.65*img.shape[0] - rightavgIntercept)/rightavgSlope)
        right_line_x2 = int((img.shape[0] - rightavgIntercept)/rightavgSlope)

        pts = np.array([[left_line_x1, int(0.65*img.shape[0])],[left_line_x2, int(img.shape[0])],[right_line_x2, int(img.shape[0])],[right_line_x1, int(0.65*img.shape[0])]], np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.fillPoly(img,[pts],(0,0,255))      
        
        
        cv2.line(img, (left_line_x1, int(0.65*img.shape[0])), (left_line_x2, int(img.shape[0])), leftColor, 10)
        cv2.line(img, (right_line_x1, int(0.65*img.shape[0])), (right_line_x2, int(img.shape[0])), rightColor, 10)
    except ValueError:
            #I keep getting errors for some reason, so I put this here. Idk if the error still persists.
        pass
                
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    """
    #to determine the lane lines from the canny edges 
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


""" 实时显示相机图像 """
def process_img(image):
    # image.save_to_disk('output/000/%06d.png' % image.frame)
    i = np.array(image.raw_data)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i3 = i2[:, :, :3]   
    i3=cv2.resize(i3,(IM_WIDTH,IM_HEIGHT))
    results = model.predict(i3)


    image = np.copy(i3)
    interest = roi(image)
    filterimg = color_filter(interest)
    canny = cv2.Canny(grayscale(filterimg), 50, 120)
    myline = hough_lines(canny, 1, np.pi/180, 10, 20, 5)
    weighted_img = cv2.addWeighted(myline, 1, image, 0.8, 0)   # to add lines to inp
    for result in results:
        boxes = result.boxes.cpu().numpy()  # Get boxes on CPU in numpy format
    
        for box in boxes: 
             # Iterate over boxes
            
            r = box.xyxy[0].astype(int)  # Get corner points as int
            class_id = int(box.cls[0])  # Get class ID
            score = box.conf.item() 
            class_name = model.names[class_id]  # Get class name using the class ID
       
            print(f"Class: {class_name}, Box: {r},score:{score}")  # Print class name and box coordinates
            w=640
            ratio = w/200000
            h=640
        
            text = f"{class_name}: {score:.2f}" 
            cv2.rectangle(weighted_img, (r[0], r[1]), (r[2], r[3]), (0, 255, 0), 2)
            
            cv2.putText(weighted_img, text, (r[0], r[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 1)
            if class_id == 9 or class_id == 0 or class_id == 1:
                            
            
                apx_distance = round((((h-r[3]))*ratio)*4.5,1)
            
                cv2.putText(weighted_img, str(apx_distance), (r[0] + 20, r[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
                            # update for text input
                               
   
    #cv2.imshow('RGB',weighted_img)

    cv2.imshow('RGB',weighted_img)
    cv2.waitKey(1)

    
    
    
    return weighted_img/255.0


actor_list = []
try:
    """ 测试条件 """
    if TEST_ID == 1:    # 单直道避障超车
        # ego_pos = [-115.4, 4.0, 11]
        # ego_ori = [0, 180, 0]
        # ego_pos = [-315.6, 33.3, 2.0]
        # ego_ori = [0, 0, 0]
        # target_pos = [347.2, 35.6, 3.0]
        # ego_pos = [210, -4.5, 1.0]
        ego_pos = [210, -4.5+3.2,1.0]
        ego_ori = [0, 180, 0]
        target_pos = [30.2, -4.8, 1.0]
        # target_pos = [5.9, -115.3, 1.0]

        # ob_pos = [-200, 33.3, 5.0]
        # ob_ori = [0, 0, 0]
        # ob_pos = [180, -6.7, 1.0]
        ob_pos = [180, -6.7+3.5, 1.0]
        # ob_pos = [180, -6.7+3.5+3.0, 1.0]
        ob_ori = [0, 180, 0]
        ob_state = ObState.ASSIGN
    elif TEST_ID == 2:      # 双车道避障超车+切换车道
        ego_pos = [210, -4.5 ,1.0]
        ego_ori = [0, 180, 0]
        target_pos = [30.2, -4.8, 1.0]
        # ob_pos = [180, -6.7, 1.0]
        # ob_pos = [180, -6.7-1.0, 1.0]
        ob_pos = [180, -6.7+1.5, 1.0]
        ob_ori = [0, 180, 0]
        ob_state = ObState.ASSIGN
    elif TEST_ID == 3:      # 动态障碍物
        ego_pos = [210, -4.5 ,1.0]
        ego_ori = [0, 180, 0]
        target_pos = [30.2, -4.8, 1.0]
        ob_pos = [180, -6.7-0.3, 1.0]
        ob_ori = [0, 180, 0]
        ob_state = ObState.ASSIGN
    elif TEST_ID == 4:      # 直道+弯道
        ego_pos = [210, -4.5 ,1.0]
        ego_ori = [0, 180, 0]
        target_pos = [5.9, -115.3, 1.0]
        ob_pos = [180, -6.7, 1.0]
        ob_ori = [0, 180, 0]
        ob_state = ObState.ASSIGN
    elif TEST_ID == 5:      # 弯道
        ego_pos = [30.2+15, -4.8-3.5, 1.0]
        ego_ori = [0, 180, 0]
        target_pos = [5.9, -115.3+40, 1.0]
        ob_pos = [180, -6.7, 1.0]
        ob_ori = [0, 180, 0]
        ob_state = ObState.VOID
    elif TEST_ID == 6:      # 随机动态障碍物（车辆）+直道和弯道
        ego_pos = [210, -4.5 ,1.0]
        ego_ori = [0, 180, 0]
        target_pos = [5.9, -115.3, 1.0]
        ob_state = ObState.RANDOM
        # ob_pos = [180, -6.7, 1.0]
        # ob_ori = [0, 180, 0]


    client = carla.Client('localhost', 2000)
    world = client.get_world()  
    blueprint_library = world.get_blueprint_library()    # 获取世界
      # 访问蓝图
    vehicle_bp = blueprint_library.find('vehicle.tesla.model3') # 提供特斯拉模型3的默认蓝图
    vehicle_bp.set_attribute('color', '255,0,0')


    #ego_point = carla.Transform(carla.Location(x=ego_pos[0],y=ego_pos[1],z=ego_pos[2]),
    #    carla.Rotation(pitch=ego_ori[0],yaw=ego_ori[1],roll=ego_ori[2]))
    #ego_point = random.choice(world.get_map().get_spawn_points())
     # 随机出生点
    spawn_point = random.choice(world.get_map().get_spawn_points())
    ego_car = world.spawn_actor(vehicle_bp, spawn_point)  # 创建汽车
    ego_car.apply_control(carla.VehicleControl(throttle=0.4, steer=0.0))    # 油门，转向
    ego_car.set_autopilot(True)  # 设置自动驾驶
    actor_list.append(ego_car)
    print("[INFO] Create car")

    camera_bp = blueprint_library.find('sensor.camera.rgb') 
    camera_bp.set_attribute('image_size_x', f'{IM_WIDTH}')
    camera_bp.set_attribute('image_size_y', f'{IM_HEIGHT}')
    camera_bp.set_attribute('fov', '90')
        # 后上方carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0))
    cam_point = carla.Transform(carla.Location(x=0.5, z=1.9))
        #cam_point = carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0))
    camera = world.spawn_actor(camera_bp, cam_point, attach_to=ego_car)
        #camera = world.spawn_actor(camera_bp, cam_point, obstacle, carla.AttachmentType.SpringArm)
    actor_list.append(camera)
    camera.listen(lambda data: process_img(data))
    
    """ 创建障碍物 """
    obstacle_list = []
    if ob_state == ObState.ASSIGN:   # 创建人为指定位置的障碍物
        obstacle_bp = blueprint_library.find('vehicle.tesla.model3')    # 创建障碍物
        obstacle_bp.set_attribute('color', '0,0,0')
        ob_point = carla.Transform(carla.Location(x=ob_pos[0],y=ob_pos[1],z=ob_pos[2]),
            carla.Rotation(pitch=ob_ori[0],yaw=ob_ori[1],roll=ob_ori[2]))
        obstacle = world.spawn_actor(obstacle_bp, ob_point)
        if TEST_ID == 3:
            throttle = 0.2
            brake = 0.0
        else:
            throttle = 0.0
            brake = 1.0
        obstacle.apply_control(carla.VehicleControl(throttle=throttle, steer=0.0, brake=brake))
        actor_list.append(obstacle)
        obstacle_list.append(obstacle)
        print("[INFO] Create obstacle")
    if SHOW_CAM and ob_state == ObState.ASSIGN:     # 加载相机
        camera_bp = blueprint_library.find('sensor.camera.rgb') 
        camera_bp.set_attribute('image_size_x', f'{IM_WIDTH}')
        camera_bp.set_attribute('image_size_y', f'{IM_HEIGHT}')
        camera_bp.set_attribute('fov', '90')
        # 后上方carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0))
        cam_point = carla.Transform(carla.Location(x=0.5, z=1.9))
        #cam_point = carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0))
        camera = world.spawn_actor(camera_bp, cam_point, attach_to=ego_car)
        #camera = world.spawn_actor(camera_bp, cam_point, obstacle, carla.AttachmentType.SpringArm)
        actor_list.append(camera)
        camera.listen(lambda data: process_img(data))
    if ob_state == ObState.RANDOM:  # 创建随机自主运动的障碍物
        vehicles_id = spawn_npc(FPS)
        for id in vehicles_id:
            obstacle_list.append(world.get_actor(id))
    time.sleep(1.0)

    """ 初始绘制 """
    debug = world.debug
    if DRAW_ROAD_LINE:    # 绘制道路边界
        roadline_arr = world.get_level_bbs(carla.CityObjectLabel.Roads)
        # ob_arr = world.get_level_bbs(carla.CityObjectLabel.Vehicles)
        # world_snapshot = world.get_snapshot()
        #for roadline in roadline_arr:
         #   debug.draw_box(roadline,roadline.rotation, 0.5, carla.Color(0,0,255,0),0)
        # transform = ego_car.get_transform()
        # del_ind = -1
        # for i, ob in enumerate(ob_arr):
        #     if ob.contains(transform.location,carla.Transform()) :
        #         print(i)
        #         del_ind = i
        #         break
        # if del_ind != -1:
        #    del(ob_arr[del_ind])
        # print(len(ob_arr))
        # for ob in ob_arr:
        #     debug.draw_box(ob,ob.rotation, 0.2, carla.Color(0,255,0,0),0) 
  #  if DRAW_OB and ob_state == ObState.ASSIGN: # 障碍物边框
        # time.sleep(0.5)
        #for vehicle in obstacle_list:
         #   ob_box, ob_rot = get_ob_box(world,vehicle)
          #  ob_vertices = ob_box.get_world_vertices(carla.Transform())        # 得到障碍物的顶点
           # for ob_vertice in ob_vertices:
            #    pos = ob_vertice
             #   debug.draw_line(pos, pos, 0.1, carla.Color(255,0,0,0), 0)
              #  debug.draw_box(ob_box, ob_rot, 0.2, carla.Color(0,255,0,0),0)
       # ob_box, ob_rot = get_ob_box(world,obstacle)
        # ob_vertices = ob_box.get_world_vertices(carla.Transform())        # 得到障碍物的顶点
        # for ob_vertice in ob_vertices:
        #     pos = ob_vertice
        #     debug.draw_line(pos, pos, 0.1, carla.Color(255,0,0,0), 0)
        #debug.draw_box(ob_box, ob_rot, 0.2, carla.Color(0,255,0,0),0)
   # print("[INFO] Init debugger")
    
    #""" 其他设置 """
    agent = DriverlessAgent(ego_car, obstacle_list, FPS)   # 自动驾驶服务创建
    destination = carla.Location(x=target_pos[0],y=target_pos[1],z=target_pos[2])   
    agent.set_destination(agent.vehicle.get_location(), destination, clean=True)    # 设置目标点
    print("[INFO] Set destination")

    settings = world.get_settings()
    settings.fixed_delta_seconds = 1.0/FPS#0.02#0.05
    settings.synchronous_mode = True    # 同步模式
    world.apply_settings(settings)
    print("[INFO] World synchronous")

    """ 开始测试 """
    print("[INFO] Mission start")
    spectator = world.get_spectator()
    tot_target_reached = 0
    num_min_waypoints = 21
    speed_record = []
    while True:
        # 监视者
        transform = ego_car.get_transform()
        spectator.set_transform(carla.Transform(transform.location + carla.Location(z=40),
            carla.Rotation(pitch=-90,yaw=180)))
        world.tick()
        agent.update_information()
         # Set new destination when target has been reached
        if len(agent.get_local_planner().waypoints_queue) < num_min_waypoints:
            agent.reroute(spawn_points)
            tot_target_reached += 1
            world.hud.notification("The target has been reached " +
                                     str(tot_target_reached) + " times.", seconds=4.0)
            break
        if len(agent.get_local_planner().waypoints_queue) == 0:
            print("[INFO] Target reached, mission accomplished...")
            break
        speed_limit = ego_car.get_speed_limit()
        agent.get_local_planner().set_speed(speed_limit)
        control = agent.run_step()
        ego_car.apply_control(control)
        speed_record.append(control.throttle)
    print("[INFO] Mission over")
    if DRAW_ALL_SPEED_FIG:
        plt.figure()
        plt.plot(speed_record)
        save_fig()
    # time.sleep(60)

finally:
    """ 摧毁actor """
    print('destroying actors')
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)
    for actor in actor_list:
        actor.destroy()
    print('done.')
    #if ob_state == ObState.RANDOM:
     #   for vehicle in obstacle_list:
      #      vehicle.destroy()