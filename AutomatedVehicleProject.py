import glob
import os
import sys
import time
import numpy as np
import cv2
import math
from tensorflow.keras.models import save_model,load_model
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import tensorflow as tf

import pygame


IMAGEWIDTH=320
IMAGEHEIGHT=240
rows=[]

display_width = 640
display_height = 480
x =  0
y = 0
black = (0,0,0)
white = (255,255,255)
green = (0, 255, 0)
blue = (0, 0, 128)

def carla_preprocess(image):
	image=image[100:200,:,:]
	image=cv2.cvtColor(image,cv2.COLOR_RGB2YUV)
	image=cv2.GaussianBlur(image,(3,3),0)
	image=cv2.resize(image,(200,66))
	image=image/255
	return image


def run_inference_for_single_image(model_fn, image):
  input_tensor = tf.convert_to_tensor(image)
  input_tensor = input_tensor[tf.newaxis,...]
  output_dict = model_fn(input_tensor)

  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections
  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

  return output_dict

def store_image_data(imagedata,ourvehicle,model,model2,gameDisplay,clock,category_index):
	brakevalue=0.0
	imagedata.convert(cc.Raw)
	i=np.array(imagedata.raw_data)
	i2=i.reshape((IMAGEHEIGHT,IMAGEWIDTH,4))
	i3=i2[:, :, : 3]
	i4 = i3[:, :, ::-1]

	
	v=ourvehicle.get_velocity()
	current_speed=float(math.sqrt(v.x**2 + v.y**2 + v.z**2))
	throttlevalue=1.0-(current_speed/8.0)
	copy_i3=np.copy(i3)
	

	finalimage=carla_preprocess(copy_i3)
	newimg=np.array([finalimage])
	#print(newimg.shape)
	prediction=model.predict(newimg)[0]
	steervalue=float(prediction[0])

	#ourvehicle.apply_control(carla.VehicleControl(throttle=throttlevalue,steer=steervalue))
	detections=run_inference_for_single_image(model2,i4)
	image_np_with_detections = i4.copy()

	if ourvehicle.is_at_traffic_light():
		traffic_light = ourvehicle.get_traffic_light()
		if traffic_light.get_state() == carla.TrafficLightState.Red or traffic_light.get_state() == carla.TrafficLightState.Yellow:
			brakevalue=1.0

			viz_utils.visualize_boxes_and_labels_on_image_array(image_np_with_detections,detections['detection_boxes'],detections['detection_classes'],detections['detection_scores'],category_index,use_normalized_coordinates=True,max_boxes_to_draw=1,min_score_thresh=.2,agnostic_mode=False)
	else:
		viz_utils.visualize_boxes_and_labels_on_image_array(image_np_with_detections,detections['detection_boxes'],detections['detection_classes'],detections['detection_scores'],category_index,use_normalized_coordinates=True,max_boxes_to_draw=1,min_score_thresh=.4,agnostic_mode=False)



	ourvehicle.apply_control(carla.VehicleControl(throttle=throttlevalue,steer=steervalue,brake=brakevalue))


	font = pygame.font.Font('freesansbold.ttf', 12)
	text = font.render("Steer: "+str(steervalue), True, green, blue)
	text2=font.render("Throttle: "+str(throttlevalue), True, green, blue)
	text3=font.render("Brake: "+str(brakevalue), True, green, blue)
	text4=font.render("Speed: "+str(current_speed)+" m/s", True, green, blue)

	textRect = text.get_rect()
	textRect2 = text2.get_rect()
	textRect3 = text3.get_rect()
	textRect4 = text4.get_rect()
	textRect.center = (100, display_height-display_height//8)
	textRect2.center = (100, 440)
	textRect3.center = (460, 420)
	textRect4.center = (510, 440)
	carImg=pygame.surfarray.make_surface(image_np_with_detections.swapaxes(0,1))
	carImg = pygame.transform.smoothscale(carImg, (640,480)) 
	rotated_image = pygame.transform.rotate(carImg, -90)
	gameDisplay.blit(carImg, (x,y))
	gameDisplay.blit(text, textRect)
	gameDisplay.blit(text2, textRect2)
	gameDisplay.blit(text3, textRect3)
	gameDisplay.blit(text4, textRect4)
	pygame.display.update()
	clock.tick(60)


def store_image_data1(imagedata,ourvehicle,model):
	global rows
	traffic_light=ourvehicle.get_traffic_light()
	if traffic_light is not None:
		if traffic_light.get_state() == carla.TrafficLightState.Red:
			i=np.array(imagedata.raw_data)
			i2=i.reshape((IMAGEHEIGHT,IMAGEWIDTH,4))
			i3=i2[:, :, : 3]
			directory=r'D:\WindowsNoEditor\PythonAPI\examples\tt'
			os.chdir(directory)
			cv2.imwrite('out%08d.jpg' % imagedata.frame,i3)
			directory2=r'D:\WindowsNoEditor\PythonAPI\examples'
			os.chdir(directory2)
			imagepath='D:\WindowsNoEditor\PythonAPI\examples\tt\out%08d.jpg' % imagedata.frame
			rows.append([imagepath,np.array([1,0,0,0])])
			traffic_light.set_state(carla.TrafficLightState.Green)
			traffic_light.set_green_time(4.0)

		if traffic_light.get_state() == carla.TrafficLightState.Yellow:
			i=np.array(imagedata.raw_data)
			i2=i.reshape((IMAGEHEIGHT,IMAGEWIDTH,4))
			i3=i2[:, :, : 3]
			directory=r'D:\WindowsNoEditor\PythonAPI\examples\tt'
			os.chdir(directory)
			cv2.imwrite('out%08d.jpg' % imagedata.frame,i3)
			directory2=r'D:\WindowsNoEditor\PythonAPI\examples'
			os.chdir(directory2)
			imagepath='D:\WindowsNoEditor\PythonAPI\examples\tt\out%08d.jpg' % imagedata.frame
			rows.append([imagepath,np.array([0,1,0,0])])
			
	return 0





#MAIN CODE---------------START
if __name__=="__main__":

	try:
	    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
	        sys.version_info.major,
	        sys.version_info.minor,
	        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
	except IndexError:
	    pass

	import carla
	from carla import ColorConverter as cc
	actorList=[] #like other vehicles pedestrians
	try:
		#connect client to the server
		#the world is the carla simulation and we as clients can load and reload the simulation/world
		#there are actors such as pedestrians,sensors and vehicles in the world and the layouts for those actors is called blueprint
		client=carla.Client("localhost",2000)
		client.set_timeout(3.0)
		world=client.get_world()
		print("connected")

		blueprint_library=world.get_blueprint_library()
		vehicle_bp= blueprint_library.filter('vehicle.audi.a2')[0] #we choose a vehicle filtered accoring to id
		#vehicle_bp=blueprint_library.filter('vehicle.harley-davidson.low_rider')[0]
		#spawn_point=random.choice(world.get_map().get_spawn_points())
		#print(spawn_point)

		spawn_point=carla.Transform(carla.Location(x=26.940020, y=302.570007, z=0.500000), carla.Rotation(pitch=0.000000, yaw=-179.999634, roll=0.000000))
		
		

		
		ourvehicle=world.spawn_actor(vehicle_bp,spawn_point) #make/spawn our vehicle in the simulation

		 #since we want to control our vehicle ourselves
		actorList.append(ourvehicle)
		weather = carla.WeatherParameters( cloudiness=0.0, precipitation=0.0, precipitation_deposits=0.0,wind_intensity=0.0,sun_azimuth_angle=70.0, sun_altitude_angle=70.0,fog_density=0.0, fog_distance=0.0, fog_falloff=0.0, wetness=0.0)
	
		#weather = carla.WeatherParameters.CloudyNoon
		world.set_weather(weather)

		#we are making use of the camera sensor and the collision sensor
		cam_bp=blueprint_library.find("sensor.camera.rgb")
		#obstacle_bp=blueprint_library.find("sensor.other.obstacle")
		cam_bp.set_attribute("image_size_x",f"{IMAGEWIDTH}")
		cam_bp.set_attribute("image_size_y",f"{IMAGEHEIGHT}")
		cam_bp.set_attribute("fov","110")
		#cam_bp.set_attribute("enable_postprocess_effects","true")
		#cam_bp.set_attribute("sensor_tick","0.5")
		#we need to spawn the rgb camera on the vehicle cuz its attached to it
		#relative_spawn_point=carla.Transform(carla.Location(x=-5.5,z=2.5)) #this location is relative to our vehicle
		

		#relative_spawn_point=carla.Transform(carla.Location(x=-1.5, z=2.5))
		relative_spawn_point=carla.Transform(carla.Location(x=-0.5,z=2.5)) #this location is relative to our vehicle
		camera_sensor=world.spawn_actor(cam_bp,relative_spawn_point,attach_to=ourvehicle)
		#obstacle_sensor=world.spawn_actor(obstacle_bp,relative_spawn_point,attach_to=ourvehicle)
		actorList.append(camera_sensor)
		#actor_list = world.get_actors().filter('vehicle.mustang.mustang')
		#secondcar=actor_list[1]
		#secondcar.destroy()

		#print(actor_list)

		#actorList.append(obstacle_sensor)
		model=load_model('D:\WindowsNoEditor\PythonAPI\examples\save_model-rgbdata-nvdia-1output-balanced\saved_model')
		model2=tf.saved_model.load("D:/WindowsNoEditor/PythonAPI/examples/workspace/exported-models/my_model_ssd_new_new/saved_model")
		PATH_TO_LABELS = "D:/WindowsNoEditor/PythonAPI/examples/workspace/data-new/label_map.pbtxt"
		category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                 use_display_name=True)		



		pygame.init()
		gameDisplay = pygame.display.set_mode((display_width,display_height),pygame.RESIZABLE,
            pygame.HWSURFACE | pygame.DOUBLEBUF)

		pygame.display.set_caption('Capstone Project')
		count=1
		clock = pygame.time.Clock()
		crashed = False
		while not crashed:
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					crashed = True
					for actor in actorList:
						actor.destroy()
			gameDisplay.fill(black)
			pygame.display.flip()
			camera_sensor.listen(lambda imagedata: store_image_data(imagedata,ourvehicle,model,model2,gameDisplay,clock,category_index))
			time.sleep(200)
		pygame.quit()
		quit()


		pass
	finally:
		for actor in actorList:
			actor.destroy()
		print("All actors destroyed!")
    		
'''
		actor_list = world.get_actors()

		for vehicle in actor_list.filter('vehicle.audi.a2'):
			print("here")
			print(vehicle)
			vehicle.destroy()

'''



#MAIN CODE END---------------