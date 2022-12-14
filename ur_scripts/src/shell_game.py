#!/usr/bin/env python

from cmath import pi
import sys
import copy
import rospy
import moveit_commander
import geometry_msgs
import moveit_msgs.msg
import cv2
import os
from imutils.video import VideoStream
import time
import pyrealsense2 as rs
import numpy as np
import argparse



class ShellGame(object):
    
    def __init__(self):
    # ##########################################################################
    # INITIALIZING MOVEIT + SOME VARIABLES
    # ##########################################################################    
        super(ShellGame, self).__init__()

        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("move_group_python_calculator", anonymous=True)

        robot = moveit_commander.RobotCommander()
        scene = moveit_commander.PlanningSceneInterface()

        group_name = "manipulator"
        move_group = moveit_commander.MoveGroupCommander(group_name)

        planning_frame = move_group.get_planning_frame()
        print("============ Planning frame: %s" % planning_frame)

        eef_link = move_group.get_end_effector_link()
        print("============ End effector link: %s" % eef_link)

        group_names = robot.get_group_names()
        print("============ Available Planning Groups:", robot.get_group_names())

        print("============ Printing robot state")
        print(robot.get_current_state())
        print("")
        
        # Misc variables
        self.box_name = ""
        self.robot = robot
        self.scene = scene
        self.move_group = move_group
        self.planning_frame = planning_frame
        self.eef_link = eef_link
        self.group_names = group_names
        
        
        
    # ##########################################################################
    # LOW LEVEL MOVEMENT FUNCTIONS
    # These functions are used as their name describes - when called, they move the robot arm in a cartesian direction, or towards a specific goal pose. 
    # ########################################################################## 
    def move_left(self, scale=0.05, zscale=0.1):
        move_group = self.move_group
        waypoints = []
        wpose = move_group.get_current_pose().pose
        wpose.position.x -= scale * 0.5
        waypoints.append(copy.deepcopy(wpose))
        (plan, fraction) = move_group.compute_cartesian_path(waypoints, scale, 0.0)
        return plan, fraction

    def move_right(self, scale=0.05, zscale=0.1):
        move_group = self.move_group
        waypoints = []
        wpose = move_group.get_current_pose().pose
        wpose.position.x += scale * 0.5
        waypoints.append(copy.deepcopy(wpose))
        (plan, fraction) = move_group.compute_cartesian_path(waypoints, scale, 0.0)
        return plan, fraction
    
    def move_up(self, scale=0.05, zscale=0.1):
        move_group = self.move_group
        waypoints = []
        wpose = move_group.get_current_pose().pose
        wpose.position.y += scale * 0.5
        waypoints.append(copy.deepcopy(wpose))
        (plan, fraction) = move_group.compute_cartesian_path(waypoints, scale, 0.0)
        return plan, fraction
    
    def move_down(self, scale=0.05, zscale=0.1):
        move_group = self.move_group
        waypoints = []
        wpose = move_group.get_current_pose().pose
        wpose.position.y -= scale * 0.5
        waypoints.append(copy.deepcopy(wpose))
        (plan, fraction) = move_group.compute_cartesian_path(waypoints, scale, 0.0)
        return plan, fraction
    
    def execute_plan(self, plan):
        move_group = self.move_group
        move_group.execute(plan, wait=True)
        
    def go_to_joint_state(self):
        move_group = self.move_group
        joint_goal = move_group.get_current_joint_values()
        joint_goal[0] = 0
        joint_goal[1] = -2 * pi / 8
        joint_goal[2] = 0
        joint_goal[3] = -2 * pi / 4
        joint_goal[4] = 0
        joint_goal[5] = 2 * pi / 6 
        move_group.go(joint_goal, wait=True)
        move_group.stop()
        
    def go_to_initial_goal(self):
        move_group = self.move_group
        joint_goal = move_group.get_current_joint_values()
        joint_goal[0] = -1.5708563963519495
        joint_goal[1] = -1.9903510252581995
        joint_goal[2] = -1.546737019215719
        joint_goal[3] = -1.1759093443499964
        joint_goal[4] = 1.5703076124191284
        joint_goal[5] = -2.296692434941427
        move_group.go(joint_goal, wait=True)
        move_group.stop()
        
    def look_to_camera(self):
        move_group = self.move_group
        joint_goal = move_group.get_current_joint_values()
        joint_goal[0] = -1.5708563963519495
        joint_goal[1] = -1.9903510252581995
        joint_goal[2] = -1.546737019215719
        joint_goal[3] = 1.1759093443499964
        joint_goal[4] = 1.5703076124191284
        joint_goal[5] = -2.296692434941427
        move_group.go(joint_goal, wait=True)
        move_group.stop()
        
    def move_left_goal(self, control):
        move_group = self.move_group
        wpose = move_group.get_current_pose().pose
        pose_goal = wpose
        pose_goal.position.x -= control
        
        move_group.set_pose_target(pose_goal)
        success = move_group.go(wait=True)
        move_group.stop()
        move_group.clear_pose_targets()
        
    def move_right_goal(self, control):
        move_group = self.move_group
        wpose = move_group.get_current_pose().pose
        pose_goal = wpose
        pose_goal.position.x += control
        
        move_group.set_pose_target(pose_goal)
        success = move_group.go(wait=True)
        move_group.stop()
        move_group.clear_pose_targets()
        
    def move_up_goal(self, control):
        move_group = self.move_group
        wpose = move_group.get_current_pose().pose
        pose_goal = wpose
        pose_goal.position.y += control
        
        move_group.set_pose_target(pose_goal)
        success = move_group.go(wait=True)
        move_group.stop()
        move_group.clear_pose_targets()
        
    def move_down_goal(self, control):
        move_group = self.move_group
        wpose = move_group.get_current_pose().pose
        pose_goal = wpose
        pose_goal.position.y -= control
        
        move_group.set_pose_target(pose_goal)
        success = move_group.go(wait=True)
        move_group.stop()
        move_group.clear_pose_targets()
        
    def move_tall_goal(self, control):
        move_group = self.move_group
        wpose = move_group.get_current_pose().pose
        pose_goal = wpose
        pose_goal.position.z -= control
        
        move_group.set_pose_target(pose_goal)
        success = move_group.go(wait=True)
        move_group.stop()
        move_group.clear_pose_targets()
        
    def move_short_goal(self, control):
        move_group = self.move_group
        wpose = move_group.get_current_pose().pose
        pose_goal = wpose
        pose_goal.position.z += control
        
        move_group.set_pose_target(pose_goal)
        success = move_group.go(wait=True)
        move_group.stop()
        move_group.clear_pose_targets()
        
    def move_rel_location(self, x, y):
        move_group = self.move_group
        wpose = move_group.get_current_pose().pose
        pose_goal = wpose
        pose_goal.position.x += x
        pose_goal.position.y += y
        
        move_group.set_pose_target(pose_goal)
        success = move_group.go(wait=True)
        move_group.stop()
        move_group.clear_pose_targets()
        
    def scan_movement_up(self):
        move_group = self.move_group
        waypoints = []
        wpose = move_group.get_current_pose().pose
        wpose.position.y += 0.05
        waypoints.append(copy.deepcopy(wpose))
        wpose.position.z -= 0.05
        waypoints.append(copy.deepcopy(wpose))
        (plan, fraction) = move_group.compute_cartesian_path(waypoints, 0.001, 0.0)

        move_group = self.move_group
        move_group.execute(plan, wait=True)
    
    def scan_movement_up_inverse(self):
        move_group = self.move_group
        waypoints = []
        wpose = move_group.get_current_pose().pose
        wpose.position.z += 0.05
        waypoints.append(copy.deepcopy(wpose))
        wpose.position.y -= 0.05
        waypoints.append(copy.deepcopy(wpose))
        (plan, fraction) = move_group.compute_cartesian_path(waypoints, 0.001, 0.0)

        move_group = self.move_group
        move_group.execute(plan, wait=True)
    
    def scan_movement_down(self):
        move_group = self.move_group
        waypoints = []
        wpose = move_group.get_current_pose().pose
        wpose.position.y -= 0.05
        waypoints.append(copy.deepcopy(wpose))
        wpose.position.z -= 0.05
        waypoints.append(copy.deepcopy(wpose))
        (plan, fraction) = move_group.compute_cartesian_path(waypoints, 0.001, 0.0)

        move_group = self.move_group
        move_group.execute(plan, wait=True)
        
    def scan_movement_down_inverse(self):
        move_group = self.move_group
        waypoints = []
        wpose = move_group.get_current_pose().pose
        wpose.position.z += 0.05
        waypoints.append(copy.deepcopy(wpose))
        wpose.position.y += 0.05
        waypoints.append(copy.deepcopy(wpose))
        (plan, fraction) = move_group.compute_cartesian_path(waypoints, 0.001, 0.0)

        move_group = self.move_group
        move_group.execute(plan, wait=True)
        
    def scan_movement_left(self):
        move_group = self.move_group
        waypoints = []
        wpose = move_group.get_current_pose().pose
        wpose.position.x -= 0.05
        waypoints.append(copy.deepcopy(wpose))
        wpose.position.z -= 0.05
        waypoints.append(copy.deepcopy(wpose))
        (plan, fraction) = move_group.compute_cartesian_path(waypoints, 0.001, 0.0)

        move_group = self.move_group
        move_group.execute(plan, wait=True)
    
    def scan_movement_left_inverse(self):
        move_group = self.move_group
        waypoints = []
        wpose = move_group.get_current_pose().pose
        wpose.position.z += 0.05
        waypoints.append(copy.deepcopy(wpose))
        wpose.position.x += 0.05
        waypoints.append(copy.deepcopy(wpose))
        (plan, fraction) = move_group.compute_cartesian_path(waypoints, 0.001, 0.0)

        move_group = self.move_group
        move_group.execute(plan, wait=True)
        
    def scan_movement_right(self):
        move_group = self.move_group
        waypoints = []
        wpose = move_group.get_current_pose().pose
        wpose.position.x += 0.05
        waypoints.append(copy.deepcopy(wpose))
        wpose.position.z -= 0.05
        waypoints.append(copy.deepcopy(wpose))
        (plan, fraction) = move_group.compute_cartesian_path(waypoints, 0.001, 0.0)

        move_group = self.move_group
        move_group.execute(plan, wait=True)
        
    def scan_movement_right_inverse(self):
        move_group = self.move_group
        waypoints = []
        wpose = move_group.get_current_pose().pose
        wpose.position.z += 0.05
        waypoints.append(copy.deepcopy(wpose))
        wpose.position.x -= 0.05
        waypoints.append(copy.deepcopy(wpose))
        (plan, fraction) = move_group.compute_cartesian_path(waypoints, 0.001, 0.0)

        move_group = self.move_group
        move_group.execute(plan, wait=True)
    
    
    
    # ##########################################################################
    # HIGH LEVEL PERCEPTION AND MOVEMENT FUNCTIONS 
    # Finds and records locations of all bowls within scene
    # ##########################################################################
    def scan_for_objects(self, pipeline, target_number):
        continue_flag = True
        triggerTime = time.time()
        startTime = time.time()
        waitTime = 3
        while continue_flag:
            #Grab frames from RealSense D435i
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Apply colormap on depth image for depth visualization
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = color_image.shape

            start = time.time()
            if (start-triggerTime >= 0.1):
                #Create a color filter to apply to the image. The one below sorts for BGR values in the range of (0,0,80) to (60,60,255); essentially, a semi-light-independent search for the red shells.
                colfil1 = np.array([0, 0, 80])
                colfil2 = np.array([60, 60, 255]) 
                mask = cv2.inRange(color_image, colfil1, colfil2)
                color_mask = cv2.bitwise_and(color_image,color_image, mask= mask)
                cv2.addWeighted(color_mask, 1, color_image, 1, 0, color_image)
                #Create OpenCV contours around any red objects found above.
                contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                lmList = [300, 250]

                # All contours detected
                bounded_contours = []
                for pic, contour in enumerate(contours): # extend this to create matrix or lists, track initial positon to estimate location in scene and store objects
                    # if counter == 0:
                    area = cv2.contourArea(contour)
                    #Filter out any small objects with area less than 5000 pixels - this stops small segments of a shell or other small objects in the environment from being localized to.
                    if (area > 5000):
                        x, y, w, h = cv2.boundingRect(contour)
                        # cv2.rectangle(qcolor_image, (x, y), (x + w, y + h), (0, 0, 255), 2) #Deprecated rectangle
                        lmList = [int(x + w/2), int(y + h/2)]

                        bounded_contours.append([x, y, w, h])

                final_contours = []

                # Creating contour check dictionary
                contour_check = {}     
                for i in range(len(bounded_contours)):
                    contour_check[i] = True

                # Looking for overlaps in contours. If there are overlaps in contours, combine them to ensure that each shell receives only one contour. 
                for i in range(len(bounded_contours)):
                    if contour_check[i] == True:
                        current_box = bounded_contours[i]
                        for j in range(i + 1, len(bounded_contours)):
                            if (((bounded_contours[j][0] < current_box[0] < bounded_contours[j][0] + bounded_contours[j][2]) or (bounded_contours[j][0] < current_box[0] + current_box[2] < bounded_contours[j][0] + bounded_contours[j][2])) and ((bounded_contours[j][0] < current_box[0] < bounded_contours[j][0] + bounded_contours[j][2]) or (bounded_contours[j][1] < current_box[1] + current_box[3] < bounded_contours[j][1] + bounded_contours[j][3]))):
                                if bounded_contours[j][0] < current_box[0]:
                                    current_box[0] = bounded_contours[j][0]
                                if (bounded_contours[j][0] + bounded_contours[j][2]) > (current_box[0] + current_box[2]):
                                    current_box[2] = bounded_contours[j][0] + bounded_contours[j][2] - current_box[0]
                                if bounded_contours[j][1] < current_box[1]:
                                    current_box[1] = bounded_contours[j][1]
                                if (bounded_contours[j][1] + bounded_contours[j][3]) > (current_box[1] + current_box[3]):
                                    current_box[3] = bounded_contours[j][1] + bounded_contours[j][3] - current_box[1]
                                contour_check[j] = False
                        final_contours.append(current_box)

                # Looking for contours entirely contained in other contours. Removing them. 
                final_contours2 = []
                for i in range(len(final_contours)):
                    for j in range(len(final_contours)):
                        if not (final_contours[i][0] < final_contours[j][0] and (final_contours[i][0] + final_contours[i][2]) > (final_contours[j][0] + final_contours[j][2]) and (final_contours[i][1] < final_contours[j][1]) and (final_contours[i][1] + final_contours[i][3]) > (final_contours[j][1] + final_contours[j][3])):
                            final_contours2.append(final_contours[j])

                final_contours = final_contours2

                largest_contours = final_contours2

                # Selecting three largest regions (future work can account for situations with more or less than 3 shells)
                largest_areas = [0, 0, 0]
                largest_contours = [[], [], []]
                for contour in bounded_contours:
                    if (contour[2] * contour[3]) > largest_areas[2]:
                        largest_areas[2] = contour[2] * contour[3]
                        largest_contours[2] = contour
                    elif (contour[2] * contour[3]) > largest_areas[1]:
                        largest_areas[1] = contour[2] * contour[3]
                        largest_contours[1] = contour
                    elif (contour[2] * contour[3]) > largest_areas[0]:
                        largest_areas[0] = contour[2] * contour[3]
                        largest_contours[0] = contour

            
                stored_pixels = []
                stored_positions = []
                # Storing positions in the form of pixel location as well as (x,y) coordinates, in meters, with respect to the camera's center as an origin. 
                for contour in largest_contours:
                    try:
                        cv2.rectangle(color_image, (contour[0], contour[1]), (contour[0] + contour[2], contour[1] + contour[3]), (255, 0, 0), 2)
                        stored_pixels.append([contour[0] + contour[2]//2, contour[1] + contour[3]//2])
                        stored_positions.append([round((contour[0] + contour[2]//2 - 320) * 0.33/640, 2), round(-1 * (contour[1] + contour[3]//2 - 240) * 0.33/640, 2)])
                    except IndexError:
                        pass

                if time.time() - startTime >= waitTime and len(stored_positions) == 3:
                    return stored_positions


                # Draws the contours. Exception handling to solve outlier cases.
                try:
                    for contour in largest_contours:
                        cv2.rectangle(color_image, (contour[0], contour[1]), (contour[0] + contour[2], contour[1] + contour[3]), (255, 0, 0), 2)
                except IndexError:
                    pass


                # Initializes the "safe" cube in which the arm will not move. When a shell is detected to be outside of the cube, the robot arm will move to correct it. 
                bounds = [295, 245, 305, 255]
                depth_limits = [-0.01, 0.01]
                cv2.rectangle(color_image, (bounds[0], bounds[1]), (bounds[2], bounds[3]), (0, 255, 0), 2)

                cv2.imshow("Video",color_image)
                cv2.waitKey(1)

                triggerTime = time.time()

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    # Looking for green patch on one of the bowls
    def scan_for_green(self, pipeline):
        green_found = False
        # triggerTime = time.time()
        delayTime = time.time()
        delay = 3
        # Allow time for alias measurements to pass
        while time.time() - delayTime <= delay:
            pass
        movement_complete = False
        iterations = 0
        while not movement_complete:
            # Check iteration to see which movement to produce. Future work can see TVFs replace this section. 
            if iterations == 0:
                self.scan_movement_up()
            elif iterations == 1:
                self.scan_movement_down()
            elif iterations == 2:
                self.scan_movement_left()
            elif iterations == 3:
                self.scan_movement_right()
                
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = color_image.shape
            
            # Take in new camera frame
            # start = time.time()
            # if (start-triggerTime >= 0.1):


            #Create a color filter to apply to the image. The one below sorts for the green mark.
            colfil1 = np.array([50, 50, 0])
            colfil2 = np.array([200, 255, 20]) 
            mask = cv2.inRange(color_image, colfil1, colfil2)
            color_mask = cv2.bitwise_and(color_image,color_image, mask= mask)
            cv2.addWeighted(color_mask, 1, color_image, 1, 0, color_image)
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            #Same as shell searching, but smaller area
            for pic, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if (area > 50):
                    green_found = True
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    print("I found green!") #Sanity check
                    

            cv2.imshow("Video",color_image)
            cv2.waitKey(1)
            
            # triggerTime = time.time()
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Check iteration to perform inverse action 
            if iterations == 0:
                self.scan_movement_up_inverse()
            elif iterations == 1:
                self.scan_movement_down_inverse()
            elif iterations == 2:
                self.scan_movement_left_inverse()
            elif iterations == 3:
                self.scan_movement_right_inverse()
                return green_found
            iterations = iterations + 1   
            
    # Moves manipulator directly above target object
    def move_to_object(self, pipeline):
        at_object = False
        triggerTime = time.time()
        while not at_object:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = color_image.shape
            
            depth_target = 0.20
            avgdepth = depth_target
            
            # Receive and process new image data if timer has expired
            start = time.time()
            if (start-triggerTime >= 0.1):
                colfil1 = np.array([0, 0, 80])
                colfil2 = np.array([0, 0, 255]) #currently red
                mask = cv2.inRange(color_image, colfil1, colfil2)
                color_mask = cv2.bitwise_and(color_image,color_image, mask= mask)
                cv2.addWeighted(color_mask, 1, color_image, 1, 0, color_image)
                contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
                lmList = [300, 250]
                
                # Thin contour list
                for pic, contour in enumerate(contours):
                    # if counter == 0:
                    area = cv2.contourArea(contour)
                    if (area > 500):
                        x, y, w, h = cv2.boundingRect(contour)
                        cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        lmList = [int(x + w/2), int(y + h/2)]
                        
                # Localization tolerance bounds
                bounds = [295, 245, 305, 255]
                depth_limits = [-0.01, 0.01]
                cv2.rectangle(color_image, (bounds[0], bounds[1]), (bounds[2], bounds[3]), (0, 255, 0), 2)

                cv2.imshow("Video",color_image)
                cv2.waitKey(1)
                
                # Attempt to track object center using contours
                try:
                    cx_dot = lmList[0]
                    cy_dot = lmList[1]
                    depth = 0
                    count = 0
                    try:
                        for i in range(lmList[0] - 25, lmList[0] + 25):
                            for j in range(lmList[1] - 25, lmList[1] + 25):
                                depth = depth + depth_frame.get_distance(i,j)
                                count = count+1
                        avgdepth = depth/count
                        cv2.rectangle(color_image, (lmList[0] - 10, lmList[0] + 10), (lmList[1] - 10, lmList[1] + 10), (0, 0, 255), 2)
                        if avgdepth < .1:
                            avgdepth = depth_target
                        
                        print(avgdepth)
                    # Thrown randomly from async with camera
                    except RuntimeError:
                        pass
                # Thrown when contour not detected, use pseudo-values that don't induce movement
                except IndexError:
                    cx_dot = bounds[0] + (bounds[2] - bounds[0]) / 2
                    cy_dot = bounds[1] + (bounds[3] - bounds[1]) / 2
                
                # Translate center deviance to actionable motion with proportional control
                if cx_dot < bounds[0]:
                    print("Out of bounds (left)")
                    control = abs(cx_dot - bounds[0]) * 0.02/100
                    self.move_left_goal(control)
                if cy_dot < bounds[1]:
                    print("Out of bounds (up)")
                    control = abs(cy_dot - bounds[1]) * 0.02/100
                    self.move_up_goal(control)
                if cx_dot > bounds[2]:
                    print("Out of bounds (right)")
                    control = abs(cx_dot - bounds[2]) * 0.02/100
                    self.move_right_goal(control)
                if cy_dot > bounds[3]:
                    print("Out of bounds (down)")
                    control = abs(cy_dot - bounds[3]) * 0.02/100
                    self.move_down_goal(control)
                # Vertical control turned off due to lack of reliability
                if avgdepth > depth_target + depth_limits[1]:
                    print("Out of bounds (tall)")
                    control = abs(avgdepth - (depth_target + depth_limits[1])) * 10/100 if abs(avgdepth - (depth_target + depth_limits[1])) * 10/100 < 0.025 else 0.025
                    # self.move_tall_goal(control)
                if avgdepth < depth_target + depth_limits[0]:
                    print("Out of bounds (short)")
                    control = abs(avgdepth - (depth_target - depth_limits[0])) * 10/100 if abs(avgdepth - (depth_target + depth_limits[1])) * 10/100 < 0.025 else 0.025
                    # self.move_short_goal(control)
                    
                triggerTime = time.time()
                
                # Exit if at object within tolerance
                if bounds[0] <= cx_dot <= bounds[2] and bounds[1] <= cy_dot <= bounds[3]:
                    at_object = True
                
                # Use q to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    # Grabs target object
    def pick(self):
        delay = 3
        delayTime = time.time()
        self.move_left_goal(0.033)
        self.move_down_goal(0.02)
        self.move_tall_goal(0.06)
        while time.time() - delayTime < delay:
            pass
        self.move_short_goal(0.06)
            


# ##########################################################################                    
# MAIN FUNCTION 
# ##########################################################################
def main():
    try:
        print("")
        print("----------------------------------------------------------")
        print("")
        input("============ Press `Enter` to initialize ...")
        
        # Initialize ezMoney Object
        ezMoney = ShellGame()
        
        # Configure depth and color streams
        pipeline = rs.pipeline()
        config = rs.config()

        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        pipeline.start(config)
        
        input("============ Press `Enter` to assume initial pose goal ...")
        ezMoney.go_to_initial_goal()
        input()
        ezMoney.look_to_camera()
        input()
        ezMoney.go_to_initial_goal()
        
        input("============ Press `Enter` to begin control ...")
        
        # Full scanning
        ezMoney.move_short_goal(0.20)
        object_locations = ezMoney.scan_for_objects(pipeline, 3)
        print('Scan Complete!')
        print(object_locations)
        
        # Scan each bowl
        ezMoney.move_tall_goal(0.20)
        delay = 3
        greens = []
        for location in object_locations:
            delayTime = time.time()
            ezMoney.move_rel_location(location[0], location[1])
            ezMoney.move_to_object(pipeline)
            ezMoney.move_left_goal(0.033)
            ezMoney.move_down_goal(0.02)
            greens.append(ezMoney.scan_for_green(pipeline))
            ezMoney.go_to_initial_goal()
        
        # picks correct bowl
        for i in range(len(greens)):
            if greens[i] == True:
                ezMoney.go_to_initial_goal()
                ezMoney.move_rel_location(object_locations[i][0], object_locations[i][1])
                ezMoney.move_to_object(pipeline)
                ezMoney.pick()
        
        input("============ Press `Enter` to return to initial position ...")
        ezMoney.go_to_initial_goal()

        cv2.destroyAllWindows() 
        
        input("============ Press `Enter` to terminate ...")
        moveit_commander.roscpp_shutdown()
        
        print("============  Shell Game Complete!")

    except rospy.ROSInterruptException:
        return
    except KeyboardInterrupt:
        return



if __name__ == "__main__":
    main()
