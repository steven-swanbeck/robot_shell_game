# ROS Packages with Python Scripts Created for ME 396P Final Project, Fall 2022, Team 09

## Mission
The goal of this project was to give a UR3 manipulator the ability to play a modified version of the often deceitful shell game, in which a gamemaster challenges an individual to track an object hidden under one of three shells as they are shuffled about. If the player correctly guesses under which shell the object is at the end of the game, they earn a prize. However, this game is usually played deceptively, where the gamemaster will secretly manipulate the location of the item out of sight of the player. 

Here, we present an alternative formation of this game, in which you, the human player, act as gamemaster and a robot must guess the location of the item after the shells have been shuffled. The robot does not watch the shuffling, meaning it may appear to make its choices randomly, giving the player a 2/3 chance of winning, meaning the formulation of the game in this way not only puts power in the player's hands to shuffle, but also doubles their random chance of winning. However, to preserve the deceptive nature of the game, the human handler of the robot marks the target bowl discretely, allowing the robot to accurately detect the correct bowl using its perceptive abilities and win money from unsuspecting players.


https://user-images.githubusercontent.com/99771915/202348599-81c0b833-6eca-4082-be7f-1034f2cbd8e7.mp4

---

## Contents
1. **ur_moveit_config**
    * The ur_moveit_config package contains the simplest possible MoveIt configuration for a Universal Robots UR3 manipulator developed using the MoveIt Setup Assistant. It depends on UR-provided packages which can be found at the [Universal Robots ROS Drivers](https://github.com/UniversalRobots/Universal_Robots_ROS_Driver) page.
2. **ur_scripts**
    * The ur_scripts package contains the Python code used to make the robot execute operations. In order for the scripts to properly execute, one must be using the same experimental setup (1 UR3 robot arm, 1 Intel RealSense D435i camera + end effector, 3 Red Shells, 1 Green Marking on a Shell)
---

## Requirements, Dependencies, and Building
These packages are built and tested on a system running ROS1 noetic on Ubuntu 20.04. Users are assumed to already have ROS noetic installed on a machine running Ubuntu 20.04 to execute this demonstration. Details of ROS installation can be found on the [ROS Tutorials](http://wiki.ros.org/ROS/Tutorials) webpage.

Use of these packages in a non-simulated environment requires the use of the official [Universal Robots ROS Drivers](https://github.com/UniversalRobots/Universal_Robots_ROS_Driver).   
1. Create a Catkin workspace:
```console
mkdir -p catkin_ws/src && cd catkin_ws
```
2. Clone the contents of this repository:
```console
git clone https://github.com/steven-swanbeck/ur3_calculator.git src/lightning_talk
```
3. Clone the UR Robots ROS Driver:
```console
git clone https://github.com/UniversalRobots/Universal_Robots_ROS_Driver.git src/Universal_Robots_ROS_Driver
```
4. And the associated description packages:
```console
git clone -b melodic-devel-staging https://github.com/ros-industrial/universal_robot.git src/universal_robot
```
5. Install all package dependencies:
```console
sudo apt update -qq
```
```console
rosdep update
```
```console
rosdep install --from-paths src --ignore-src -r -y
```
6. Make the workspace:
```console
catkin_make
```
7. Source the workspace:
```console
source devel/setup.bash
```
8. Install remaining Python dependencies:
```console
pip install -r requirements.txt
```
---
## Demo You Could Run Now
The shell game requires an Intel RealSense camera and all of our other harware and parts to reproduce, so we're providing a simualted demo that works with most webcams using hand tracking to control the end-effector position of the UR3. To run it, use:
```console
roslaunch ur_moveit_config demo.launch
```
Open and source another terminal, then run:
```console
rosrun ur_scripts hand_tracking.py
```
You should then be able to reproduce results similar to those shown below.

[HandTrack.webm](https://user-images.githubusercontent.com/99771915/202353380-6606532c-889d-4f19-9824-c336622b92ac.webm)

---

## Full Shell Game Demo with Real UR3
The full demo uses shell_game.py, located within the src folder of the ur_scripts package.

https://user-images.githubusercontent.com/99771915/202349276-863eb793-f228-4302-8dc6-286ca12326e9.mov

---
