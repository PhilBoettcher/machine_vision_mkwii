# Phillip Boettcher
# CS 455
# Final Project
import cv2 as cv
from time import time
from window_capture import WindowCapture
from vision import Vision
import bot_actions as bot_actions
from threading import Thread

# Setting up the window capturing class
window_name = 'Dolphin 2409 | JIT64 DC | OpenGL | HLE | Mario Kart Wii (RMCE01)'
wincap = WindowCapture(window_name)
# Initializing the machine vision handling class
vision = Vision()

# global variable (not great i know) used trigger seperate thread for the bot actions
bot_in_action = False
# bot actions
def bot_handler(centroid_info, image_width, position_slopes):
    if len(centroid_info) > 0:
        input = bot_actions.steer_player_with_curve(position_slopes, centroid_info)
        
    # resetting our global variable to false
    global bot_in_action
    bot_in_action is False

# used for tracking the fps
loop_time = time()
while(True):

    # getting a screenshot of desired window
    screenshot = wincap.get_screenshot()

    # running YOLO inference and drawing objects with OpenCV
    yolo_object_detection, centroid_info, position_slopes = vision.yolo_segmentation(screenshot)

    # starting a thread to handle the actions the bot takes
    if bot_in_action is False:
        bot_in_action is True
        image_width = screenshot.shape[1]
        t = Thread(target=bot_handler, args=(centroid_info, image_width, position_slopes))
        t.start()
            

    # using OpenCV to show the screenshot after YOLO inference and OpenCV drawing
    cv.imshow('Computer Vision', yolo_object_detection)

    # FPS printed in terminal
    print('FPS {}'.format(1 / (time() - loop_time)))
    loop_time = time()

    key = cv.waitKey(1)
    if key == ord('q'):
        cv.destroyAllWindows()
        break

print('Done.')
