# Phillip Boettcher
# CS 455
# Final Project
from pynput.keyboard import Controller

# Initialize the controller
keyboard = Controller()

# Function to handle key presses and releases
def handle_key_press(key, condition):
    if condition:
            keyboard.press(key)
            print(f"Pressed {key}")
            key_press = True
    else:
            keyboard.release(key)
            print(f"Released {key}")
            key_press = False
    
    return key_press

def steer_player_with_curve(position_slopes, centroid_info):
    look_ahead_slope = position_slopes['look_slope']
    player_x, _ = centroid_info['player']
    road_x, _ = centroid_info['road']

    delta_distance = player_x - road_x

    dead_zone = 2 
    if abs(delta_distance) < dead_zone and abs(look_ahead_slope) > 10:
        return
    
    slope_threshold = 3
    distance_threshold = 20

    #key presses based on the slopes and distance
    # steering should be adjusted based on road slope
    handle_key_press('a', look_ahead_slope < slope_threshold and look_ahead_slope > 0)
    handle_key_press('d', look_ahead_slope > -slope_threshold and look_ahead_slope < 0)


    # based on the player's distance from the road line
    handle_key_press('a', delta_distance > distance_threshold)
    handle_key_press('d', delta_distance < -distance_threshold)
