# Phillip Boettcher
# CS 455
# Final Project
import cv2 as cv
import numpy as np
from ultralytics import YOLO
import logging


# initializeing roboflow YOLOv8 model
pt_file = 'YOLO_tools\\best-seg.pt'
model = YOLO(pt_file)
logging.getLogger('ultralytics').setLevel(logging.ERROR)

class Vision():
    # for object annotation
    color_map = {
        "offroad": (0, 0, 255),  
        "road": (0, 255, 0),  
        "player": (255, 0, 0) 
    }

    def __init__(self):
        pass

    def get_contour_centroid(self, contour):
        M = cv.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M["m00"])
            cy = int(M['m01'] / M["m00"])
            return (cx, cy)
        else:
            return None

    def yolo_segmentation(self, image_feed):
        # draws segmentations from YOLOv8 model
        results = model(image_feed, conf=0.6)
        image_with_masks = self.draw_segmentation_masks(image_feed, results, self.color_map)
        return image_with_masks
    
    def draw_position_lines(self, image, centroid_info):
        player_x, player_y = centroid_info['player']
        road_x, road_y = centroid_info['road']

        if 'look_ahead_point' in centroid_info:
            position_slopes = {}
            try:
                look_x, look_y = centroid_info['look_ahead_point']
                cv.line(image, (road_x, road_y), (look_x, look_y), (0,255,255, 4))
                look_slope = (look_y - road_y) / (look_x - road_x)
                P0_b = road_y - (look_slope * road_x)  # Intercept of road line
                P0_y = look_slope*player_x + P0_b

                # Perpendicular line from player to road line
                m_perp = -1 / look_slope
                P1_b = player_y - (m_perp * player_x)  # Intercept of player to road line


                x_intersect = (P1_b - P0_b) / (look_slope + (1/look_slope))
                y_intersect = look_slope * x_intersect + P0_b

                cv.line(image, (player_x, player_y), (int(x_intersect), int(y_intersect)), (255,255,0), 3)
                cv.circle(image, (int(x_intersect), int(y_intersect)), 5, (255,255,0), -1)
            except ValueError:
                pass
            position_slopes.update({'look_slope': look_slope, 'player_slope': m_perp})

        return image, position_slopes

    def draw_segmentation_masks(self, image, results, color_map):
        original_height, original_width, _ = image.shape 
        masks = results[0].masks  
        boxes = results[0].boxes 
        names = results[0].names 

        positional_info = {}
        centroid_info = {}


        for i, (mask, box, class_id, confidence) in enumerate(zip(masks.data, boxes.xyxy, boxes.cls, boxes.conf)):

            binary_mask = mask.cpu().numpy().astype(np.uint8) * 255
            binary_mask = cv.resize(binary_mask, (original_width, original_height), interpolation=cv.INTER_NEAREST)

            color = color_map.get(names[int(class_id)], (255, 255, 255))
            colored_mask = np.zeros_like(image, dtype=np.uint8)
            for i in range(3):
                colored_mask[:, :, i] = binary_mask * color[i]

            image = cv.addWeighted(image, 1, colored_mask, 0.5, 0)

            contours, _ = cv.findContours(binary_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            cv.drawContours(image, contours, -1, color, thickness=2)

            x1, y1, x2, y2 = map(int, box) 
            label = f"{names[int(class_id)]} {confidence:.2f}"
            text_size = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            text_x = x1 
            text_y = max(y1 - 10, 0)

            cv.rectangle(image, (text_x, text_y - text_size[1]), (text_x + text_size[0], text_y + 2), color, -1)
            cv.putText(image, label, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

            class_name = names[int(class_id)]
            positional_info[class_name] = {
                'binary_mask': binary_mask,
                'contours': contours
            }

        centroid_info_update = {}
        for class_name, info in positional_info.items():
            if class_name == 'road' and info['contours']:
                # Combine all points from all contours to find the global top line
                all_points = np.concatenate(info['contours'], axis=0)  # shape: N x 1 x 2
                all_points = all_points.reshape(-1, 2)
                min_y = np.min(all_points[:, 1])
                # Find all points at this y
                top_points = all_points[all_points[:, 1] == min_y]
                if len(top_points) > 0: 
                    min_x = np.min(top_points[:,0])
                    max_x = np.max(top_points[:,0])
                    midpoint_x = (min_x + max_x) // 2
                    cv.circle(image, (midpoint_x, min_y), 5, (0, 255, 255), -1)
            for contour in info['contours']:
                centroid = self.get_contour_centroid(contour)
                if centroid:
                    cv.circle(image, centroid, 5, (0, 255, 255), -1)
                    cv.putText(image, class_name, (centroid[0], centroid[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                    centroid_info_update.update({class_name: centroid})
                    if class_name == 'road':
                        centroid_info_update.update({'look_ahead_point': (midpoint_x, min_y)})

        centroid_info.update(centroid_info_update)

        image, position_slopes = self.draw_position_lines(image, centroid_info)

        return image, centroid_info, position_slopes
    
