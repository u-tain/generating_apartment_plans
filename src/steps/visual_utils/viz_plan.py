import cv2
import numpy as np
from ..add_door_window_step.utils import is_point_in_rectangle,calculate_distance

def draw_circle(center, text, mask,step=0):
    cv2.circle(mask, (int(center[0])+step,int(center[1])+step),5,255,-1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_color = (255, 0, 0)  
    thickness = 2

    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Calculate the text position to center it within the circle
    text_x = int(center[0])+step - text_width // 2+ 15
    text_y = int(center[1])+step + text_height // 2 

    # Add text to the image
    cv2.putText(mask, text, (text_x, text_y), font, font_scale, font_color, thickness)
    return mask

def mask_from_points(points, order,
                     door_points=None,
                     window_points=None):
    # отрисовываем точки
    min_x = np.min(np.array(points)[:, 0])
    min_y = np.min(np.array(points)[:, 1])
    max_x= np.max(np.array(points)[:, 0])
    max_y = np.max(np.array(points)[:, 1])

    # Если минимальные значения отрицательные, сдвигаем все точки
    if min_x < 0 or min_y < 0:
        shift_x = abs(min_x) if min_x < 0 else 0
        shift_y = abs(min_y) if min_y < 0 else 0
    else:
        shift_x = 0
        shift_y = 0
    stride = int(max(shift_x,shift_y)) + 50

    h = w = int(max(max_x,max_y)) +stride + 100


    mask = np.zeros((h,w),dtype=np.uint8)
    
    for i in range(len(points)):
        draw_circle((int(points[i][0]),int(points[i][1])),order[i],mask,stride)
    
    for i in range(len(points)):
        if i!= len(points)-1:
            cv2.line(mask,(int(points[i][0]+stride),int(points[i][1])+stride),(int(points[i+1][0]+stride),int(points[i+1][1])+stride),color=255,thickness=10)
        else:
            cv2.line(mask,(int(points[0][0]+stride),int(points[0][1])+stride),(int(points[i][0]+stride),int(points[i][1])+stride),color=255,thickness=10)
    
    if door_points:
        for door_points_1,door_points_2 in door_points:
        
            cv2.line(mask,(int(door_points_1[0]+stride),int(door_points_1[1])+stride),(int(door_points_2[0]+stride),int(door_points_2[1])+stride),color=0,thickness=10)

            door_points_1 = [int(point+stride) for point in door_points_1]
            door_points_2 = [int(point+stride) for point in door_points_2]
            length = calculate_distance(door_points_1,door_points_2)
            if door_points_1[0]==door_points_2[0]:
                angle = -90  
                angle_rad = np.radians(angle)

                end_x = int(door_points_1[0]+ length)
                end_y = int(door_points_1[1] )

                if is_point_in_rectangle((end_x, end_y),*points[-3:],points[0]):
                    line_end = (end_x, end_y)
                    angle=-angle
                else:
                    line_end = (int(door_points_1[0] + length), int(door_points_1[1]))
            else:
                angle = 90
                angle_rad = np.radians(angle)

                # Координаты конечной точки линии, выходящей из P1 под углом 60 градусов
                end_x = int(door_points_1[0] + length * np.cos(angle_rad))
                end_y = int(door_points_1[1] - length * np.sin(angle_rad))  # Вырезаем по Y, чтобы угол был внутри
                # Проверка, что конечная точка в пределах маски
                if is_point_in_rectangle((end_x, end_y),*points[-3:],points[0]):
                    line_end = (end_x, end_y)
                    angle=-angle
                else:
                    line_end = (int(door_points_1[0]), int(door_points_1[1] + length))
            stopping_point = line_end
            # cv2.line(mask, door_points_1, stopping_point, color=255, thickness=2)
            # cv2.ellipse(mask, door_points_1, (int(length), int(length)), 0, 0, angle, 255, thickness=1)

    if window_points:
        window_points_1,window_points_2=window_points

        window_points_1 = [int(point+stride) for point in window_points_1]
        window_points_2 = [int(point+stride) for point in window_points_2]
        cv2.line(mask, window_points_1, window_points_2, color=0, thickness=10)
        window_points_1[1] -=5
        window_points_2[1] -=5

        cv2.line(mask, window_points_1, window_points_2, color=255, thickness=2)

        window_points_1[1] +=10
        window_points_2[1] +=10

        cv2.line(mask, window_points_1, window_points_2, color=255, thickness=2)
        print(window_points_1,window_points_2)
    return mask