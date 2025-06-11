import os
import  cv2
import matplotlib.pyplot as plt
import numpy as np

from .preprocess import get_order
from .make_plan import make_room_plan
from .postprocess import make_plan_postprocess, change_len_wall,angle_between_line_and_plane
from .utils import  get_angle_btw_lines,get_original_lenght,get_angle_between_planes
from ..visual_utils.viz_plan import mask_from_points

class Make_2d_Plan_step:
    def __init__(self,flat_path, room,planes):
        self.flat_path = flat_path
        self.room = room
        self.room_path = os.path.join(self.flat_path, 'by_rooms', self.room)
        self.planes = planes
    
    def run(self, viz=False):
        # определяем порядок фото 
        if os.path.exists(os.path.join(self.room_path,'dump_match_pairs')):
            if len(os.listdir(os.path.join(self.room_path,'dump_match_pairs')))-1 != len([item for item in os.listdir(self.room_path) if '.' in item]):
                order = get_order(os.path.join(self.room_path,'dump_match_pairs'))
            else:
                print('Матчей меньше чем необходимо')
        else:
            order = []
        
        if len(order)==0:
            order = os.listdir(self.room_path)
            order = [item[:-4] for item in order if '.' in item]
            self.planes['order'] = order
        
        if len(order)==1 and len(self.planes[order[0]]['cross_point']) == 2: 
            points = self.planes[order[0]]['cross_point']
            points = [item[:2] for item in points]
            order=['1','2','3','4']
            points.append([points[1][0]+150,points[1][1]])
            points.append([points[1][0]+150,points[0][1]])
            points[1][0] = points[0][0]
            final_points = points
            res = mask_from_points(points,order)
            # строим план на основе одной фото
            
        elif len(order)==1 and len(self.planes[order[0]]['cross_point']) > 2:
            points = self.planes[order[0]]['cross_point']
            points = [item[:2] for item in points]
            order = [str(item+1) for item in range(len(self.planes[order[0]]['cross_point'])+2)]
            
            angle = get_angle_btw_lines([points[0],points[1]],[points[1],points[2]])
            if  angle<89 or angle>91:
                if angle>91:
                    lenght = get_original_lenght(points[0],points[1])
                    points[1][1] = points[2][1]
                    points[1][0] = points[0][0]
            
            angle = get_angle_btw_lines([points[-3],points[-2]],[points[-2],points[-1]])
            if  angle<89 or angle>91:
                if angle>91:

                    points[-2][1] = points[-3][1]
                    points[-2][0] = points[-1][0]
                    points[0][1] = points[2][1]-lenght

            points.append([points[0][0]-450,points[-1][1]])
            points.append([points[0][0]-450,points[0][1]])

            final_points = points
            res = mask_from_points(points,order)

        elif len(order)==2:
            all_points_num = sum([len(self.planes[item]['cross_point']) for item in order])
            if all_points_num >=4:
                if len(os.listdir(os.path.join(self.room_path,'dump_match_pairs')))==1:
                    angle = 180
                    stride = 150
                    theta = np.radians(angle)
                    target_room = order[0]
                    final_points = [point[:2] for point in self.planes[target_room]['cross_point']]
                    for point in  self.planes[order[1]]['cross_point']:
                        final_points.append(np.array([point[0] * np.cos(theta) - point[1] * np.sin(theta),point[0] * np.sin(theta) + point[1] * np.cos(theta)]) + stride)


                    x_fixed_value = self.planes[order[0]]['cross_point'][0][0]
                    coefficients = [1,0,0,-x_fixed_value] # коэф плоскости

                    N1 = coefficients[:-1]  
                    N2 = self.planes[order[0]]['floor'][:-1]

                    angle = get_angle_between_planes(N1, N2)
                    order = [str(i+1) for i in range(len(final_points))]
                    

                    final_points[2] = change_len_wall(final_points,1,2,angle *2)
                    final_points[3] = change_len_wall(final_points,0,3,angle *2)

                    final_points[2][1] = final_points[1][1]
                    final_points[3][1] = final_points[0][1]

                    final_points[1][0] = final_points[0][0]
                    final_points[3][0] = final_points[2][0]
                    res =  mask_from_points(final_points, order)
                    # self.planes['order'] = order

        elif len(order)==1 and len(self.planes[order[0]]['cross_point']) ==1:
            print('недостаточно данных для построения плана')
        else:       
            points = dict.fromkeys(order)
            for item in order:
                points[item] = [item[:-1] for item in self.planes[item]['cross_point']]

            res, final_points = make_room_plan(self.room_path, order,order[-2], points)
            res ,final_points = make_plan_postprocess(final_points,order,order[-2], self.planes)

            self.planes['order'] = order
        self.planes['final_points'] = final_points
        os.makedirs(os.path.join(self.room_path, 'results'),exist_ok=True)
        if res is not None:
            cv2.imwrite(os.path.join(self.room_path, 'results','plan.png'),res)

        if viz:
            plt.imshow(res)
            plt.show()
        
        return self.planes
