import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from .preprocessing import detect_position
from .utils import get_distance,get_object_points
from ..segmentation_step.postprocess import get_box
from ..visual_utils.viz_plan import mask_from_points

class Add_door_window_step:
    def __init__(self,flat_path, room,planes):
        self.flat_path = flat_path
        self.room = room
        self.room_path = os.path.join(self.flat_path, 'by_rooms', self.room)
        self.planes = planes
    
    def run(self, viz):
        # определяем где дверь
        # TODO добавить информацию как далеко она от угла

        if len(self.planes['order'])==1:
            image_w_door=True
            image_w_window = None
            self.planes['order'] = self.planes['order']*4
            center = np.array(self.planes['final_points'][-3])-np.array(self.planes['final_points'][-2])

            door_size = 160
            door_point1,door_point2 = get_object_points(*self.planes['final_points'][-2:],door_size,int(center[0]//2-door_size//2)-60)

        else:
            door_points = []
            doors = [item  for item in self.planes['order'] if  self.planes[item]['has_door']==True and self.planes[item]['has_window']==False]
            print(doors)
            pos = []
            for imdoor in doors:
                image_w_door = imdoor
                path = os.path.join(self.room_path,'segms')
                doors_segms_path = [os.path.join(path,item) for item in os.listdir(path) if 'door' in item]
                print(doors_segms_path)
                print(image_w_door)
            for i,door in enumerate(doors_segms_path):
                print(os.path.basename(door).split('_')[0])
                imname = os.path.basename(door).split('_')[0]
                if len(self.planes['order'])!=len(self.planes['final_points']):
                
                    idx = self.planes['order'].index(imname)
                    idx = idx + len(self.planes[imname]['cross_point'])
                    order = [str(i) for i in range(len(self.planes['final_points']))]
                else:
                    order = self.planes['order']
                self.planes, pos = detect_position(os.path.basename(door).split('_')[0], self.planes, door, f'door_{i}',idx)
                # pos = [self.planes[imname][item] for item in  [key for key in self.planes[imname].keys() if 'door_' in key]]
                # pos = np.unique(pos)
                print(pos)
                idx = [order.index(item) for item in pos.split('_')]
                point1 = np.array(self.planes['final_points'][idx[0]]) 
                point2 = np.array(self.planes['final_points'][idx[1]])
                mask = cv2.cvtColor(cv2.imread(door),cv2.COLOR_BGR2RGB)

                box = get_box(mask[:,:,-1])
                distance,closest_distance = get_distance(box,self.planes['final_points'][self.planes['order'].index(image_w_door)])
                door_point1,door_point2 = get_object_points(point1,point2,distance-30,closest_distance//10)                    
                door_points.append([list(door_point1),list(door_point2)])

            # определяем где окна
            image_w_window = None
            windows = [item  for item in self.planes['order'] if  self.planes[item]['has_window']==True]
            if len(windows) > 0:
                image_w_window = windows[0]
                path = os.path.join(self.room_path,'segms',f'{image_w_window}_window.png')
                self.planes,pos = detect_position(image_w_window, self.planes, path,'window')

                pos = np.unique([self.planes[item]['window_pos'] for item in windows[:1]])
                if len(pos) ==1:
                    pos = pos[0]
                    idx = [self.planes['order'].index(item) for item in pos.split('_')]

                    point1 = np.array(self.planes['final_points'][idx[0]]) 
                    point2 = np.array(self.planes['final_points'][idx[1]])
                    
                    mask = cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB)
                    box = get_box(mask[:,:,-1])
                    distance,closest_distance = get_distance(box,self.planes['final_points'][self.planes['order'].index(image_w_window)])

                    window_point1, window_point2 = get_object_points(point1,point2,distance,closest_distance)


        # наносим на план
        if image_w_door:
            door_points = door_points
        else:
            door_points=None

        if image_w_window:
            window_points=[window_point1, window_point2]
        else:
            window_points= None
        
        print([list([list(item) for item in door_points])])


        # point4,pointl,pointt,pointr = self.planes['final_points']
        mask = mask_from_points(self.planes['final_points'],
                        self.planes['order'] if len(self.planes['order']) == len(self.planes['final_points']) else [str(item) for item in range(len(self.planes['final_points']))],
                        door_points=door_points,
                        window_points=window_points)
        cv2.imwrite(os.path.join(self.room_path, 'results','plan_full.png'),mask)
        json_file_path = os.path.join(self.room_path, 'results','results.json')
        with open(json_file_path, 'w') as json_file:
            json.dump({'final_points':[list(item) if not isinstance(item,list) else item for item in  self.planes['final_points'] ],
                       'door_points':[list([list(item) for item in door_points])],
                       'window_points':list([list(item) for item in window_points]) if window_points is not None else None,
                       'order': self.planes['order']}, json_file)
        
        plt.imshow(mask)

        


