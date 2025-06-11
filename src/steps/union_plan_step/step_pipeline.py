import os
import sys
import json
from .union_plan import *
from .utils import get_mask_center, draw_result

class Union_step:
    def __init__(self,flat_path):
        self.flat_path = flat_path
        self.room_path = os.path.join(self.flat_path, 'by_rooms')
        self.segm_path = os.path.join(self.room_path, 'segms')
        self.result_dict = {}
    

    def run(self,viz=False):
        # выбираем комнату
        rooms_doors_dict = {}
        for item in os.listdir(self.room_path):
            with open(os.path.join(self.room_path, item, 'results', 'results.json'), 'r') as f:
                results = json.load(f)
            rooms_doors_dict[item] = len(results['door_points'])
        

        print(rooms_doors_dict)
        target_room = max(rooms_doors_dict, key=rooms_doors_dict.get)
        print(target_room)

        imgs = os.listdir(os.path.join(self.room_path, target_room))
        imgs = [item.split('.')[0] for item in imgs if '.' in item]
        with open(os.path.join(self.room_path, target_room, 'results', 'results.json'), 'r') as f:
                results = json.load(f)
        for item in imgs:
            print(item)
            doors_segms = [mask for mask in os.listdir(os.path.join(self.room_path, target_room,'segms')) if 'door' in mask and mask.split('_')[0]==item]
            if len(doors_segms) > 0:
                doors_segms = ['_'.join(mask.split('_')[1:]).split('.')[0] for mask in doors_segms]
                self.result_dict[item] = {door : {'center': get_mask_center(os.path.join(self.room_path, target_room,'segms', f'{item}_{door}.png')),
                                                  'door_pos':results[item]['door_pos'][i],
                                                  'door_points':results[item]['door_points'][i],} 
                                          for i,door in enumerate(doors_segms)}
        print(self.result_dict)
        # ищем матчи для дверей
        have_matches = find_matches(self.room_path, target_room)
        print(have_matches)

        self.result_dict = match_room_door(have_matches, self.flat_path, target_room, self.result_dict)
        
        self.result_dict = define_doors_rooms(self.result_dict)

        # итерируюсь по комнатам
        # смотрим на размер дверного проема
        # если он меньше или больше масштабируем размеры комнаты
        # обновляем координаты комнаты сравняв координаты дверного проема
        for img in list(self.result_dict.keys())[:2]:
            print(img)
            for door in list(self.result_dict[img].keys())[:2]:
                if 'is_exit' not in self.result_dict[img][door].keys():
                    match_room = self.result_dict[img][door]['match_room']
                    file_path = os.path.join(self.room_path, str(match_room), 'results','results.json')
                    with open(file_path, 'r') as file:
                        data = json.load(file)

                    scale_factor = distance(*self.result_dict[img][door]['door_points'])/distance(*data['door_points'][0])
                    # print(distance(*data['door_points'][0]))
                    # print(distance(*self.result_dict[img][door]['door_points']))
                    print('scale_factor',scale_factor)
                    # print()

                    final_points = data['final_points']
                    door_points = data['door_points'][0]  # Извлекаем список точек
                    window_points = data['window_points']

                    scaled_final_points = scale_final_points(final_points, scale_factor)

                    # Масштабируем двери и окна
                    new_door_positions = scale_position(door_points, scaled_final_points)
                    new_window_positions = None
                    if window_points is not None:
                        new_window_positions = scale_position(window_points, scaled_final_points)
                    
                    # обновляем координаты комнаты
                    self.result_dict[img][door]['connect_room_points'],window_pose = update_room_plan(scaled_final_points,new_door_positions, self.result_dict[img][door] ,results['final_points'],new_window_positions)
                    if window_pose is not None:
                        self.result_dict[img][door]['window_pos'] = window_pose

                    # построцесинг 
                    self.result_dict['ordered_points'] = results['final_points']
                    
                self.result_dict = remove_cross_walls(self.result_dict)
                self.result_dict = stick_walls(self.result_dict)
                self.result_dict = stick_out_doors(self.result_dict)
                self.result_dict = remove_cross_walls(self.result_dict)    

        # сохраняем результат
        for img in self.result_dict.keys():
            if img != 'ordered_points':
                for door in self.result_dict[img].keys():
                    for key in self.result_dict[img][door].keys():
                        if isinstance(self.result_dict[img][door][key], (np.ndarray,tuple)):
                            self.result_dict[img][door][key] = np.array(self.result_dict[img][door][key])
                            if len(self.result_dict[img][door][key].shape)==1:
                                self.result_dict[img][door][key] =[int(item) if isinstance(item, (np.int32, np.int64)) else float(item) for item in self.result_dict[img][door][key]]
                            else:
                                self.result_dict[img][door][key] = [list(item) for item in self.result_dict[img][door][key]]
            else:
                if isinstance(self.result_dict[img], (np.ndarray,tuple)):
                            self.result_dict[img] = np.array(self.result_dict[img])
                            if len(self.result_dict[img].shape)==1:
                                self.result_dict[img] =[int(item) if isinstance(item, (np.int32, np.int64)) else float(item) 
                                                        for item in self.result_dict[img]]
                            else:
                                self.result_dict[img] = [list(item) for item in self.result_dict[img]]
        mask = draw_result(self.result_dict)

        plt.imshow(mask)
        os.makedirs(os.path.join(self.flat_path, 'results'), exist_ok=True)
        cv2.imwrite(os.path.join(self.flat_path,'results','plan_test.png'), mask)

        json_file_path = os.path.join(self.flat_path, 'results','full_results.json')
        with open(json_file_path, 'w') as json_file:
            json.dump(self.result_dict, json_file) 




