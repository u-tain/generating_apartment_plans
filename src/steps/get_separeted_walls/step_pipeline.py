import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

from .preprocessing import make_normal
from .separete import make_clastering, separate_masks


class Get_Separeted_walls_step:
    def __init__(self,flat_path, room):
        self.flat_path = flat_path
        self.room = room
        self.room_path = os.path.join(self.flat_path, 'by_rooms', self.room)
        self.segm_path = os.path.join(self.room_path, 'segms')
        self.depth_path = os.path.join(self.room_path, 'depth')

    def run(self, viz=False):
        imgs = os.listdir(self.room_path)
        imgs = [item[:-4] for item in imgs if '.' in item]
        # num_walls = num_walls_flat[int(self.flat_path.split('_')[-1])][str(self.room)]

        os.makedirs(os.path.join(self.room_path,'separeted_walls'),exist_ok=True)
        os.makedirs(os.path.join(self.room_path,'normal'),exist_ok=True)
        os.makedirs(os.path.join(self.room_path,'wide_contours'),exist_ok=True)

        for i,item in enumerate(imgs):
            normal = make_normal(os.path.join(self.depth_path, f'{item}_depth.npy'),
                                 os.path.join(self.segm_path, f'{item}_wall.png'))
            
            kernel = np.ones((15,5))
            flat = int(self.flat_path.split('_')[-1])
            # if 'custom_const' in num_walls_flat[flat].keys():
            #     if item in num_walls_flat[flat]['custom_const'].keys():
            #         kernel = np.ones(num_walls_flat[flat]['custom_const'][item])

            # wide_contours = get_instances(normal,
            #                                 os.path.join(self.segm_path, f'{item}_wall.png'),
            #                                 kernel) 
            
            # walls = get_separated_walls(wide_contours,
            #                                   os.path.join(self.segm_path, f'{item}_wall.png'),
            #                                   num_walls[item])

            walls = cv2.cvtColor(cv2.imread(os.path.join(self.segm_path, f'{item}_wall.png')),cv2.COLOR_BGR2RGB)[:,:,-1]
            masks = make_clastering(normal,walls)
            walls = separate_masks(walls,masks)
            
            for wall in range(len(walls)):
                cv2.imwrite(os.path.join(self.room_path,'separeted_walls',item+f'_wall_{wall+1}.png'),walls[wall])
                # cv2.imwrite(os.path.join(self.room_path,'wide_contours',item+'_wide_contours.png'),wide_contours)

            np.save(os.path.join(self.room_path,'normal',item+'_normal.npy'),normal)
            # cv2.imwrite(os.path.join(self.room_path,'normal',item+'_normal.png'),normal)

            if viz:
                cols = 3+len(walls) # Количество столбцов для отображения
                rows = 1  # Число строк
                fig, axes = plt.subplots(rows, cols, figsize=(15, 2 * rows))

                axes[0].imshow(cv2.cvtColor(cv2.imread(os.path.join(self.room_path,imgs[i]+'.jpg')), cv2.COLOR_BGR2RGB))
                axes[1].imshow(normal)
                # axes[2].imshow(wide_contours*cv2.imread(os.path.join(self.segm_path,f'{item}_wall.png'))[:,:,-1])
                for wall in range(len(walls)):
                    axes[3+wall].imshow(walls[wall])
                axes[cols-1].imshow(cv2.cvtColor(cv2.imread(os.path.join(self.segm_path,f'{item}_floor.png')), cv2.COLOR_BGR2RGB)[:,:,-1])

                axes[0].axis('off') 
                axes[1].axis('off') 
                axes[2].axis('off') 
                for wall in range(len(walls)):
                    axes[3+wall].axis('off') 
                axes[cols-1].axis('off') 
                fig.show()
