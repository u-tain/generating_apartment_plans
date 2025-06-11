import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from .depth_model import DepthMap


class Depth_Map_step:
    def __init__(self,flat_path, room):
        self.flat_path = flat_path
        self.room = room
        self.room_path = os.path.join(self.flat_path, 'by_rooms', self.room)
        self.depth_path = os.path.join(self.room_path,'depth')


    def run(self, viz=False):
        imgs = os.listdir(self.room_path)
        imgs = [item[:-4] for item in imgs if '.' in item]
        if not os.path.exists(self.depth_path):
            m = DepthMap(self.room_path)
            m.start_predict()
        if viz:
            imgs = os.listdir(self.room_path)
            imgs = [item[:-4] for item in imgs if '.' in item]

            if len(imgs)!=1:
                num_images = len(imgs)*2
                cols = 2  # Количество столбцов для отображения
                rows = (num_images // cols) + (num_images % cols > 0)  # Число строк

                # Создание фигуры для отображения изображений
                fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))

                # Плоский список осей для удобного перебора
                for i,img in enumerate(imgs):
                    axes[i][0].imshow(cv2.cvtColor(cv2.imread(os.path.join(self.room_path,img+'.jpg')), cv2.COLOR_BGR2RGB))
                    axes[i][1].imshow(np.load(os.path.join(self.depth_path,img+'_depth.npy')))
                    axes[i][0].axis('off') 
                    axes[i][1].axis('off') 
                    axes[i][0].set_title(os.path.basename(img))  # Заголовок с названием файла

                axes = axes.flatten() if num_images > 1 else [axes]
                for ax in axes[num_images:]:
                    ax.remove()
            else:
                num_images = len(imgs)*2
                cols = 2  # Количество столбцов для отображения
                rows = 1
                fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))

                axes[0].imshow(cv2.cvtColor(cv2.imread(os.path.join(self.room_path,imgs[0]+'.jpg')), cv2.COLOR_BGR2RGB))
                axes[1].imshow(np.load(os.path.join(self.depth_path,imgs[0]+'_depth.npy')))
                axes[0].axis('off') 
                axes[1].axis('off') 
                axes[0].set_title(os.path.basename(imgs[0]))
                
                plt.show()