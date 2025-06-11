import cv2
import numpy as np

def make_normal(depth_path, mask_path):
    depth = np.load(depth_path)
    mask_path = cv2.imread(mask_path)[:,:,-1]
    depth *= mask_path
    
    height, width = depth.shape
    depth = cv2.resize(depth, (width * 3, height * 3), interpolation=cv2.INTER_CUBIC)

    zx = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=15)
    zy = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=15)
    
    normal = np.dstack((-zx, -zy, np.ones_like(depth)))
    n = np.linalg.norm(normal, axis=2)
    normal[:, :, 0] /= n
    normal[:, :, 1] /= n
    normal[:, :, 2] /= n

    normal += 1
    normal /= 2

    normal=cv2.resize(normal, (width , height ), interpolation=cv2.INTER_CUBIC)

    return normal