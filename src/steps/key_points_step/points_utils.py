import os 
import numpy as np

def loadNPZ(npz_file,dir,treshold=0.5):    
    npz = np.load(os.path.join(dir,'dump_match_pairs',npz_file),allow_pickle=True)

    matching_indexes1 =  npz['matches'][npz['matches']>-1]
    matching_indexes2 = npz['matches'][npz['match_confidence']>=treshold]

    matching_indexes = [item for item in matching_indexes1 if item in matching_indexes2]
    point_set2 = npz['keypoints1'][matching_indexes]
    point_set1 = []
   
    for i in range(len(npz['matches'])):
     if npz['matches'][i] !=-1 and npz['match_confidence'][i]>=treshold:
         point_set1.append(npz['keypoints0'][i])

    point_set1 = np.array(point_set1)
    point_set2 = np.array(point_set2)
    
    return point_set1, point_set2