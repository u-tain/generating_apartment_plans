import os
import cv2
import torch
import math
import numpy as np
import itertools
from pathlib import Path
import matplotlib.cm as cm
from dataclasses import dataclass, field
from itertools import combinations
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import shutil


from .points_utils import  loadNPZ
from .models.matching import Matching
from .models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)


@dataclass
class Super_Glue:
    flat_path: str
    sg_params: dict = field(init=False)
    imgs:list = None

    def __post_init__(self):
        self.sg_params = {
                            'resize': [640],
                            'superglue': 'indoor',
                            'max_keypoints': 1024,
                            'nms_radius': 4,
                            'resize_float': 'store_true',
                            'input_dir': self.flat_path,
                            'input_pairs': os.path.join (self.flat_path, 'dump_match_pairs','pairs.txt'),
                            'output_dir': os.path.join (self.flat_path, 'dump_match_pairs'),
                            'viz': 'store_true',
                            'fast_viz': False,
                            'keypoint_threshold': 0.05,
                            'match_threshold': 0.8,
                            'sinkhorn_iterations':20,
                            'viz_extension':'png',
                            'eval': False,
                            'cache': 'store_true',
                            'max_length': -1,
                            'show_keypoints':'store_true',
                            'opencv_display':'store_true',
                        }
    
    def preprocess_stage(self):
        print('начинаем препроцессинг')
        # получаем путь до изображений
        # составляем файл с парами
        if self.imgs is None:
            imgs = os.listdir(self.flat_path)
            imgs = [item for item in imgs if '.' in item]
        else:
            imgs = self.imgs
        rem = ['dump_match_pairs','pairs.txt', 'plan.jpg','url.txt','depth_imgs']
        os.makedirs(os.path.join(self.flat_path,'dump_match_pairs'),exist_ok=True)
        for item in rem:
            if item in imgs:
                imgs.remove(item)
        pairs = list(itertools.combinations(imgs, 2)) # получаем возможные комбинации

        with open(os.path.join(self.flat_path,'dump_match_pairs', 'pairs.txt'), 'w') as file:
            for i in range(len(pairs)):
                file.write(f"{pairs[i][0]} {pairs[i][1]}\n")
                file.write(f"{pairs[i][1]} {pairs[i][0]}\n")

        # изменяем размер всех изображений в удобный формат
        # пока рассматриваем только горизонтальные
        for item in imgs:
            img = cv2.imread(os.path.join(self.flat_path,item))
            img = cv2.resize(img, (640, 480))
            cv2.imwrite(os.path.join(self.flat_path,item),img)
        print('препроцессинг закончен')

        
    def prediction_stage(self):
        with open(self.sg_params['input_pairs'], 'r') as f:
            pairs = [l.split() for l in f.readlines()]

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        config = {
            'superpoint': {
                'nms_radius': self.sg_params['nms_radius'],
                'keypoint_threshold': self.sg_params['keypoint_threshold'],
                'max_keypoints': self.sg_params['max_keypoints']
            },
            'superglue': {
                'weights': self.sg_params['superglue'],
                'sinkhorn_iterations': self.sg_params['sinkhorn_iterations'],
                'match_threshold':self.sg_params['match_threshold'],
            }
        }

        matching = Matching(config).eval().to(device)

        # Create the output directories if they do not exist already.
        input_dir = Path(self.sg_params['input_dir'])
        # print('Looking for data in directory \"{}\"'.format(input_dir))
        output_dir = Path(self.sg_params['output_dir'])
        output_dir.mkdir(exist_ok=True, parents=True)
        
        if self.sg_params['viz']:
            print('результат будет записан в ',
                'папку \"{}\"'.format(output_dir))

        for i, pair in enumerate(pairs):
            name0, name1 = pair[:2]
            stem0, stem1 = Path(name0).stem, Path(name1).stem
            matches_path = output_dir / '{}_{}_matches.npz'.format(stem0, stem1)
            eval_path = output_dir / '{}_{}_evaluation.npz'.format(stem0, stem1)
            viz_path = output_dir / '{}_{}_matches.{}'.format(stem0, stem1, self.sg_params['viz_extension'])
            viz_eval_path = output_dir / \
                '{}_{}_evaluation.{}'.format(stem0, stem1, self.sg_params['viz_extension'])

            # Handle --cache logic.
            do_match = True
            do_eval = self.sg_params['eval']
            do_viz = self.sg_params['viz']
            do_viz_eval = self.sg_params['eval'] and self.sg_params['viz']
            if self.sg_params['cache']:
                if matches_path.exists():
                    try:
                        results = np.load(matches_path)
                    except:
                        raise IOError('Cannot load matches .npz file: %s' %
                                    matches_path)

                    kpts0, kpts1 = results['keypoints0'], results['keypoints1']
                    matches, conf = results['matches'], results['match_confidence']
                    do_match = False
                
                if self.sg_params['eval'] and eval_path.exists():
                    try:
                        results = np.load(eval_path)
                    except:
                        raise IOError('Cannot load eval .npz file: %s' % eval_path)
                    err_R, err_t = results['error_R'], results['error_t']
                    precision = results['precision']
                    matching_score = results['matching_score']
                    num_correct = results['num_correct']
                    epi_errs = results['epipolar_errors']
                    do_eval = False
                if self.sg_params['viz'] and viz_path.exists():
                    do_viz = False
                if self.sg_params['viz'] and self.sg_params['eval'] and viz_eval_path.exists():
                    do_viz_eval = False


            # If a rotation integer is provided (e.g. from EXIF data), use it:
            if len(pair) >= 5:
                rot0, rot1 = int(pair[2]), int(pair[3])
            else:
                rot0, rot1 = 0, 0

            # Load the image pair.
            image0, inp0, scales0 = read_image(
                input_dir / name0, device, self.sg_params['resize'], rot0, self.sg_params['resize_float'])
            image1, inp1, scales1 = read_image(
                input_dir / name1, device, self.sg_params['resize'], rot1, self.sg_params['resize_float'])
            if image0 is None or image1 is None:
                print('Problem reading image pair: {} {}'.format(
                    input_dir/name0, input_dir/name1))
                exit(1)
            # timer.update('load_image')

            if do_match:
                # Perform the matching.
                pred = matching({'image0': inp0, 'image1': inp1})
                pred = {k: v[0].detach().cpu().numpy() for k, v in pred.items()}
                kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
                matches, conf = pred['matches0'], pred['matching_scores0']
                # timer.update('matcher')

                # Write the matches to disk.
                out_matches = {'keypoints0': kpts0, 'keypoints1': kpts1,
                            'matches': matches, 'match_confidence': conf}
                np.savez(str(matches_path), **out_matches)

            # Keep the matching keypoints.
            valid = matches > -1
            mkpts0 = kpts0[valid]
            mkpts1 = kpts1[matches[valid]]
            mconf = conf[valid]

            if do_eval:
                # Estimate the pose and compute the pose error.
                assert len(pair) == 38, 'Pair does not have ground truth info'
                K0 = np.array(pair[4:13]).astype(float).reshape(3, 3)
                K1 = np.array(pair[13:22]).astype(float).reshape(3, 3)
                T_0to1 = np.array(pair[22:]).astype(float).reshape(4, 4)

                # Scale the intrinsics to resized image.
                K0 = scale_intrinsics(K0, scales0)
                K1 = scale_intrinsics(K1, scales1)

                # Update the intrinsics + extrinsics if EXIF rotation was found.
                if rot0 != 0 or rot1 != 0:
                    cam0_T_w = np.eye(4)
                    cam1_T_w = T_0to1
                    if rot0 != 0:
                        K0 = rotate_intrinsics(K0, image0.shape, rot0)
                        cam0_T_w = rotate_pose_inplane(cam0_T_w, rot0)
                    if rot1 != 0:
                        K1 = rotate_intrinsics(K1, image1.shape, rot1)
                        cam1_T_w = rotate_pose_inplane(cam1_T_w, rot1)
                    cam1_T_cam0 = cam1_T_w @ np.linalg.inv(cam0_T_w)
                    T_0to1 = cam1_T_cam0

                epi_errs = compute_epipolar_error(mkpts0, mkpts1, T_0to1, K0, K1)
                correct = epi_errs < 5e-4
                num_correct = np.sum(correct)
                precision = np.mean(correct) if len(correct) > 0 else 0
                matching_score = num_correct / len(kpts0) if len(kpts0) > 0 else 0

                thresh = 1.  # In pixels relative to resized image size.
                ret = estimate_pose(mkpts0, mkpts1, K0, K1, thresh)
                if ret is None:
                    err_t, err_R = np.inf, np.inf
                else:
                    R, t, inliers = ret
                    err_t, err_R = compute_pose_error(T_0to1, R, t)

                # Write the evaluation results to disk.
                out_eval = {'error_t': err_t,
                            'error_R': err_R,
                            'precision': precision,
                            'matching_score': matching_score,
                            'num_correct': num_correct,
                            'epipolar_errors': epi_errs}
                np.savez(str(eval_path), **out_eval)
                # timer.update('eval')

            if do_viz:
                # Visualize the matches.
                color = cm.jet(mconf)
                text = [
                    'SuperGlue',
                    'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
                    'Matches: {}'.format(len(mkpts0)),
                ]
                if rot0 != 0 or rot1 != 0:
                    text.append('Rotation: {}:{}'.format(rot0, rot1))

                # Display extra parameter info.
                k_thresh = matching.superpoint.config['keypoint_threshold']
                m_thresh = matching.superglue.config['match_threshold']
                small_text = [
                    'Keypoint Threshold: {:.4f}'.format(k_thresh),
                    'Match Threshold: {:.2f}'.format(m_thresh),
                    'Image Pair: {}:{}'.format(stem0, stem1),
                ]

                make_matching_plot(
                    image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
                    text, viz_path,self.sg_params['show_keypoints'],
                    self.sg_params['fast_viz'], self.sg_params['opencv_display'], 'Matches', small_text)

                # timer.update('viz_match')

            if do_viz_eval:
                # Visualize the evaluation results for the image pair.
                color = np.clip((epi_errs - 0) / (1e-3 - 0), 0, 1)
                color = error_colormap(1 - color)
                deg, delta = ' deg', 'Delta '
                if not self.sg_params['fast_viz']:
                    deg, delta = '°', '$\\Delta$'
                e_t = 'FAIL' if np.isinf(err_t) else '{:.1f}{}'.format(err_t, deg)
                e_R = 'FAIL' if np.isinf(err_R) else '{:.1f}{}'.format(err_R, deg)
                text = [
                    'SuperGlue',
                    '{}R: {}'.format(delta, e_R), '{}t: {}'.format(delta, e_t),
                    'inliers: {}/{}'.format(num_correct, (matches > -1).sum()),
                ]
                if rot0 != 0 or rot1 != 0:
                    text.append('Rotation: {}:{}'.format(rot0, rot1))

                # Display extra parameter info (only works with --fast_viz).
                k_thresh = matching.superpoint.config['keypoint_threshold']
                m_thresh = matching.superglue.config['match_threshold']
                small_text = [
                    'Keypoint Threshold: {:.4f}'.format(k_thresh),
                    'Match Threshold: {:.2f}'.format(m_thresh),
                    'Image Pair: {}:{}'.format(stem0, stem1),
                ]

                make_matching_plot(
                    image0, image1, kpts0, kpts1, mkpts0,
                    mkpts1, color, text, viz_eval_path,
                    self.sg_params['show_keypoints'], self['sg_params'].fast_viz,
                    self.sg_params['opencv_display'], 'Relative Pose', small_text)


            if self.sg_params['eval']:
                # Collate the results into a final table and print to terminal.
                pose_errors = []
                precisions = []
                matching_scores = []
                for pair in pairs:
                    name0, name1 = pair[:2]
                    stem0, stem1 = Path(name0).stem, Path(name1).stem
                    eval_path = output_dir / \
                        '{}_{}_evaluation.npz'.format(stem0, stem1)
                    results = np.load(eval_path)
                    pose_error = np.maximum(results['error_t'], results['error_R'])
                    pose_errors.append(pose_error)
                    precisions.append(results['precision'])
                    matching_scores.append(results['matching_score'])
                thresholds = [5, 10, 20]
                aucs = pose_auc(pose_errors, thresholds)
                aucs = [100.*yy for yy in aucs]
                prec = 100.*np.mean(precisions)
                ms = 100.*np.mean(matching_scores)
                print('Evaluation Results (mean over {} pairs):'.format(len(pairs)))
                print('AUC@5\t AUC@10\t AUC@20\t Prec\t MScore\t')
                print('{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t'.format(
                    aucs[0], aucs[1], aucs[2], prec, ms))

        pass

    def postprocess_stage(self):
        print('начинаем постпроцессинг')
        # удаляем все те пары между которыми 0 совпадений
        res_list = [item[:-4] for item in os.listdir(os.path.join(self.flat_path,'dump_match_pairs'))]
        res_list = list(set(res_list))
        res_list.pop(res_list.index('pairs'))
        for item in res_list:
            set1,set2 = loadNPZ(item+'.npz',self.flat_path)
            if len(set1) < 4 and len(set2) < 4:
                os.remove(os.path.join(self.flat_path,'dump_match_pairs',item+'.npz'))
                os.remove(os.path.join(self.flat_path,'dump_match_pairs',item+'.png'))

        # TODO добавить отсеивание по наименьшему среднему рассточни. между точками
        res_list = []
        for item in os.listdir(os.path.join(self.flat_path,'dump_match_pairs')):
            if 'pairs' not in item:
                res_list.append(item.split('_')[0])
                res_list.append(item.split('_')[1])
        res_list = list(set(res_list))
        
        pairs = list(combinations(res_list, 2))


        def get_mean_dist(set1,set2):
            def euclidean_distance(point1, point2):
                return math.sqrt((point1[0] - point2[0]+640) ** 2 + (point1[1] - point2[1]) ** 2)
            distances = []
            for p1, p2 in zip(set1, set2):
                distance = euclidean_distance(p1, p2)
                distances.append(distance)
            return sum(distances) / len(distances)
        
        for pair in pairs:
            if os.path.exists(os.path.join(self.flat_path,'dump_match_pairs',f'{pair[0]}_{pair[1]}_matches.png')) and os.path.exists(os.path.join(self.flat_path,'dump_match_pairs',f'{pair[1]}_{pair[0]}_matches.png')):
                set1,set2 = loadNPZ(os.path.join(self.flat_path,'dump_match_pairs',f'{pair[0]}_{pair[1]}_matches.npz'), self.flat_path)
                mean1 = get_mean_dist(set1,set2)

                set3,set4 = loadNPZ(os.path.join(self.flat_path,'dump_match_pairs',f'{pair[1]}_{pair[0]}_matches.npz'), self.flat_path)
                mean2 = get_mean_dist(set3,set4)

                if mean1 < mean2:
                    os.remove(os.path.join(self.flat_path,'dump_match_pairs',f'{pair[0]}_{pair[1]}_matches.png'))
                    os.remove(os.path.join(self.flat_path,'dump_match_pairs',f'{pair[0]}_{pair[1]}_matches.npz'))
                else: 
                    os.remove(os.path.join(self.flat_path,'dump_match_pairs',f'{pair[1]}_{pair[0]}_matches.png'))
                    os.remove(os.path.join(self.flat_path,'dump_match_pairs',f'{pair[1]}_{pair[0]}_matches.npz'))

        pass

    def viz_result(self):
        os.makedirs(os.path.join(self.flat_path,'graphs'),exist_ok=True)

        files = os.listdir(os.path.join(self.flat_path,'dump_match_pairs'))
        imgs = [os.path.join(self.flat_path,'dump_match_pairs',item) for item in files if '.png' in item]
        if len(files)==3:
            shutil.copy(os.path.join(self.flat_path,'dump_match_pairs',imgs[0]), 
                                 os.path.join(self.flat_path,'graphs','key_points.png'))
        elif len(files)==1:
                 print('Не найдено матчей между фото')
        else:
            num_images = len(imgs)
            cols = 1  # Количество столбцов для отображения
            rows = (num_images // cols) + (num_images % cols > 0)  if num_images !=1 else 1
            if rows !=1:
                # Создание фигуры для отображения изображений
                fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))

                # Плоский список осей для удобного перебора
                axes = axes.flatten() if num_images > 1 else [axes]

                for ax, png_file in zip(axes, imgs):
                    img = mpimg.imread(png_file)  # Чтение изображения
                    ax.imshow(img)
                    ax.axis('off')  # Убрать оси
                    ax.set_title(os.path.basename(png_file.split('/')[-1]))  # Заголовок с названием файла

                # Удалить лишние оси
                for ax in axes[num_images:]:
                    ax.remove()

                plt.tight_layout()
                os.makedirs(os.path.join(self.flat_path,'graphs'),exist_ok=True)
                plt.savefig(os.path.join(self.flat_path,'graphs','key_points.png'), bbox_inches='tight')
                plt.close(fig)

                fig,ax=None,None
            else: 
                img = mpimg.imread(png_file)  # Чтение изображения
                plt.imshow(img)
                plt.axis('off')
                os.makedirs(os.path.join(self.flat_path,'graphs'),exist_ok=True)
                plt.savefig(os.path.join(self.flat_path,'graphs','key_points.png'), bbox_inches='tight')
                plt.close()
