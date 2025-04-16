import sys
import os
import numpy as np
current_dir=os.path.dirname(os.path.abspath(__file__))
parent_dir=os.path.dirname(current_dir)
sys.path.append(parent_dir)
from pathlib import Path
import logging
from pyquaternion import Quaternion
# 配置日志级别和输出格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



class Data():
    def __init__(self):
        pass
from nuscenes.nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
class NUSCENES(Data):
    def __init__(self,root_path,can_bus_root_path,info_prefix,version,dataset_name,out_dir,max_sweeps=10):
        super(NUSCENES,self).__init__()
        self.nusc=NuScenes(version=version,dataroot=root_path,verbose=True)
        self.nusc_can_bus=None
        if can_bus_root_path:
            self.nusc_can_bus=NuScenesCanBus(dataroot=can_bus_root_path)
        from nuscenes.utils import splits
        available_vers=['v1.0-trainval','v1.0-test','v1.0-mini']
        assert version in available_vers
        if version=='v1.0-trainval':
            train_scenes=splits.train
            val_scenes=splits.val
        elif version=='v1.0-test':
            train_scenes=splits.test
            val_scenes=[]
        elif version=='v1.0-mini':
            train_scenes=splits.mini_train
            val_scenes=splits.mini_val
        else:
            raise ValueError('unknown')
        available_scenes=self.get_available_scenes()
        available_scene_name=[s['name'] for s in available_scenes]
        # 获取训练和验证
        train_scenes=list(filter(lambda x:x in available_scene_name,train_scenes))
        train_scenes=set([available_scenes[available_scene_name.index(s)]['token'] for s in train_scenes])
        val_scenes=list(filter(lambda x: x in available_scene_name,val_scenes))
        val_scenes=set([available_scenes[available_scene_name.index(s)]['token'] for s in val_scenes])

        test='test' in version
        if test:
            logging.info(f"test scene:{len(train_scenes)}")
        else:
            logging.info(f"train scene:{len(train_scenes)},val scene:{len(val_scenes)}")
        train_nusc_infos,val_nusc_infos=self._fill_trainval_infos(train_scenes,val_scenes,test,max_sweeps=max_sweeps)

        logging.info(type(self.nusc.sample))


    def get_available_scenes(self): #检查路径是否合法，lidar的路径，实际上内部原理黑盒
        available_scenes=[]
        for scene in self.nusc.scene: #针对scene级别的数据
            scene_token=scene["token"]
            scene_rec=self.nusc.get('scene',scene_token) #可见内部结构是每个scene有一个token，同时在nusc也就是整体数据结构出，有一个scene字典，字典键是token，内容是这个scene的内容
            sample_rec=self.nusc.get('sample',scene_rec['first_sample_token']) #在sample字典中，查找键为上一个scene字典中查询结果的first_sample_token
            sd_rec=self.nusc.get('sample_data',sample_rec['data']['LIDAR_TOP']) #
            has_more_frames=True
            scene_not_exist=False
            while has_more_frames:
                lidar_path,boxes,_=self.nusc.get_sample_data(sd_rec['token']) #内部会去拼接data_root路径
                lidar_path=str(lidar_path)
                if os.getcwd() in lidar_path:
                    lidar_path=lidar_path.split(f'{os.getcwd()}')[-1]
                if not isinstance(lidar_path,str) and not isinstance(lidar_path,Path): #直到找不到
                    logging.warning(f"illegal lidar_path:{lidar_path}")
                    scene_not_exist=True
                    break
                else:
                    break
            if scene_not_exist:
                continue
            available_scenes.append(scene)
        logging.info(f"exist scene num:{len(available_scenes)}")
        return available_scenes
                
    def _fill_trainval_infos(self,
                         train_scenes,
                         val_scenes,
                         test=False,
                         max_sweeps=10):
        """Generate the train/val infos from the raw data.

        Args:
            nusc (:obj:`NuScenes`): Dataset class in the nuScenes dataset.
            train_scenes (list[str]): Basic information of training scenes.
            val_scenes (list[str]): Basic information of validation scenes.
            test (bool): Whether use the test mode. In the test mode, no
                annotations can be accessed. Default: False.
            max_sweeps (int): Max number of sweeps. Default: 10.

        Returns:
            tuple[list[dict]]: Information of training set and validation set
                that will be saved to the info file.
        """
        train_nusc_infos = []
        val_nusc_infos = []
        frame_idx = 0
        for sample in self.nusc.sample:
            lidar_token = sample['data']['LIDAR_TOP']
            sd_rec = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            cs_record = self.nusc.get('calibrated_sensor',
                                sd_rec['calibrated_sensor_token'])
            pose_record = self.nusc.get('ego_pose', sd_rec['ego_pose_token'])
            lidar_path, boxes, _ = self.nusc.get_sample_data(lidar_token)

            
            can_bus = _get_can_bus_info(nusc, nusc_can_bus, sample)
            ##
            info = {
                'lidar_path': lidar_path,
                'token': sample['token'],
                'prev': sample['prev'],
                'next': sample['next'],
                'can_bus': can_bus,
                'frame_idx': frame_idx,  # temporal related info
                'sweeps': [],
                'cams': dict(),
                'scene_token': sample['scene_token'],  # temporal related info
                'lidar2ego_translation': cs_record['translation'],
                'lidar2ego_rotation': cs_record['rotation'],
                'ego2global_translation': pose_record['translation'],
                'ego2global_rotation': pose_record['rotation'],
                'timestamp': sample['timestamp'],
            }

            if sample['next'] == '':
                frame_idx = 0
            else:
                frame_idx += 1

            l2e_r = info['lidar2ego_rotation']
            l2e_t = info['lidar2ego_translation']
            e2g_r = info['ego2global_rotation']
            e2g_t = info['ego2global_translation']
            l2e_r_mat = Quaternion(l2e_r).rotation_matrix
            e2g_r_mat = Quaternion(e2g_r).rotation_matrix

            # obtain 6 image's information per frame
            camera_types = [
                'CAM_FRONT',
                'CAM_FRONT_RIGHT',
                'CAM_FRONT_LEFT',
                'CAM_BACK',
                'CAM_BACK_LEFT',
                'CAM_BACK_RIGHT',
            ]
            for cam in camera_types:
                cam_token = sample['data'][cam]
                cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)
                cam_info = obtain_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat,
                                            e2g_t, e2g_r_mat, cam)
                cam_info.update(cam_intrinsic=cam_intrinsic)
                info['cams'].update({cam: cam_info})

            # obtain sweeps for a single key-frame
            sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            sweeps = []
            while len(sweeps) < max_sweeps:
                if not sd_rec['prev'] == '':
                    sweep = obtain_sensor2top(nusc, sd_rec['prev'], l2e_t,
                                            l2e_r_mat, e2g_t, e2g_r_mat, 'lidar')
                    sweeps.append(sweep)
                    sd_rec = nusc.get('sample_data', sd_rec['prev'])
                else:
                    break
            info['sweeps'] = sweeps
            # obtain annotation
            if not test:
                annotations = [
                    nusc.get('sample_annotation', token)
                    for token in sample['anns']
                ]
                locs = np.array([b.center for b in boxes]).reshape(-1, 3)
                dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
                rots = np.array([b.orientation.yaw_pitch_roll[0]
                                for b in boxes]).reshape(-1, 1)
                velocity = np.array(
                    [nusc.box_velocity(token)[:2] for token in sample['anns']])
                valid_flag = np.array(
                    [(anno['num_lidar_pts'] + anno['num_radar_pts']) > 0
                    for anno in annotations],
                    dtype=bool).reshape(-1)
                # convert velo from global to lidar
                for i in range(len(boxes)):
                    velo = np.array([*velocity[i], 0.0])
                    velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                        l2e_r_mat).T
                    velocity[i] = velo[:2]

                names = [b.name for b in boxes]
                for i in range(len(names)):
                    if names[i] in NuScenesDataset.NameMapping:
                        names[i] = NuScenesDataset.NameMapping[names[i]]
                names = np.array(names)
                # we need to convert rot to SECOND format.
                gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)
                assert len(gt_boxes) == len(
                    annotations), f'{len(gt_boxes)}, {len(annotations)}'
                info['gt_boxes'] = gt_boxes
                info['gt_names'] = names
                info['gt_velocity'] = velocity.reshape(-1, 2)
                info['num_lidar_pts'] = np.array(
                    [a['num_lidar_pts'] for a in annotations])
                info['num_radar_pts'] = np.array(
                    [a['num_radar_pts'] for a in annotations])
                info['valid_flag'] = valid_flag

            if sample['scene_token'] in train_scenes:
                train_nusc_infos.append(info)
            else:
                val_nusc_infos.append(info)

        return train_nusc_infos, val_nusc_infos     


if __name__=='__main__':
    root_path="./data/nuscenes"
    can_bus_root_path=""
    info_prefix=""
    version="v1.0-mini"
    dataset_name="nuscenes"
    out_dir="./data/nuscenes"
    nu=NUSCENES(root_path,can_bus_root_path,info_prefix,version,dataset_name,out_dir)
    nu.get_available_scenes()
        