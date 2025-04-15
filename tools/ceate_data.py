import sys
import os

current_dir=os.path.dirname(os.path.abspath(__file__))
parent_dir=os.path.dirname(current_dir)
sys.path.append(parent_dir)
from pathlib import Path
import logging

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
                
            


if __name__=='__main__':
    root_path="./data/nuscenes"
    can_bus_root_path=""
    info_prefix=""
    version="v1.0-mini"
    dataset_name="nuscenes"
    out_dir="./data/nuscenes"
    nu=NUSCENES(root_path,can_bus_root_path,info_prefix,version,dataset_name,out_dir)
    nu.get_available_scenes()
        