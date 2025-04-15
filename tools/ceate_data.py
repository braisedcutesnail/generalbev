import sys
import os

current_dir=os.path.dirname(os.path.abspath(__file__))
parent_dir=os.path.dirname(current_dir)
sys.path.append(parent_dir)





class Data():
    def __init__(self):
        pass

class Nuscenes(Data):
    def __init__(self,root_path,can_bus_root_path,info_prefix,version,dataset_name,out_dir,max_sweeps=10):
        super(Nuscenes,self).__init__()
        from nuscenes.nuscenes import NuScenes
        from nuscenes.can_bus.can_bus_api import NuScenesCanBus
        nusc=Nuscenes(version=version,dataroot=root_path,verbose=True)
        nusc_can_bus=NuScenesCanBus(dataroot=can_bus_root_path)
        