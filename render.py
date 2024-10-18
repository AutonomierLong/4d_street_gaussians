import torch 
import os
import json
from tqdm import tqdm
from lib.models.street_gaussian_model import StreetGaussianModel 
from lib.models.street_gaussian_renderer import StreetGaussianRenderer
from lib.datasets.dataset import Dataset
from lib.models.scene import Scene
from lib.utils.general_utils import safe_state
from lib.config import cfg
from lib.visualizers.base_visualizer import BaseVisualizer as Visualizer
from lib.visualizers.street_gaussian_visualizer import StreetGaussianVisualizer
import time
import numpy as np
from lib.utils.graphics_utils import fov2focal, getProjectionMatrix, getWorld2View2, getProjectionMatrixK

def render_sets():
    cfg.render.save_image = True
    cfg.render.save_video = False

    with torch.no_grad():
        dataset = Dataset()
        gaussians = StreetGaussianModel(dataset.scene_info.metadata)
        scene = Scene(gaussians=gaussians, dataset=dataset)
        renderer = StreetGaussianRenderer()

        times = []
        if not cfg.eval.skip_train:
            save_dir = os.path.join(cfg.model_path, 'train', "ours_{}".format(scene.loaded_iter))
            visualizer = Visualizer(save_dir)
            cameras = scene.getTrainCameras()    

            for idx, camera in enumerate(tqdm(cameras, desc="Rendering Training View")):
                
                torch.cuda.synchronize()
                start_time = time.time()
                result = renderer.render(camera, gaussians)
                
                torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) * 1000)
                
                visualizer.visualize(result, camera)

        if not cfg.eval.skip_test:
            save_dir = os.path.join(cfg.model_path, 'test', "ours_{}".format(scene.loaded_iter))
            visualizer = Visualizer(save_dir)
            cameras =  scene.getTestCameras()
            for idx, camera in enumerate(tqdm(cameras, desc="Rendering Testing View")):
                
                torch.cuda.synchronize()
                start_time = time.time()
                
                result = renderer.render(camera, gaussians)
                                
                torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) * 1000)
                
                visualizer.visualize(result, camera)
        
        print(times)        
        print('average rendering time: ', sum(times[1:]) / len(times[1:]))
                
def render_trajectory():
    cfg.render.save_image = False
    cfg.render.save_video = True
    
    with torch.no_grad():
        dataset = Dataset()        
        gaussians = StreetGaussianModel(dataset.scene_info.metadata)

        scene = Scene(gaussians=gaussians, dataset=dataset)
        renderer = StreetGaussianRenderer()
        
        save_dir = os.path.join(cfg.model_path, 'trajectory', "ours_{}".format(scene.loaded_iter))
        visualizer = StreetGaussianVisualizer(save_dir)
        
        train_cameras = scene.getTrainCameras()
        test_cameras = scene.getTestCameras()
        cameras = train_cameras + test_cameras
        cameras = list(sorted(cameras, key=lambda x: x.id))

        for idx, camera in enumerate(tqdm(cameras, desc="Rendering Trajectory")):
            result = renderer.render_all(camera, gaussians)  
            visualizer.visualize(result, camera)

        visualizer.summarize()

def render_sets_offset():
    cfg.render.save_image = True
    cfg.render.save_video = True

    with torch.no_grad():
        dataset = Dataset()
        gaussians = StreetGaussianModel(dataset.scene_info.metadata)
        scene = Scene(gaussians=gaussians, dataset=dataset)
        renderer = StreetGaussianRenderer()

        times = []
        if not cfg.eval.skip_train:
            save_dir = os.path.join(cfg.model_path, 'train', "offset_ours_{}".format(scene.loaded_iter))
            visualizer = Visualizer(save_dir)
            cameras = scene.getTrainCameras()
            import ipdb
            ipdb.set_trace()

            delta = np.array([1.0, 0.0, 0.0])
                
            for idx, camera in enumerate(tqdm(cameras, desc="Rendering Training View")):
                camera.T += delta
                # change attributes associated with R, T
                camera.world_view_transform = torch.tensor(getWorld2View2(camera.R, camera.T, camera.trans, camera.scale)).transpose(0, 1).cuda()
                camera.full_proj_transform = (camera.world_view_transform.unsqueeze(0).bmm(camera.projection_matrix.unsqueeze(0))).squeeze(0)
                camera.camera_center = camera.world_view_transform.inverse()[3, :3]
                
                torch.cuda.synchronize()
                start_time = time.time()
                result = renderer.render(camera, gaussians)
                
                torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) * 1000)
                
                visualizer.visualize(result, camera)

        if not cfg.eval.skip_test:
            save_dir = os.path.join(cfg.model_path, 'test', "ours_{}".format(scene.loaded_iter))
            visualizer = Visualizer(save_dir)
            cameras =  scene.getTestCameras()
            for idx, camera in enumerate(tqdm(cameras, desc="Rendering Testing View")):
                
                torch.cuda.synchronize()
                start_time = time.time()
                
                result = renderer.render(camera, gaussians)
                                
                torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) * 1000)
                
                visualizer.visualize(result, camera)
        
        print(times)        
        print('average rendering time: ', sum(times[1:]) / len(times[1:]))


def render_sets_trajectory_offset():
    cfg.render.save_image = False
    cfg.render.save_video = True
    
    with torch.no_grad():
        dataset = Dataset()        
        gaussians = StreetGaussianModel(dataset.scene_info.metadata)

        scene = Scene(gaussians=gaussians, dataset=dataset)
        renderer = StreetGaussianRenderer()
        
        save_dir = os.path.join(cfg.model_path, 'trajectory', "offset_ours_{}".format(scene.loaded_iter))
        visualizer = StreetGaussianVisualizer(save_dir)
        
        train_cameras = scene.getTrainCameras()
        test_cameras = scene.getTestCameras()
        cameras = train_cameras + test_cameras
        cameras = list(sorted(cameras, key=lambda x: x.id))

        delta = np.array([0.5, 0.0, 0.0])

        for idx, camera in enumerate(tqdm(cameras, desc="Rendering Trajectory")):
            camera.T += delta
            # change attributes associated with R, T
            camera.world_view_transform = torch.tensor(getWorld2View2(camera.R, camera.T, camera.trans, camera.scale)).transpose(0, 1).cuda()
            camera.full_proj_transform = (camera.world_view_transform.unsqueeze(0).bmm(camera.projection_matrix.unsqueeze(0))).squeeze(0)
            camera.camera_center = camera.world_view_transform.inverse()[3, :3]
            result = renderer.render_all(camera, gaussians)  
            visualizer.visualize(result, camera)

        visualizer.summarize()
            
if __name__ == "__main__":
    print("Rendering " + cfg.model_path)
    safe_state(cfg.eval.quiet)
    
    if cfg.mode == 'evaluate':
        render_sets()
    elif cfg.mode == 'trajectory':
        render_trajectory()
    elif cfg.mode == 'offset':
        render_sets_offset()
    elif cfg.mode == 'offset_trajectory':
        render_sets_trajectory_offset()
    else:
        raise NotImplementedError()
