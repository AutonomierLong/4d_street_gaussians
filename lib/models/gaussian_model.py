import torch
import torch.nn as nn
import numpy as np
import os
from simple_knn._C import distCUDA2
from lib.config import cfg
from lib.utils.general_utils import inverse_sigmoid, get_expon_lr_func, quaternion_to_matrix, build_rotation_4d, build_scaling_rotation_4d
from lib.utils.sh_utils import RGB2SH
from lib.utils.graphics_utils import BasicPointCloud
from lib.utils.general_utils import strip_symmetric, build_scaling_rotation
from lib.utils.system_utils import mkdir_p
from lib.utils.data_utils import to_cuda
from plyfile import PlyData, PlyElement
from lib.utils.camera_utils import Camera


class GaussianModel(nn.Module):
    def __init__(self, model_name='background', num_classes=1, gaussian_dim: int = 3, time_duration: list = [0, 10], rot_4d: bool = False):
        super().__init__()
        cfg_model = cfg.model.gaussian
        self.model_name = model_name
        
        # semantic
        self.num_classes = num_classes  
        self.semantic_mode = cfg_model.get('semantic_mode', 'logits')
        assert self.semantic_mode in ['logits', 'probabilities']
        
        # spherical harmonics
        default_max_sh_degree = cfg_model.get('sh_degree')
        if self.model_name == 'background':
            self.max_sh_degree = cfg_model.get('sh_degree_background', default_max_sh_degree)
        elif self.model_name == 'sky':
            self.max_sh_degree = cfg_model.get('sh_degree_sky', default_max_sh_degree)
        else:
            self.max_sh_degree = cfg_model.get('sh_degree_obj', default_max_sh_degree)
        self.active_sh_degree = self.max_sh_degree
        
        # original gaussian initialization
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._semantic = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        self.gaussian_dim = gaussian_dim
        self._t = torch.empty(0)
        self._scaling_t = torch.empty(0)
        self.time_duration = time_duration
        self.rot_4d = rot_4d
        self._rotation_r = torch.empty(0)
        # self.force_sh_3d = force_sh_3d
        self.t_gradient_accum = torch.empty(0)
        if self.rot_4d: # or self.force_sh_3d:
            assert self.gaussian_dim == 4
        self.env_map = torch.empty(0)

        self.active_sh_degree_t = 0
        self.max_sh_degree_t = self.max_sh_degree

        self.setup_functions()
    
    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[..., 0] = fused_color

        if self.gaussian_dim == 4:
            if pcd.time is None:
                fused_times = (torch.rand(fused_point_cloud.shape[0], 1, device="cuda") * 1.2 - 0.1) * (self.time_duration[1] - self.time_duration[0]) + self.time_duration[0]
            else:
                fused_times = torch.from_numpy(pcd.time).cuda().float()

        print(f"Number of points at initialisation for {self.model_name}: ", fused_point_cloud.shape[0])
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        if self.gaussian_dim == 4:
            # dist_t = torch.clamp_min(distCUDA2(fused_times.repeat(1,3)), 1e-10)[...,None]
            dist_t = torch.zeros_like(fused_times, device="cuda") + (self.time_duration[1] - self.time_duration[0]) / 5
            scales_t = torch.log(torch.sqrt(dist_t))
            if self.rot_4d:
                rots_r = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
                rots_r[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        semamtics = torch.zeros((fused_point_cloud.shape[0], self.num_classes), dtype=torch.float, device="cuda")
        
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._semantic = nn.Parameter(semamtics.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        
        if self.gaussian_dim == 4:
            self._t = nn.Parameter(fused_times.requires_grad_(True))
            self._scaling_t = nn.Parameter(scales_t.requires_grad_(True))
            if self.rot_4d:
                self._rotation_r = nn.Parameter(rots_r.requires_grad_(True))

    # TODO: Need to be made compatible with 4dgs.
    def make_ply(self):
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        semantic = self._semantic.detach().cpu().numpy()
        if self.gaussian_dim == 4:
            t = self._t.detach().cpu().numpy()
            rotation_r = self._rotation_r.detach().cpu().numpy()
            scaling_t = self._scaling_t.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        if self.gaussian_dim == 4:
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, semantic, t, rotation_r, scaling_t), axis=1)
        else:
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, semantic), axis=1)
        elements[:] = list(map(tuple, attributes))
        
        return elements

    # TODO: Need to be made compatible with 4dgs.
    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))
        elements = self.make_ply()
        elements = PlyElement.describe(elements, 'vertex')
        PlyData([elements]).write(path)
        
    # TODO: Need to be made compatible with 4dgs.
    def load_ply(self, path=None, input_ply=None):
        if path is None:
            plydata = input_ply
        else:
            plydata = PlyData.read(path)
            plydata = plydata.elements[0]

        xyz = np.stack((np.asarray(plydata["x"]),
                        np.asarray(plydata["y"]),
                        np.asarray(plydata["z"])),  axis=1)
        opacities = np.asarray(plydata["opacity"])[..., np.newaxis]
 
        base_f_names = [p.name for p in plydata.properties if p.name.startswith("f_dc_")]
        base_f_names = sorted(base_f_names, key = lambda x: int(x.split('_')[-1]))
        extra_f_names = [p.name for p in plydata.properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        features_dc = np.zeros((xyz.shape[0], len(base_f_names)))
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(base_f_names):
            features_dc[:, idx] = np.asarray(plydata[attr_name])
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata[attr_name])
        features_dc = features_dc.reshape(features_dc.shape[0], 3, -1)
        features_extra = features_extra.reshape(features_extra.shape[0], 3, -1)
       
        scale_names = [p.name for p in plydata.properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata[attr_name])

        rot_names = [p.name for p in plydata.properties if p.name.startswith("rot_") and not p.name.startswith("rot_r")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata[attr_name])

        semantic_names = [p.name for p in plydata.properties if p.name.startswith("semantic_")]
        semantic_names = sorted(semantic_names, key = lambda x: int(x.split('_')[-1]))
        semantic = np.zeros((xyz.shape[0], len(semantic_names)))
        for idx, attr_name in enumerate(semantic_names):
            semantic[:, idx] = np.asarray(plydata[attr_name])

        t_names = [p.name for p in plydata.properties if p.name.startswith("t_")]
        t_names = sorted(t_names, key = lambda x: int(x.split('_')[-1]))
        t = np.zeros((xyz.shape[0], len(t_names)))
        for idx, attr_name in enumerate(t_names):
            t[:, idx] = np.asarray(plydata[attr_name])

        rotation_r_names = [p.name for p in plydata.properties if p.name.startswith("t_")]
        rotation_r_names = sorted(rotation_r_names, key = lambda x: int(x.split('_')[-1]))
        rotation_r = np.zeros((xyz.shape[0], len(rotation_r_names)))
        for idx, attr_name in enumerate(rotation_r_names):
            rotation_r[:, idx] = np.asarray(plydata[attr_name])

        scaling_t_names = [p.name for p in plydata.properties if p.name.startswith("t_")]
        scaling_t_names = sorted(scaling_t_names, key = lambda x: int(x.split('_')[-1]))
        scaling_t = np.zeros((xyz.shape[0], len(scaling_t_names)))
        for idx, attr_name in enumerate(scaling_t_names):
            scaling_t[:, idx] = np.asarray(plydata[attr_name])
 
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._semantic = nn.Parameter(torch.tensor(semantic, dtype=torch.float, device="cuda").requires_grad_(True))
        self._t = nn.Parameter(torch.tensor(t, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation_r = nn.Parameter(torch.tensor(rotation_r, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling_t = nn.Parameter(torch.tensor(scaling_t, dtype=torch.float, device="cuda").requires_grad_(True))
        self.active_sh_degree = self.max_sh_degree

        if gaussian_dim ==4:
            self.active_sh_degree_t = self.max_sh_degree


            
    def load_state_dict(self, state_dict):  
        self._xyz = state_dict['xyz']  
        self._features_dc = state_dict['feature_dc']
        self._features_rest = state_dict['feature_rest']
        self._scaling = state_dict['scaling']
        self._rotation = state_dict['rotation']
        self._opacity = state_dict['opacity']
        self._semantic = state_dict['semantic']
        self._t = state_dict['t']
        self._rotation_r = state_dict['rotation_r']
        self._scaling_t = state_dict['scaling_t']
        
        if cfg.mode == 'train':
            self.training_setup()
            if 'spatial_lr_scale' in state_dict:
                self.spatial_lr_scale = state_dict['spatial_lr_scale'] 
            if 'denom' in state_dict:
                self.denom = state_dict['denom'] 
            if 'max_radii2D' in state_dict:
                self.max_radii2D = state_dict['max_radii2D'] 
            if 'xyz_gradient_accum' in state_dict:
                self.xyz_gradient_accum = state_dict['xyz_gradient_accum']
            if 't_gradient_accum' in state_dict:
                self.t_gradient_accum = state_dict['t_gradient_accum']
            if 'active_sh_degree' in state_dict:
                self.active_sh_degree = state_dict['active_sh_degree']
            if 'optimizer' in state_dict:
                self.optimizer.load_state_dict(state_dict['optimizer'])

        
    def state_dict(self, is_final=False):
        state_dict = {
            'xyz': self._xyz,
            'feature_dc': self._features_dc,
            'feature_rest': self._features_rest,
            'scaling': self._scaling,
            'rotation': self._rotation,
            'opacity': self._opacity,
            'semantic': self._semantic,
            't': self._t,
            'rotation_r': self._rotation_r,
            'scaling_t': self._scaling_t
        }
        
        if not is_final:
            state_dict_extra = {
                'spatial_lr_scale': self.spatial_lr_scale,
                'denom': self.denom,
                'max_radii2D': self.max_radii2D,
                'xyz_gradient_accum': self.xyz_gradient_accum,
                't_gradient_accum': self.xyz_gradient_accum,
                'active_sh_degree': self.active_sh_degree,
                'optimizer': self.optimizer.state_dict(),
            }
            
            state_dict.update(state_dict_extra)
        
        return state_dict
    
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        def build_covariance_from_scaling_rotation_4d(scaling, scaling_modifier, rotation_l, rotation_r, dt=0.0):
            L = build_scaling_rotation_4d(scaling_modifier * scaling, rotation_l, rotation_r)
            actual_covariance = L @ L.transpose(1, 2)
            cov_11 = actual_covariance[:,:3,:3]
            cov_12 = actual_covariance[:,0:3,3:4]
            cov_t = actual_covariance[:,3:4,3:4]
            current_covariance = cov_11 - cov_12 @ cov_12.transpose(1, 2) / cov_t
            symm = strip_symmetric(current_covariance)
            if dt.shape[1] > 1:
                mean_offset = (cov_12.squeeze(-1) / cov_t.squeeze(-1))[:, None, :] * dt[..., None]
                mean_offset = mean_offset[..., None]  # [num_pts, num_time, 3, 1]
            else:
                mean_offset = cov_12.squeeze(-1) / cov_t.squeeze(-1) * dt
            return symm, mean_offset.squeeze(-1)
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        
        # self.covariance_activation = build_covariance_from_scaling_rotation
        if not self.rot_4d:
            self.covariance_activation = build_covariance_from_scaling_rotation
        else:
            self.covariance_activation = build_covariance_from_scaling_rotation_4d

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_scaling_t(self):
        return self.scaling_activation(self._scaling_t)

    @property
    def get_scaling_xyzt(self):
        return self.scaling_activation(torch.cat([self._scaling, self._scaling_t], dim = 1))
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_rotation_r(self):
        return self.rotation_activation(self._rotation_r)
    
    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_t(self):
        return self._t

    @property
    def get_xyzt(self):
        return torch.cat([self._xyz, self._t], dim = 1)
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_semantic(self):
        if self.semantic_mode == 'logits':
            return self._semantic
        elif self.semantic_mode == 'probabilities':
            return torch.nn.functional.softmax(self._semantic, dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_max_sh_channels(self):
        if self.gaussian_dim == 3 or self.force_sh_3d:
            return (self.max_sh_degree+1)**2
        elif self.gaussian_dim == 4 and self.max_sh_degree_t == 0:
            return sh_channels_4d[self.max_sh_degree]
        elif self.gaussian_dim == 4 and self.max_sh_degree_t > 0:
            return (self.max_sh_degree+1)**2 * (self.max_sh_degree_t + 1)
    
    def get_cov_t(self, scaling_modifier = 1):
        if self.rot_4d:
            L = build_scaling_rotation_4d(scaling_modifier * self.get_scaling_xyzt, self._rotation, self._rotation_r)
            actual_covariance = L @ L.transpose(1, 2)
            return actual_covariance[:,3,3].unsqueeze(1)
        else:
            return self.get_scaling_t * scaling_modifier

    def get_marginal_t(self, timestamp, scaling_modifier = 1): # Standard
        sigma = self.get_cov_t(scaling_modifier)
        return torch.exp(-0.5*(self.get_t-timestamp)**2/sigma) # / torch.sqrt(2*torch.pi*sigma)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def get_current_covariance_and_mean_offset(self, scaling_modifier = 1, timestamp = 0.0):
        try:
            return self.covariance_activation(self.get_scaling_xyzt, scaling_modifier, 
                                                                self._rotation, 
                                                                self._rotation_r,
                                                                dt = timestamp - self.get_t)
        except:
            import ipdb; 
            ipdb.set_trace()
    
    # TODO: May need to modify this function.
    def get_normals(self, camera: Camera):
        scales, rotations = self.get_scaling, self.get_rotation    
        rotations_mat = quaternion_to_matrix(rotations)    
        min_scales = torch.argmin(scales, dim=-1)
        indices = torch.arange(min_scales.shape[0])
        normals = rotations_mat[indices, :, min_scales]

        # points from gaussian to camera
        dir_pp = (self.get_xyz - camera.camera_center.repeat(self._xyz.shape[0], 1))
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True) # (N, 3)
        dotprod = torch.sum(-dir_pp_normalized * normals, dim=1, keepdim=True) # (N, 1)
        normals = torch.where(dotprod >= 0, normals, -normals) 

        return normals
        
    def scale_flatten_loss(self):
        scales = self.get_scaling
        sorted_scales = torch.sort(scales, dim=1, descending=False).values
        s1, s2, s3 = sorted_scales[:, 0], sorted_scales[:, 1], sorted_scales[:, 2]
        s1 = torch.clamp(s1, 0, 30)
        s2 = torch.clamp(s2, 1e-5, 30)
        s3 = torch.clamp(s3, 1e-5, 30)
        scale_flatten_loss = torch.abs(s1).mean()
        scale_flatten_loss += torch.abs(s2 / s3 + s3 / s2 - 2.).mean()
        return scale_flatten_loss
        
    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
        elif self.max_sh_degree_t and self.active_sh_degree_t < self.max_sh_degree_t:
            self.active_sh_degree_t += 1

    def training_setup(self):
        args = cfg.optim
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 2), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.active_sh_degree = 0
                
        l = [
            {'params': [self._xyz], 'lr': args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': args.rotation_lr, "name": "rotation"},
            {'params': [self._semantic], 'lr': args.semantic_lr, "name": "semantic"},
        ]

        if self.gaussian_dim == 4: # TODO: tune time_lr_scale
            if args.position_t_lr_init < 0:
                args.position_t_lr_init = args.position_lr_init
            self.t_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            l.append({'params': [self._t], 'lr': args.position_t_lr_init * self.spatial_lr_scale, "name": "t"})
            l.append({'params': [self._scaling_t], 'lr': args.scaling_lr, "name": "scaling_t"})
            if self.rot_4d:
                l.append({'params': [self._rotation_r], 'lr': args.rotation_lr, "name": "rotation_r"})
        
        self.percent_dense = args.percent_dense
        self.percent_big_ws = args.percent_big_ws
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=args.position_lr_init * self.spatial_lr_scale,
            lr_final=args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=args.position_lr_delay_mult,
            max_steps=args.position_lr_max_steps
        )
        
        self.densify_and_prune_list = ['xyz, f_dc, f_rest, opacity, scaling, rotation, semantic, t, scaling_t, rotation_r']
        self.scalar_dict = dict()
        self.tensor_dict = dict()  
        
    def update_optimizer(self):
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        for i in range(self._semantic.shape[1]):
            l.append('semantic_{}'.format(i))
        
        if self.gaussian_dim == 4:
            for i in range(self._t.shape[1]):
                l.append('t_{}'.format(i))
            for i in range(self._rotation_r.shape[1]):
                l.append('rot_r_{}'.format(i))
            for i in range(self._scaling_t.shape[1]):
                l.append('scaling_t_{}'.format(i))

        return l


    def reset_optimizer(self, tensors_dict):
        optimizable_tensors = {}

        name_list = tensors_dict.keys()
        for group in self.optimizer.param_groups:
            if group['name'] in name_list:
                reset_tensor = tensors_dict[group['name']]

                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(reset_tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(reset_tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(reset_tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_optimizer(self, mask, prune_list = None):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:                
            if group['name'] not in prune_list:
                continue
        
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors
    
    def cat_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        name_list = tensors_dict.keys()
        for group in self.optimizer.param_groups:
            if group['name'] not in name_list:
                continue
            
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors
    
    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        d = {'opacity': opacities_new}
        optimizable_tensors = self.reset_optimizer(d)
        self._opacity = optimizable_tensors["opacity"]

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self.prune_optimizer(valid_points_mask, 
            prune_list = ['xyz', 'f_dc', 'f_rest', 'opacity', 'scaling', 'rotation', 'semantic', 't', 'scaling_t', 'rotation_r'])

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._semantic = optimizable_tensors["semantic"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        if self.gaussian_dim == 4:
            self._t = optimizable_tensors['t']
            self._scaling_t = optimizable_tensors['scaling_t']
            if self.rot_4d:
                self._rotation_r = optimizable_tensors['rotation_r']
            self.t_gradient_accum = self.t_gradient_accum[valid_points_mask]


    def densification_postfix(self, tensors_dict):
        optimizable_tensors = self.cat_optimizer(tensors_dict)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._semantic = optimizable_tensors["semantic"]

        if self.gaussian_dim == 4:
            self._t = optimizable_tensors['t']
            self._scaling_t = optimizable_tensors['scaling_t']
            if self.rot_4d:
                self._rotation_r = optimizable_tensors['rotation_r']
            self.t_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        
        cat_points_num = self.get_xyz.shape[0] - self.xyz_gradient_accum.shape[0]
        self.xyz_gradient_accum = torch.cat([self.xyz_gradient_accum, torch.zeros(cat_points_num, 2).cuda()], dim=0)
        self.denom = torch.cat([self.denom, torch.zeros(cat_points_num, 1).cuda()], dim=0)
        self.max_radii2D = torch.cat([self.max_radii2D, torch.zeros(cat_points_num).cuda()], dim=0)

        # self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, grads_t=None, grad_t_threshold=None, N=2):
        n_init_points = self.get_xyz.shape[0]
                
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        padded_extent = torch.zeros((n_init_points), device="cuda")
        padded_extent[:grads.shape[0]] = scene_extent
        
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > self.percent_dense * padded_extent)

        self.scalar_dict['points_split'] = selected_pts_mask.sum().item()
        print(f'Number of points to split: {selected_pts_mask.sum()}')

        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_semantic = self._semantic[selected_pts_mask].repeat(N, 1)
        
        if not self.rot_4d:
            stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
            means = torch.zeros((stds.size(0), 3), device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = quaternion_to_matrix(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
            new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
            new_t = None
            new_scaling_t = None
            new_rotation_r = None
            if self.gaussian_dim == 4:
                stds_t = self.get_scaling_t[selected_pts_mask].repeat(N,1)
                means_t = torch.zeros((stds_t.size(0), 1),device="cuda")
                samples_t = torch.normal(mean=means_t, std=stds_t)
                new_t = samples_t + self.get_t[selected_pts_mask].repeat(N, 1)
                new_scaling_t = self.scaling_inverse_activation(self.get_scaling_t[selected_pts_mask].repeat(N,1) / (0.8*N))
        else:
            stds = self.get_scaling_xyzt[selected_pts_mask].repeat(N,1)
            means = torch.zeros((stds.size(0), 4),device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation_4d(self._rotation[selected_pts_mask], self._rotation_r[selected_pts_mask]).repeat(N,1,1)
            new_xyzt = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyzt[selected_pts_mask].repeat(N, 1)
            new_xyz = new_xyzt[...,0:3]
            new_t = new_xyzt[...,3:4]
            new_scaling_t = self.scaling_inverse_activation(self.get_scaling_t[selected_pts_mask].repeat(N,1) / (0.8*N))
            new_rotation_r = self._rotation_r[selected_pts_mask].repeat(N,1)

        self.densification_postfix({
            "xyz": new_xyz, 
            "f_dc": new_features_dc, 
            "f_rest": new_features_rest, 
            "opacity": new_opacity, 
            "scaling" : new_scaling, 
            "rotation" : new_rotation,
            "semantic" : new_semantic,
            "t" : new_t,
            "scaling_t" : new_scaling_t,
            "rotation_r" : new_rotation_r
        })

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent, grads_t=None, grad_t_threshold=None):
        
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent)

        self.scalar_dict['points_clone'] = selected_pts_mask.sum().item()
        print(f'Number of points to clone: {selected_pts_mask.sum()}')
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacity = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_semantic = self._semantic[selected_pts_mask]
        new_t = None
        new_scaling_t = None
        new_rotation_r = None
        if self.gaussian_dim == 4:
            new_t = self._t[selected_pts_mask]
            new_scaling_t = self._scaling_t[selected_pts_mask]
            if self.rot_4d:
                new_rotation_r = self._rotation_r[selected_pts_mask]

        self.densification_postfix({
            "xyz": new_xyz, 
            "f_dc": new_features_dc, 
            "f_rest": new_features_rest, 
            "opacity": new_opacity, 
            "scaling" : new_scaling, 
            "rotation" : new_rotation,
            "semantic" : new_semantic,
            "t" : new_t,
            "scaling_t" : new_scaling_t,
            "rotation_r" : new_rotation_r
        })
        
    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, max_grad_t=None):
        grads = self.xyz_gradient_accum[:, 0:1] / self.denom
        grads[grads.isnan()] = 0.0

        if self.gaussian_dim == 4:
                grads_t = self.t_gradient_accum / self.denom
                grads_t[grads_t.isnan()] = 0.0
        else:
            grads_t = None
        
        self.scalar_dict.clear()
        self.scalar_dict['points_total'] = self.get_xyz.shape[0]
        print(f'Number of current gaussians: {self.get_xyz.shape[0]}')

        # Clone and Split        
        self.densify_and_clone(grads, max_grad, extent, grads_t, max_grad_t)
        self.densify_and_split(grads, max_grad, extent, grads_t, max_grad_t)

        # Prune 
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > extent * self.percent_big_ws
            # prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
            prune_mask = torch.logical_or(prune_mask, big_points_ws)

        self.prune_points(prune_mask)
        self.scalar_dict['points_pruned'] = prune_mask.sum().item()            
        print(f'Number of pruned gaussians: {prune_mask.sum()}')

        
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 2), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        torch.cuda.empty_cache()
        
        return self.scalar_dict, self.tensor_dict

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter, 0:1] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
        if self.gaussian_dim == 4:
            self.t_gradient_accum[update_filter] += avg_t_grad[update_filter]
        
    def add_densification_stats_grad(self, viewspace_point_tensor_grad, update_filter):
        self.xyz_gradient_accum[update_filter, 0:1] += torch.norm(viewspace_point_tensor_grad[update_filter, :2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
        if self.gaussian_dim == 4:
            self.t_gradient_accum[update_filter] += avg_t_grad[update_filter]
        
        
    def parse_camera(self, camera: Camera):
        pass
