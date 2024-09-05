import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from lib.utils.sh_utils import eval_sh, eval_shfs_4d
from lib.models.gaussian_model import GaussianModel
from lib.models.street_gaussian_model import StreetGaussianModel
from typing import Union

from lib.utils.camera_utils import Camera
from lib.config import cfg

class GaussianRenderer():
    def __init__(
        self,         
    ):
        self.cfg = cfg.render

    def render(
        self, 
        viewpoint_camera: Camera,
        pc: Union[GaussianModel, StreetGaussianModel],
        convert_SHs_python = None, 
        compute_cov3D_python = None, 
        scaling_modifier = None, 
        override_color = None
    ):

        bg_color = [1, 1, 1] if cfg.data.white_background else [0, 0, 0]
        bg_color = torch.tensor(bg_color).float().cuda()
        bg_depth = torch.tensor([0]).float().cuda()
        scaling_modifier = scaling_modifier or self.cfg.scaling_modifier
        convert_SHs_python = convert_SHs_python or self.cfg.convert_SHs_python
        compute_cov3D_python = compute_cov3D_python or self.cfg.compute_cov3D_python
        override_color = override_color or self.cfg.override_color
    
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg_color=bg_color,
            bg_depth=bg_depth,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=self.cfg.debug
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = pc.get_xyz
        means2D = screenspace_points
        opacity = pc.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        scales_t = None
        rotations = None
        rotations_r = None
        ts = None
        cov3D_precomp = None
        if compute_cov3D_python:
            if pc.rot_4d:
                cov3D_precomp, delta_mean = pc.get_current_covariance_and_mean_offset(scaling_modifier, viewpoint_camera.timestamp)
                means3D = means3D + delta_mean
            else:
                cov3D_precomp = pc.get_covariance(scaling_modifier)
            if pc.gaussian_dim == 4:
                marginal_t = pc.get_marginal_t(viewpoint_camera.timestamp)
                # marginal_t = torch.clamp_max(marginal_t, 1.0) # NOTE: 这里乘完会大于1，绝对不行——marginal_t应该用个概率而非概率密度 暂时可以clamp一下，后期用积分 —— 2d 也用的clamp
                opacity = opacity * marginal_t
        else:
            scales = pc.get_scaling
            rotations = pc.get_rotation
            if pc.gaussian_dim = 4:
                scales_t = pc.get_scaling_t
                ts = pc.get_t
                if pc.rot_4d:
                    rotations_r = pc.get_rotation_r

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if override_color is None:
            if convert_SHs_python:
                shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
                dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
                dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                if pc.gaussian_dim == 3:# or pc.force_sh_3d:
                    sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                elif pc.gaussian_dim == 4:
                    dir_t = (pc.get_t - viewpoint_camera.timestamp).detach()
                    sh2rgb = eval_shfs_4d(pc.active_sh_degree, pc.active_sh_degree_t, shs_view, dir_pp_normalized, dir_t, pc.time_duration[1] - pc.time_duration[0])
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
                # sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                # colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = pc.get_features
                if pc.gaussian_dim == 4 and ts is None:
                    ts = pc.get_t
        else:
            colors_precomp = override_color

        flow_2d = torch.zeros_like(pc.get_xyz[:,:2])
    
    # Prefilter
    if pipe.compute_cov3D_python and pc.gaussian_dim == 4:
        mask = marginal_t[:,0] > 0.05
        if means2D is not None:
            means2D = means2D[mask]
        if means3D is not None:
            means3D = means3D[mask]
        if ts is not None:
            ts = ts[mask]
        if shs is not None:
            shs = shs[mask]
        if colors_precomp is not None:
            colors_precomp = colors_precomp[mask]
        if opacity is not None:
            opacity = opacity[mask]
        if scales is not None:
            scales = scales[mask]
        if scales_t is not None:
            scales_t = scales_t[mask]
        if rotations is not None:
            rotations = rotations[mask]
        if rotations_r is not None:
            rotations_r = rotations_r[mask]
        if cov3D_precomp is not None:
            cov3D_precomp = cov3D_precomp[mask]
        if flow_2d is not None:
            flow_2d = flow_2d[mask]

        # Rasterize visible Gaussians to image, obtain their radii (on screen). 
        rendered_color, radii, rendered_depth, rendered_acc, rendered_semantic = rasterizer(
            means3D = means3D,
            means2D = means2D,
            opacities = opacity,
            shs = shs,
            colors_precomp = colors_precomp,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp,
            semantics = None,
            flow_ed2d = flow_2d, 
            ts = ts,
            scales_t = scales_t,
            rotations_r = totations_r
        )  

        if cfg.mode != 'train':
            rendered_color = torch.clamp(rendered_color, 0., 1.)        
        
        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {"rgb": rendered_color,
                "acc": rendered_acc,
                "depth": rendered_depth,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii, 
                "flow": flow}
