def render_kernel(
        self, 
        viewpoint_camera: Camera,
        pc: StreetGaussianModel,
        convert_SHs_python = None, 
        compute_cov3D_python = None, 
        scaling_modifier = None, 
        override_color = None,
        white_background = cfg.data.white_background,
    ):

        if(compute_cov3D_python):
            print("compute_cov3D_python")
        
        if pc.num_gaussians == 0:
            if white_background:
                rendered_color = torch.ones(3, int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), device="cuda")
            else:
                rendered_color = torch.zeros(3, int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), device="cuda")
            
            rendered_acc = torch.zeros(1, int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), device="cuda")
            rendered_semantic = torch.zeros(0, int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), device="cuda")
            
            return {
                "rgb": rendered_color,
                "acc": rendered_acc,
                "semantic": rendered_semantic,
            }

        # Set up rasterization configuration and make rasterizer
        bg_color = [1, 1, 1] if white_background else [0, 0, 0]
        bg_color = torch.tensor(bg_color).float().cuda()
        scaling_modifier = scaling_modifier or self.cfg.scaling_modifier
        rasterizer = make_rasterizer(viewpoint_camera, pc.max_sh_degree, bg_color, scaling_modifier)
        
        convert_SHs_python = convert_SHs_python or self.cfg.convert_SHs_python
        compute_cov3D_python = compute_cov3D_python or self.cfg.compute_cov3D_python

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        # screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
        if cfg.mode == 'train':
            screenspace_points = torch.zeros((pc.num_gaussians, 3), requires_grad=True).float().cuda() + 0
            try:
                screenspace_points.retain_grad()
            except:
                pass
        else:
            screenspace_points = None 

        means3D = []
        opacity = []
        scales = None
        rotations = None
        cov3D_precomp = []

        xyz_bkgd = pc.get_xyz_bkgd
        opacity_bkgd = pc.get_opacity_bkgd
        cov3D_bkgd = pc.get_cov_bkgd
        means3D.append(xyz_bkgd)
        opacity.append(opacity_bkgd)
        cov3D_precomp.append(cov3D_bkgd)

        if len(pc.graph_obj_list) > 0:
            xyzs_local = []
            cov3D_objs = []

            for i, obj_name in enumerate(pc.graph_obj_list):
                obj_model: GaussianModelActor = getattr(pc, obj_name)
                xyz_local = obj_model.get_xyz
                #xyzs_local.append(xyz_local)

                opacity_obj = obj_model.get_opacity
                marginal_t = obj_model.get_marginal_t(viewpoint_camera.timestamp)
                opacity_obj = opacity_obj * marginal_t
                opacity.append(opacity_obj)

                cov3D_obj, delta_mean = pc.get_current_covariance_and_mean_offset(scaling_modifier, viewpoint_camera.timestamp)
                cov3D_objs.append(conv3D_obj)
                xyz_local = xyz_local + delta_mean
                xyzs_local.append(xyz_local)
                
            xyzs_local = torch.cat(xyzs_local, dim=0)
            xyzs_local = xyzs_local.clone()
            xyzs_local[pc.flip_mask, pc.flip_axis] *= -1
            obj_rots = quaternion_to_matrix(pc.obj_rots)
            xyzs_obj = torch.einsum('bij, bj -> bi', obj_rots, xyzs_local) + pc.obj_trans
            means3D.append(xyzs_obj)
            cov3D_precomp.append(cov3D_objs)

        means3D = torch.cat(xyzs, dim=0)
        opacity = torch.cat(opacity, dim=0)
        cov3D_precomp = torch.cat(cov3D_precomp, dim=0)
        means2D = screenspace_points
        # opacity = pc.get_opacity
        
        
        # for obj_name in pc.graph_obj_list:
        #     obj_model: GaussianModelActor = getattr(pc, obj_name)
        #     opacity_obj = obj_model.get_opacity
        #     marginal_t = obj_model.get_marginal_t(viewpoint_camera.timestamp)
        #     opacity_obj = opacity_obj * marginal_t
        #     opacity.append(opacity_obj)
        

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        # scales = None
        # rotations = None
        # cov3D_precomp = None
        # if compute_cov3D_python:
        #     if pc.rot_4d:
        #         cov3D_precomp, delta_mean = pc.get_current_covariance_and_mean_offset(scaling_modifier, viewpoint_camera.timestamp)
        #         means3D = means3D + delta_mean
        #     else:
        #         cov3D_precomp = pc.get_covariance(scaling_modifier)
            #if pc.gaussian_dim == 4:
                
                # marginal_t = torch.clamp_max(marginal_t, 1.0) # NOTE: 这里乘完会大于1，绝对不行——marginal_t应该用个概率而非概率密度 暂时可以clamp一下，后期用积分 —— 2d 也用的clamp
            # cov3D_precomp = pc.get_covariance(scaling_modifier)
        # else:
        #     scales = pc.get_scaling
        #     rotations = pc.get_rotation
        #     # if pc.gaussian_dim == 4:
        #     scales_t = pc.get_scaling_t
        #     ts = pc.get_t
        #     #    if pc.rot_4d:
        #     rotations_r = pc.get_rotation_r

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if override_color is None:
            if convert_SHs_python:
                print("convert_SHs_python")
                shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
                dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                # if pc.gaussian_dim == 3 or pc.force_sh_3d:
                #     sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                # elif pc.gaussian_dim == 4:
                #     dir_t = (pc.get_t - viewpoint_camera.timestamp).detach()
                #     sh2rgb = eval_shfs_4d(pc.active_sh_degree, pc.active_sh_degree_t, shs_view, dir_pp_normalized, dir_t, pc.time_duration[1] - pc.time_duration[0])
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                try:
                    shs = pc.get_features
                except:
                    colors_precomp = pc.get_colors(viewpoint_camera.camera_center)
        else:
            colors_precomp = override_color

        # TODO: add more feature here
        feature_names = []
        feature_dims = []
        features = []
        
        if cfg.render.render_normal:
            normals = pc.get_normals(viewpoint_camera)
            feature_names.append('normals')
            feature_dims.append(normals.shape[-1])
            features.append(normals)

        if cfg.data.get('use_semantic', False):
            semantics = pc.get_semantic
            feature_names.append('semantic')
            feature_dims.append(semantics.shape[-1])
            features.append(semantics)
        
        if len(features) > 0:
            features = torch.cat(features, dim=-1)
        else:
            features = None

        # flow_2d = torch.zeros_like(pc.get_xyz[:,:2])
        
        # # Prefilter
        # if compute_cov3D_python #and pc.gaussian_dim == 4:
        #     mask = marginal_t[:,0] > 0.05
        #     if means2D is not None:
        #         means2D = means2D[mask]
        #     if means3D is not None:
        #         means3D = means3D[mask]
        #     if ts is not None:
        #         ts = ts[mask]
        #     if shs is not None:
        #         shs = shs[mask]
        #     if colors_precomp is not None:
        #         colors_precomp = colors_precomp[mask]
        #     if opacity is not None:
        #         opacity = opacity[mask]
        #     if scales is not None:
        #         scales = scales[mask]
        #     if scales_t is not None:
        #         scales_t = scales_t[mask]
        #     if rotations is not None:
        #         rotations = rotations[mask]
        #     if rotations_r is not None:
        #         rotations_r = rotations_r[mask]
        #     if cov3D_precomp is not None:
        #         cov3D_precomp = cov3D_precomp[mask]
        #     if flow_2d is not None:
        #         flow_2d = flow_2d[mask]
        
        
        # Rasterize visible Gaussians to image, obtain their radii (on screen). 
        rendered_color, radii, rendered_depth, rendered_acc, rendered_feature = rasterizer(
            means3D = means3D,
            means2D = means2D,
            opacities = opacity,
            shs = shs,
            colors_precomp = colors_precomp,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp,
            semantics = features,
            # flow_2d = flow_2d,
            # ts = ts,
            # scales_t = scales_t,
            # rotations_r = rotations_r
        )  
        
        if cfg.mode != 'train':
            rendered_color = torch.clamp(rendered_color, 0., 1.)
        
        rendered_feature_dict = dict()
        if rendered_feature.shape[0] > 0:
            rendered_feature_list = torch.split(rendered_feature, feature_dims, dim=0)
            for i, feature_name in enumerate(feature_names):
                rendered_feature_dict[feature_name] = rendered_feature_list[i]
        
        if 'normals' in rendered_feature_dict:
            rendered_feature_dict['normals'] = torch.nn.functional.normalize(rendered_feature_dict['normals'], dim=0)
                
        if 'semantic' in rendered_feature_dict:
            rendered_semantic = rendered_feature_dict['semantic']
            semantic_mode = cfg.model.gaussian.get('semantic_mode', 'logits')
            assert semantic_mode in ['logits', 'probabilities']
            if semantic_mode == 'logits': 
                pass # return raw semantic logits
            else:
                rendered_semantic = rendered_semantic / (torch.sum(rendered_semantic, dim=0, keepdim=True) + 1e-8) # normalize to probabilities
                rendered_semantic = torch.log(rendered_semantic + 1e-8) # change for cross entropy loss

            rendered_feature_dict['semantic'] = rendered_semantic
        
        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        
        result = {
            "rgb": rendered_color,
            "acc": rendered_acc,
            "depth": rendered_depth,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii, 
            # "flow": flow
        }
        
        result.update(rendered_feature_dict)
        
        return result