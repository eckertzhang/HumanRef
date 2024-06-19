# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import math
import os, sys

import cv2
import numpy as np
import torch
from PIL import ImageColor
from pytorch3d.renderer import (
    AlphaCompositor,
    BlendParams,
    FoVOrthographicCameras,
    FoVPerspectiveCameras,
    OrthographicCameras,
    PerspectiveCameras,
    MeshRasterizer,
    MeshRenderer,
    MeshRendererWithFragments,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
    RasterizationSettings,
    SoftSilhouetteShader,
    TexturesVertex,
    blending,
    look_at_view_transform,
)
from pytorch3d.renderer.mesh import TexturesVertex
from pytorch3d.structures import Meshes
from termcolor import colored
from tqdm import tqdm

import mesh_renderer.render_utils as util
sys.path.append(os.path.join(os.getcwd(), 'third_parties'))
from third_parties.ECON.lib.common.imutils import blend_rgb_norm
from third_parties.ECON.lib.dataset.mesh_util import get_visibility


class cleanShader(torch.nn.Module):
    def __init__(self, blend_params=None):
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs):

        # get renderer output
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)
        images = blending.softmax_rgb_blend(texels, fragments, blend_params, znear=-256, zfar=256)

        return images


class MeshRender:
    def __init__(self, size=512, device=torch.device("cuda:0"), geo_prior_type='econ'):
        self.device = device
        self.size = size
        self.geo_prior_type = geo_prior_type
        self.uv_rasterizer = util.Pytorch3dRasterizer(self.size)
        # rasterizer
        self.raster_settings_mesh = RasterizationSettings(
            image_size=self.size,
            blur_radius=np.log(1.0 / 1e-4) * 1e-7,
            bin_size=-1,
            faces_per_pixel=30,
        )
        ang_x = torch.tensor(90 / 180) * torch.pi
        ang_y = torch.tensor(90 / 180) * torch.pi  # -90 to make the front <--> back, satisfing azimuth=0 is front, azimuth=180 is back
        # ang_y = torch.tensor(-90 / 180) * torch.pi  # -90 to make the front <--> back, satisfing azimuth=0 is front, azimuth=180 is back
        ang_z = torch.tensor(0 / 180) * torch.pi
        r_x = torch.tensor([
            [1, 0, 0, 0],
            [0, torch.cos(ang_x), torch.sin(ang_x), 0],
            [0, -torch.sin(ang_x), torch.cos(ang_x), 0],
            [0, 0, 0, 1]
        ])
        r_y = torch.tensor([
            [torch.cos(ang_y), 0, -torch.sin(ang_y), 0],
            [0, 1, 0, 0],
            [torch.sin(ang_y), 0, torch.cos(ang_y), 0],
            [0, 0, 0, 1]
        ])
        r_z = torch.tensor([
            [torch.cos(ang_z), torch.sin(ang_z), 0, 0],
            [-torch.sin(ang_z), torch.cos(ang_z), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        self.three2pytorch3d = torch.mm(r_z, torch.mm(r_y, r_x)).to(self.device)
        self.correct_dir= torch.tensor([
            [-1., 0, 0, 0],
            [0, 1., 0, 0],
            [0, 0, -1., 0],
            [0, 0, 0, 1.]
        ]).to(self.device)
        # align to up-z and front-x
        self.dirs = ["+x", "+y", "+z", "-x", "-y", "-z"]
        self.dir2vec = {
            "+x": np.array([1, 0, 0]),
            "+y": np.array([0, 1, 0]),
            "+z": np.array([0, 0, 1]),
            "-x": np.array([-1, 0, 0]),
            "-y": np.array([0, -1, 0]),
            "-z": np.array([0, 0, -1]),
        }
        
        pytorch3d_mesh_up = '+y'
        pytorch3d_mesh_front = '-z'
        z_, x_ = (
            self.dir2vec[pytorch3d_mesh_up],
            self.dir2vec[pytorch3d_mesh_front],
        )
        y_ = np.cross(z_, x_)
        self.std2mesh = torch.tensor(np.stack([x_, y_, z_], axis=0).T, dtype=torch.float32).to(self.device)
        self.mesh2std = torch.inverse(self.std2mesh).to(self.device)

    def load_meshes(self, verts, faces):
        """load mesh into the pytorch3d renderer

        Args:
            verts ([N,3] / [B,N,3]): array or tensor
            faces ([N,3]/ [B,N,3]): array or tensor
        """
        # verts = torch.mm(self.mesh2std, verts.T).T

        # array or tensor
        if not torch.is_tensor(verts):
            verts = torch.tensor(verts)   
            faces = torch.tensor(faces)
        
        from scipy.spatial.transform import Rotation
        rot_y = Rotation.from_euler('y', 180, degrees=True).as_matrix().astype(np.float32)
        self.rot_y = torch.from_numpy(rot_y)
        rot_z = Rotation.from_euler('z', 180, degrees=True).as_matrix().astype(np.float32)
        self.rot_z = torch.from_numpy(rot_z)
        if self.geo_prior_type == 'econ_smpl' or self.geo_prior_type == 'econ' or self.geo_prior_type == 'cape_smpl' or self.geo_prior_type == 'thuman2_smpl':
            verts = torch.matmul(self.rot_y, verts.T).T
        if verts.ndimension() == 2:
            verts = verts.float().unsqueeze(0).to(self.device)
            faces = faces.long().unsqueeze(0).to(self.device)
        if verts.shape[0] != faces.shape[0]:
            faces = faces.repeat(len(verts), 1, 1).to(self.device)
        self.meshes = Meshes(verts, faces).to(self.device)

        # texture only support single mesh
        if len(self.meshes) == 1:
            self.meshes.textures = TexturesVertex(
                verts_features=(self.meshes.verts_normals_padded() + 1.0) * 0.5
            )

    def get_depth_normal(self, c2w_three, K=None, bg="gray"):
        if self.geo_prior_type == 'econ_smpl' or self.geo_prior_type == 'econ' or self.geo_prior_type == 'cape_smpl' or self.geo_prior_type == 'thuman2_smpl':
            c2w = torch.mm(self.three2pytorch3d, c2w_three.to(self.device))
            c2w[:3, 3] = c2w[:3, :3].t()@c2w[:3, 3]
            # c2w = torch.mm(c2w, self.correct_dir)  # make the azimuth & elevation angles astisfy threestudio
            # c2w[:3, 3] = -c2w[:3, :3].t()@c2w[:3, 3]
            R = c2w[:3, :3][None]
            T = c2w[:3, 3][None]
            camera = FoVOrthographicCameras(
                device=self.device,
                R=R,
                T=T,
                znear=1.0,
                zfar=-1.0,
                max_y=1.0,
                min_y=-1.0,
                max_x=1.0,
                min_x=-1.0,
            )
            # camera = OrthographicCameras(
            #     device=self.device,
            #     R=R,
            #     T=T,
            # )
        elif self.geo_prior_type == 'hybrik':
            # R = torch.mm(self.std2mesh, c2w_three[:3, :3])[None]
            # T = torch.mm(self.std2mesh, c2w_three[:3, 3:])[None, :, 0]
            # w2c_std = torch.inverse(c2w_three)
            # R = torch.mm(w2c_std[:3, :3], self.mesh2std)[None]

            # x_std, y_std, z_std, t_std = c2w_three[:3, 0], c2w_three[:3, 1], c2w_three[:3, 2], c2w_three[:3, 3]
            # c2w_py3d = torch.eye(4, device=self.device)
            # c2w_py3d[:3, 0] = -y_std
            # c2w_py3d[:3, 1] = z_std
            # c2w_py3d[:3, 2] = -x_std
            # c2w_py3d[:3, 3] = torch.mm(self.std2mesh, t_std[:, None])[:, 0]
            # R = c2w_py3d[:3, :3][None]
            # T = c2w_py3d[:3, 3][None]

            c2w = torch.mm(self.three2pytorch3d, c2w_three.to(self.device))
            c2w[:3, 3] = c2w[:3, :3].t()@c2w[:3, 3]
            R = c2w[:3, :3][None]
            T = c2w[:3, 3][None]

            camera = FoVPerspectiveCameras(
                device=self.device,
                R=R,
                T=T,
            )

        blendparam = BlendParams(1e-4, 1e-8, np.array(ImageColor.getrgb(bg)) / 255.0)
        
        meshRas = MeshRasterizer(cameras=camera, raster_settings=self.raster_settings_mesh)
        renderer = MeshRendererWithFragments(
            rasterizer=meshRas,
            shader=cleanShader(blend_params=blendparam),
        )
        current_mesh = self.meshes[0]
        current_mesh.textures = TexturesVertex(
            verts_features=(current_mesh.verts_normals_padded() + 1.0) * 0.5
        )

        # fragments = meshRas(current_mesh.extend(1))
        images, fragments = renderer(current_mesh.extend(1))

        depth = fragments.zbuf[..., 0]

        # normal = ((images[:, :, :, :3] - 0.5) * 2.0).permute(0, 3, 1, 2)
        normal0 = ((images[:, :, :, :3] - 0.5) * 2.0)
        nn0 = torch.mm(self.mesh2std, normal0.view(-1, 3).T).T
        normal = nn0.view(normal0.shape).permute(0, 3, 1, 2)

        mask = images[:, :, :, 3]

        # import imageio
        # name = str('a woman in red pants and a white shirt holding a shopping bag').replace(' ', '_')
        # aa=depth[0].detach().cpu().numpy()
        # aa[aa<0]=0
        # imageio.imwrite(os.path.join( f'{name}.png'), (aa/(aa.max())*255).astype(np.uint8))
        # bb=((normal[0]+1.)*0.5).permute(1,2,0).detach().cpu().numpy()
        # imageio.imwrite(os.path.join(f'nn_{name}.png'), (bb*255).astype(np.uint8))

        return depth, normal, mask
