import bisect
import math
import os
import trimesh
import cv2, imageio
import numpy as np
import random
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, IterableDataset
from dataclasses import dataclass, field
from mesh_renderer.meshrender import MeshRender
import threestudio
from threestudio import register
from threestudio.data.uncond import (
    RandomCameraDataModuleConfig,
    RandomCameraDataset,
    RandomCameraIterableDataset,
)
from threestudio.utils.base import Updateable
from threestudio.utils.config import parse_structured
from threestudio.utils.misc import get_rank, get_device
from threestudio.utils.ops import (
    get_mvp_matrix,
    get_projection_matrix,
    get_ray_directions,
    get_rays,
    get_orthoghphic_ray_directions,
    get_orthoghphic_rays,
)
from threestudio.utils.typing import *


def gen_random_view_angle(dir_select, num_per_dir):
    out = {}
    for dir_ in dir_select:
        if dir_ == 'front':
            # azimuth, elevation
            out['front']= [[random.randint(-15, 15), random.randint(-15, 15)] for i in range(num_per_dir)]
        elif dir_ == 'back':
            out['back']= [[random.randint(165, 195), random.randint(-15, 15)] for i in range(num_per_dir)]
        elif dir_ == 'left':
            out['left']= [[random.randint(75, 105), random.randint(-15, 15)] for i in range(num_per_dir)]
        elif dir_ == 'right':
            out['right']= [[random.randint(-105, -75), random.randint(-15, 15)] for i in range(num_per_dir)]
        elif dir_ == 'left_front':
            out['left_front']= [[random.randint(15, 75), random.randint(-15, 15)] for i in range(num_per_dir)]
        elif dir_ == 'left_back':
            out['left_back']= [[random.randint(105, 165), random.randint(-15, 15)] for i in range(num_per_dir)]
        elif dir_ == 'right_front':
            out['right_front']= [[random.randint(-75, -15), random.randint(-15, 15)] for i in range(num_per_dir)]
        elif dir_ == 'right_back':
            out['right_back']= [[random.randint(-165, -105), random.randint(-15, 15)] for i in range(num_per_dir)]
    return out

def warp_mtx_on_std(ang_x_deg, ang_y_deg, ang_z_deg):
        ang_x = torch.tensor(ang_x_deg / 180) * torch.pi
        ang_y = torch.tensor(ang_y_deg / 180) * torch.pi 
        ang_z = torch.tensor(ang_z_deg / 180) * torch.pi
        r_x = torch.tensor([
            [1, 0, 0],
            [0, torch.cos(ang_x), torch.sin(ang_x)],
            [0, -torch.sin(ang_x), torch.cos(ang_x)]
        ])
        r_y = torch.tensor([
            [torch.cos(ang_y), 0, -torch.sin(ang_y)],
            [0, 1, 0],
            [torch.sin(ang_y), 0, torch.cos(ang_y)]
        ])
        r_z = torch.tensor([
            [torch.cos(ang_z), torch.sin(ang_z), 0],
            [-torch.sin(ang_z), torch.cos(ang_z), 0],
            [0, 0, 1]
        ])
        return torch.mm(r_z, torch.mm(r_y, r_x))



@dataclass
class SingleImageDataModuleConfig:
    height: Any = 64
    width: Any = 64
    resolution_milestones: List[int] = field(default_factory=lambda: [])
    height_img: int = 512
    width_img: int = 512
    eval_height: int = 512
    eval_width: int = 512
    default_elevation_deg: float = 0.0
    default_azimuth_deg: float = -180.0
    default_camera_distance: float = 1.2
    default_fovy_deg: float = 60.0
    image_path: str = ""
    prompt: str = ""
    workspace: str = ""
    geo_prior_type: str = ""
    use_random_camera: bool = True
    random_camera: dict = field(default_factory=dict)
    rays_noise_scale: float = 2e-3
    batch_size: int = 1
    ## Orthograph related
    use_orthograph: bool = True
    radius: float = 1.0
    run_local_rendering: bool = True
    ## PIDM constrain related
    use_pidm_prior: bool = False
    ckpt_pidm: str = ""
    save_dir_pidm: str = ""


class SingleImageRandomDataset(Dataset):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = get_device()
        random_camera_cfg = parse_structured(RandomCameraDataModuleConfig, self.cfg.get("random_camera", {}))
        self.random_pose_generator = RandomCameraDataset(random_camera_cfg, split)

        # img_name = os.path.splitext(os.path.basename(self.cfg.image_path))[0]
        # if self.cfg.geo_prior_type == 'econ_smpl' or self.cfg.geo_prior_type == 'econ':
        #     econ_out_path = f'/apdcephfs/private_eckertzhang/Codes/Results/Results_ECON/{img_name}'
        #     if self.cfg.geo_prior_type == 'econ_smpl':
        #         obj_file = os.path.join(econ_out_path, f'econ/obj/{img_name}_smpl_00.obj')
        #     else:
        #         obj_file = os.path.join(econ_out_path, f'econ/obj/{img_name}_{0}_full.obj')
        # elif self.cfg.geo_prior_type == 'cape_smpl':
        #     econ_out_path = f'/apdcephfs/share_1330077/eckertzhang/Dataset/cape_test_data/results_econ/{img_name}'
        #     obj_file = os.path.join(econ_out_path, f'econ/obj/{img_name}_smpl_00_prior.obj')
        # elif self.cfg.geo_prior_type == 'thuman2_smpl':
        #     econ_out_path = f'/apdcephfs/share_1330077/eckertzhang/Dataset/thuman2_icon/results_econ/{img_name}'
        #     obj_file = os.path.join(econ_out_path, f'econ/obj/{img_name}_smpl_00_prior.obj')
        # mesh = trimesh.load(obj_file)
        # verts = torch.tensor(mesh.vertices).float()
        # faces = torch.tensor(mesh.faces).long()
        # self.mesh_render = MeshRender(size=512, geo_prior_type=self.cfg.geo_prior_type, device=self.device)
        # self.mesh_render.load_meshes(verts, faces)
        

    def __len__(self):
        return len(self.random_pose_generator)

    def collate(self, batch):
        batch = torch.utils.data.default_collate(batch)
        # batch.update({"mesh_render": self.mesh_render})
        return batch

    def __getitem__(self, index):
        batch = self.random_pose_generator[index]
        return batch

class SingleImageIterableDataset(IterableDataset, Updateable):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.rank = get_rank()
        self.device = get_device()
        self.cfg: SingleImageDataModuleConfig = cfg
        del cfg
        self.heights: List[int] = (
            [self.cfg.height] if isinstance(self.cfg.height, int) else self.cfg.height
        )
        self.widths: List[int] = (
            [self.cfg.width] if isinstance(self.cfg.width, int) else self.cfg.width
        )
        assert len(self.heights) == len(self.widths)
        self.resolution_milestones: List[int]
        if len(self.heights) == 1 and len(self.widths) == 1:
            if len(self.cfg.resolution_milestones) > 0:
                threestudio.warn(
                    "Ignoring resolution_milestones since height and width are not changing"
                )
            self.resolution_milestones = [-1]
        else:
            assert len(self.heights) == len(self.cfg.resolution_milestones) + 1
            self.resolution_milestones = [-1] + self.cfg.resolution_milestones
        
        self.directions_unit_focals = [
            get_ray_directions(H=height, W=width, focal=1.0).to(self.device) 
            if not self.cfg.use_orthograph else get_orthoghphic_ray_directions(H=height, W=width, radius=self.cfg.radius).to(self.device)
            for (height, width) in zip(self.heights, self.widths)
        ]  # (global rendering)
        
        self.height: int = self.heights[0]
        self.width: int = self.widths[0]
        self.directions_unit_focal = self.directions_unit_focals[0]
        if self.cfg.run_local_rendering:
            self.general_origins_locals = [
                get_orthoghphic_ray_directions(H=height, W=width, radius=self.cfg.radius*0.33).to(self.device)
                for (height, width) in zip(self.heights, self.widths)
            ]   # only for orthoghphic rendering (local rendering)
            self.general_origins_local = self.general_origins_locals[0]

        if self.cfg.use_random_camera:
            random_camera_cfg = parse_structured(RandomCameraDataModuleConfig, self.cfg.get("random_camera", {}))
            self.random_pose_generator = RandomCameraIterableDataset(random_camera_cfg)

        # camera parameters
        self.fovy = torch.deg2rad(torch.FloatTensor([self.cfg.default_fovy_deg])).to(self.device)
        self.elevation_deg = torch.FloatTensor([self.cfg.default_elevation_deg]).to(self.device)
        self.azimuth_deg = torch.FloatTensor([self.cfg.default_azimuth_deg]).to(self.device)
        self.azimuth_deg_back = torch.FloatTensor([self.cfg.default_azimuth_deg + 180]).to(self.device)
        self.camera_distance = torch.FloatTensor([self.cfg.default_camera_distance]).to(self.device)
        self.camera_position, self.c2w = self.get_c2w(self.elevation_deg, self.azimuth_deg, self.camera_distance)
        self.camera_position_back, self.c2w_back = self.get_c2w(self.elevation_deg, self.azimuth_deg_back, self.camera_distance)
        self.light_position: Float[Tensor, "1 3"] = self.camera_position

        # load guided mesh (ECON OR HybrIK)
        img_name = os.path.splitext(os.path.basename(self.cfg.image_path))[0].replace('_0_rgba', '')
        if self.cfg.geo_prior_type == 'econ_smpl' or self.cfg.geo_prior_type == 'econ':
            econ_out_path = os.path.join(os.getenv('ECON_PATH'), img_name)
            if self.cfg.geo_prior_type == 'econ_smpl':
                obj_file = os.path.join(econ_out_path, f'econ/obj/{img_name}_smpl_00.obj')
            else:
                obj_file = os.path.join(econ_out_path, f'econ/obj/{img_name}_{0}_full.obj')
        elif self.cfg.geo_prior_type == 'cape_smpl':
            econ_out_path =  os.path.join(os.getenv('Human3D_PATH'), f'cape_test_data/results_econ/{img_name}') 
            obj_file = os.path.join(econ_out_path, f'econ/obj/{img_name}_smpl_00_prior.obj')
        elif self.cfg.geo_prior_type == 'thuman2_smpl':
            econ_out_path = os.path.join(os.getenv('Human3D_PATH'), f'thuman2/results_econ/{img_name}')
            obj_file = os.path.join(econ_out_path, f'econ/obj/{img_name}_smpl_00_prior.obj')
        mesh = trimesh.load(obj_file)
        verts = torch.tensor(mesh.vertices).float()
        faces = torch.tensor(mesh.faces).long()
        self.mesh_render = MeshRender(size=512, geo_prior_type=self.cfg.geo_prior_type)
        self.mesh_render.load_meshes(verts, faces)
        proj_mtx: Float[Tensor, "4 4"] = get_projection_matrix(
            self.fovy, self.width / self.height, 0.1, 100.0
        )
        self.depth, self.normal, self.mask_front = self.mesh_render.get_depth_normal(self.c2w[0], K=proj_mtx, bg="gray")
        self.depth_back, self.normal_back, self.mask_back = self.mesh_render.get_depth_normal(self.c2w_back[0], K=proj_mtx, bg="gray")
        self.normal, self.normal_back = self.normal.permute(0, 2, 3, 1), self.normal_back.permute(0, 2, 3, 1)
        self.mask_front, self.mask_back = self.mask_front.unsqueeze(-1), self.mask_back.unsqueeze(-1)
        
        # save rendered depth & normal    
        name = self.cfg.prompt.replace(' ', '_')
        bb = ((self.normal[0]+1.)*0.5).detach().cpu().numpy()
        imageio.imwrite(os.path.join(self.cfg.workspace, f'vis_normal-{name}-front.png'), (bb*255).astype(np.uint8))
        bb = ((self.normal_back[0]+1.)*0.5).detach().cpu().numpy()
        imageio.imwrite(os.path.join(self.cfg.workspace, f'vis_normal-{name}-back.png'), (bb*255).astype(np.uint8))
        del bb

        # Use the croped image to replace input image as GT
        self.mask_flip = None
        self.depth = self.depth.unsqueeze(-1).to(self.device)
        self.depth_back = self.depth_back.unsqueeze(-1).to(self.device)
        if self.cfg.geo_prior_type == 'econ_smpl' or self.cfg.geo_prior_type == 'cape_smpl' or self.cfg.geo_prior_type == 'thuman2_smpl':
            path_croped_img = os.path.join(econ_out_path, f'econ/imgs_crop/{img_name}_0_rgba.png')
        assert os.path.exists(path_croped_img)
        rgba_init = cv2.cvtColor(
            cv2.imread(path_croped_img, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA
        )
        rgba = (
            cv2.resize(
                rgba_init, (self.cfg.width_img, self.cfg.height_img), interpolation=cv2.INTER_AREA
            ).astype(np.float32)
            / 255.0
        )
        rgb = rgba[..., :3]
        self.rgb: Float[Tensor, "1 H W 3"] = (
            torch.from_numpy(rgb).unsqueeze(0).contiguous().to(self.device)
        )
        self.mask: Float[Tensor, "1 H W 1"] = (
            torch.from_numpy(rgba[..., 3:] > 0.5).unsqueeze(0).to(self.device)
        )
        self.mask_flip = torch.flip(self.mask, [-2])
        print(
            f"[INFO] single image dataset: load normal from ECON model"
        )
        if self.cfg.run_local_rendering:
            self.cent_head_rate, self.cent_foot_rate = 0.8, 0.7
            h, w = rgba_init.shape[:2]
            radius_crop = int(h*(1-self.cent_head_rate))
            rgba_head = rgba_init[:radius_crop, int((w-radius_crop)/2):int((w+radius_crop)/2)]
            radius_crop = int(h*(1-self.cent_foot_rate))
            rgba_foot = rgba_init[-radius_crop:, int((w-radius_crop)/2):int((w+radius_crop)/2)]
            rgba_head = (
                cv2.resize(
                    rgba_head, (self.cfg.width_img, self.cfg.height_img), interpolation=cv2.INTER_AREA
                ).astype(np.float32) / 255.0
            )
            rgba_foot = (
                cv2.resize(
                    rgba_foot, (self.cfg.width_img, self.cfg.height_img), interpolation=cv2.INTER_AREA
                ).astype(np.float32) / 255.0
            )

            # imageio.imwrite("test.png", (rgba_head*255.).astype('uint8'))
            self.rgb_head: Float[Tensor, "1 H W 3"] = (
                torch.from_numpy(rgba_head[..., :3]).unsqueeze(0).contiguous().to(self.device)
            )
            self.mask_head: Float[Tensor, "1 H W 1"] = (
                torch.from_numpy(rgba_head[..., 3:] > 0.5).unsqueeze(0).to(self.device)
            )
            self.rgb_foot: Float[Tensor, "1 H W 3"] = (
                torch.from_numpy(rgba_foot[..., :3]).unsqueeze(0).contiguous().to(self.device)
            )
            self.mask_foot: Float[Tensor, "1 H W 1"] = (
                torch.from_numpy(rgba_foot[..., 3:] > 0.5).unsqueeze(0).to(self.device)
            )

        # Estimate fore/back-ground Normals via ECON
        from third_parties.ECON.lib.common.config import cfg as cfg_n
        from third_parties.ECON.apps.Normal import Normal
        cfg_n.merge_from_file("./third_parties/ECON/configs/econ.yaml")
        cfg_n.merge_from_file("./third_parties/ECON/lib/pymafx/configs/pymafx_config.yaml")
        if os.getenv('WEIGHT_PATH') is not None:
            cfg_n.root = os.path.join(os.getenv('WEIGHT_PATH'), 'econ_weights/')
            cfg_n.ckpt_dir = os.path.join(os.getenv('WEIGHT_PATH'), 'econ_weights/ckpt/')
            cfg_n.normal_path = os.path.join(os.getenv('WEIGHT_PATH'), 'econ_weights/ckpt/normal.ckpt')
            cfg_n.ifnet_path = os.path.join(os.getenv('WEIGHT_PATH'), 'econ_weights/ckpt/ifnet.ckpt')
        normal_net = Normal.load_from_checkpoint(
            cfg=cfg_n, checkpoint_path=cfg_n.normal_path, map_location=self.device, strict=False
        )
        normal_net = normal_net.to(self.device)
        normal_net.netG.eval()
        warp_mtx = torch.mm(self.mesh_render.rot_y.to(self.device), self.mesh_render.std2mesh)
        warp_mtx_b = torch.mm(self.mesh_render.rot_z.to(self.device), self.mesh_render.std2mesh)
        mesh_render_temp = MeshRender(size=512, geo_prior_type=self.cfg.geo_prior_type)
        mesh_render_temp.load_meshes(verts*torch.tensor([-1.0, 1.0, 1.0]), faces)
        _, normal_back_temp, _ = mesh_render_temp.get_depth_normal(self.c2w_back[0], K=proj_mtx, bg="gray")
        in_tensor = {
            "image": (self.rgb*self.mask).permute(0,3,1,2).to(self.device), 
            "mask": self.mask.permute(0,3,1,2).to(self.device),
            "T_normal_F": torch.mm(warp_mtx, self.normal.view(-1, 3).T).T.view(self.rgb.shape).permute(0,3,1,2),
            "T_normal_B": torch.mm(warp_mtx_b, normal_back_temp.permute(0,2,3,1).view(-1, 3).T).T.view(self.rgb.shape).permute(0,3,1,2)
        }
        with torch.no_grad():
            in_tensor["normal_F"], in_tensor["normal_B"] = normal_net.netG(in_tensor)
        self.normal_e = torch.mm(torch.inverse(warp_mtx), in_tensor["normal_F"].permute(0,2,3,1).view(-1, 3).T).T.view(self.rgb.shape)
        self.normal_back_e = torch.mm(torch.inverse(warp_mtx), in_tensor["normal_B"].permute(0,2,3,1).view(-1, 3).T).T.view(self.rgb.shape)
        self.normal_back_e = torch.flip(self.normal_back_e, [-2]) 
        
        # save estimated normal   
        name = self.cfg.prompt.replace(' ', '_')
        bb = ((self.normal_e[0]+1.)*0.5).detach().cpu().numpy()
        imageio.imwrite(os.path.join(self.cfg.workspace, f'vis_guide-normal-{name}-front-e.png'), (bb*255).astype(np.uint8))
        bb = ((self.normal_back_e[0]+1.)*0.5).detach().cpu().numpy()
        imageio.imwrite(os.path.join(self.cfg.workspace, f'vis_guide-normal-{name}-back-e.png'), (bb*255).astype(np.uint8))
        del bb, normal_net, in_tensor, mesh_render_temp

        # Human Parsing to produce fliped clothes for back view
        import SCHP.networks as networks
        from SCHP.utils.transforms import transform_logits
        from SCHP.datasets.simple_extractor_dataset import SimpleFolderDataset
        num_classes = 20
        input_size = [473, 473]
        ckpt = os.path.join(os.getenv('WEIGHT_PATH'), 'SCHP/exp-schp-201908261155-lip.pth')
        model = networks.init_model('resnet101', num_classes=num_classes, pretrained=None)
        state_dict = torch.load(ckpt)['state_dict']
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        model.to(self.device)
        model.eval()
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
        ])
        dataset = SimpleFolderDataset(root=path_croped_img, input_size=input_size, transform=transform)
        dataloader = DataLoader(dataset)
        # label = [0'Background', 1'Hat', 2'Hair', 3'Glove', 4'Sunglasses', 5'Upper-clothes', 6'Dress', 7'Coat',
        #           8'Socks', 9'Pants', 10'Jumpsuits', 11'Scarf', 12'Skirt', 13'Face', 14'Left-arm', 15'Right-arm',
        #           16'Left-leg', 17'Right-leg', 18'Left-shoe', 19'Right-shoe']  # 5,6,7,9,10,12
        with torch.no_grad():
            for idx, batch in enumerate(dataloader):
                image, meta = batch
                img_name = meta['name'][0]
                c = meta['center'].numpy()[0]
                s = meta['scale'].numpy()[0]
                w = meta['width'].numpy()[0]
                h = meta['height'].numpy()[0]

                output = model(image.cuda())
                upsample = torch.nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
                upsample_output = upsample(output[0][-1][0].unsqueeze(0))
                upsample_output = upsample_output.squeeze()
                upsample_output = upsample_output.permute(1, 2, 0)  # CHW -> HWC
                logits_result = transform_logits(upsample_output.data.cpu().numpy(), c, s, w, h, input_size=input_size)
                parsing_result = np.argmax(logits_result, axis=2)

                mask_cloth = 1.*((parsing_result==5)+(parsing_result==6)+(parsing_result==7)+(parsing_result==9)+(parsing_result==10)+(parsing_result==12))
                mask_cloth = 1.*(mask_cloth>0)
                mask_face = 1.*(parsing_result==13)
                mask_head = 1.*((parsing_result==1)+(parsing_result==2)+(parsing_result==13))
                mask_upper0 = 1.* (parsing_result==7)
                remove_lining = False
                if (mask_upper0).sum() > 10:
                    remove_lining = True
                    mask_upper0 = cv2.resize(mask_upper0, (self.cfg.width_img, self.cfg.height_img), interpolation=cv2.INTER_AREA)
                    mask_upper0: Float[Tensor, "1 H W 1"] = (torch.from_numpy(mask_upper0 > 0.5).unsqueeze(0).unsqueeze(3).to(self.device)) * 1.
                    _, _, band_upper0 = self.deduce_bbox_from_mask(mask_upper0, padding_x=[0,0], padding_y=[0,0])
                    mask_lining = 1.*((parsing_result==5)+(parsing_result==6)+(parsing_result==10))
                    mask_lining = cv2.resize(mask_lining, (self.cfg.width_img, self.cfg.height_img), interpolation=cv2.INTER_AREA)
                    mask_lining: Float[Tensor, "1 H W 1"] = (torch.from_numpy(mask_lining > 0.5).unsqueeze(0).unsqueeze(3).to(self.device)) * 1.
                    mask_to_remove = mask_lining * band_upper0
                    # mask_upper = 1.*((parsing_result==14)+(parsing_result==15)+(parsing_result==7))
                    mask_upper = 1.*((parsing_result==5)+(parsing_result==6)+(parsing_result==7)+(parsing_result==10)+(parsing_result==14)+(parsing_result==15))
                else:
                    mask_upper = 1.*((parsing_result==5)+(parsing_result==6)+(parsing_result==7)+(parsing_result==10)+(parsing_result==14)+(parsing_result==15))
                mask_lower = 1.*((parsing_result==9)+(parsing_result==12))
                mask_foot = 1.*((parsing_result==8)+(parsing_result==16)+(parsing_result==17)+(parsing_result==18)+(parsing_result==19))

        mask_cloth = cv2.resize(mask_cloth, (self.cfg.width_img, self.cfg.height_img), interpolation=cv2.INTER_AREA)
        mask_face = cv2.resize(mask_face, (self.cfg.width_img, self.cfg.height_img), interpolation=cv2.INTER_AREA)
        mask_head = cv2.resize(mask_head, (self.cfg.width_img, self.cfg.height_img), interpolation=cv2.INTER_AREA)
        mask_upper = cv2.resize(mask_upper, (self.cfg.width_img, self.cfg.height_img), interpolation=cv2.INTER_AREA)
        mask_lower = cv2.resize(mask_lower, (self.cfg.width_img, self.cfg.height_img), interpolation=cv2.INTER_AREA)
        mask_foot = cv2.resize(mask_foot, (self.cfg.width_img, self.cfg.height_img), interpolation=cv2.INTER_AREA)

        # cloth related 
        self.mask_cloth: Float[Tensor, "1 H W 1"] = (
                torch.from_numpy(mask_cloth > 0.5).unsqueeze(0).unsqueeze(3).to(self.device)
            ) * self.mask * 1.
        self.mask_cloth = 1.- 1.*(self.mask_expanding(1-self.mask_cloth, 11)>0.85)
        self.cloth = self.rgb*self.mask_cloth
        self.mask_cloth_flip = torch.flip(self.mask_cloth, [-2])
        self.cloth_flip = torch.flip(self.cloth, [-2])
        name = self.cfg.prompt.replace(' ', '_')
        imageio.imwrite(os.path.join(self.cfg.workspace, f'vis_guide-cloth-{name}-back.png'), (self.cloth_flip[0].cpu().numpy()*255.).astype('uint8'))

        # mask of parts
        mask_face: Float[Tensor, "1 H W 1"] = (torch.from_numpy(mask_face > 0.5).unsqueeze(0).unsqueeze(3).to(self.device)) * 1.
        _, self.bbox_face, self.band_face = self.deduce_bbox_from_mask(mask_face, padding_x=[40,40], padding_y=[35,10])
        rgb_without_face = self.rgb*(1-self.bbox_face)
        # rgb_without_face = self.rgb*(1-mask_face)
        imageio.imwrite(os.path.join(self.cfg.workspace, f'vis_img_noface-{name}.png'), (rgb_without_face[0].cpu().numpy()*255.).astype('uint8'))
        mask_head: Float[Tensor, "1 H W 1"] = (torch.from_numpy(mask_head > 0.5).unsqueeze(0).unsqueeze(3).to(self.device)) * 1.
        mask_upper: Float[Tensor, "1 H W 1"] = (torch.from_numpy(mask_upper > 0.5).unsqueeze(0).unsqueeze(3).to(self.device)) * 1.
        mask_lower: Float[Tensor, "1 H W 1"] = (torch.from_numpy(mask_lower > 0.5).unsqueeze(0).unsqueeze(3).to(self.device)) * 1.
        mask_foot: Float[Tensor, "1 H W 1"] = (torch.from_numpy(mask_foot > 0.5).unsqueeze(0).unsqueeze(3).to(self.device)) * 1.
        if remove_lining:
            mask_upper = mask_upper * (1. - mask_to_remove)
        _, bbox_head, band_head = self.deduce_bbox_from_mask(mask_head, padding_x=[10,10], padding_y=[8,15])
        _, bbox_upper, band_upper = self.deduce_bbox_from_mask(mask_upper, padding_x=[10,10], padding_y=[8,8])
        _, bbox_lower, band_lower = self.deduce_bbox_from_mask(mask_lower, padding_x=[10,10], padding_y=[8,8])
        _, bbox_foot, band_foot = self.deduce_bbox_from_mask(mask_foot, padding_x=[10,10], padding_y=[8,8])
        if remove_lining:
            bbox_upper = bbox_upper * (1. - mask_to_remove)
        mask_ref_head = bbox_head * self.mask
        mask_ref_upper = bbox_upper * self.mask
        mask_ref_lower = bbox_lower * self.mask
        mask_ref_foot = bbox_foot * self.mask
        self.mask_parts_ref = torch.cat([1.-1.*self.mask, mask_ref_head, mask_ref_upper, mask_ref_lower, mask_ref_foot]).permute(0, 3, 1, 2)
        self.band_parts = torch.cat([torch.ones_like(band_head), band_head, band_upper, band_lower, band_foot]).permute(0, 3, 1, 2)
        # vis
        rgb_part = self.rgb*mask_ref_head + torch.ones_like(self.rgb)*(1-mask_ref_head)
        imageio.imwrite(os.path.join(self.cfg.workspace, f'vis_img_0head-{name}.png'), (rgb_part[0].cpu().numpy()*255.).astype('uint8'))
        rgb_part = self.rgb*mask_ref_upper + torch.ones_like(self.rgb)*(1-mask_ref_upper)
        imageio.imwrite(os.path.join(self.cfg.workspace, f'vis_img_1upper-{name}.png'), (rgb_part[0].cpu().numpy()*255.).astype('uint8'))
        rgb_part = self.rgb*mask_ref_lower + torch.ones_like(self.rgb)*(1-mask_ref_lower)
        imageio.imwrite(os.path.join(self.cfg.workspace, f'vis_img_2lower-{name}.png'), (rgb_part[0].cpu().numpy()*255.).astype('uint8'))
        rgb_part = self.rgb*mask_ref_foot + torch.ones_like(self.rgb)*(1-mask_ref_foot)
        imageio.imwrite(os.path.join(self.cfg.workspace, f'vis_img_3foot-{name}.png'), (rgb_part[0].cpu().numpy()*255.).astype('uint8'))



    def mask_expanding(self, mask, k_size=9):
        """
        mask: Torch.Tensor [B, h, w, 1]
        """
        mask = mask.permute(0,3,1,2).detach()
        m = torch.nn.MaxPool2d(k_size, stride=1, padding=int((k_size-1)/2))
        mask_dilated = m(mask)
        return mask_dilated.permute(0,2,3,1)

    def get_c2w(self, elevation_deg, azimuth_deg, camera_distance):
        elevation = elevation_deg.to(self.device) * math.pi / 180
        azimuth = azimuth_deg.to(self.device) * math.pi / 180
        camera_distance = camera_distance.to(self.device)
        # convert spherical coordinates to cartesian coordinates
        # right hand coordinate system, x back, y right, z up
        # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
        camera_position: Float[Tensor, "1 3"] = torch.stack(
            [
                camera_distance * torch.cos(elevation) * torch.cos(azimuth),
                camera_distance * torch.cos(elevation) * torch.sin(azimuth),
                camera_distance * torch.sin(elevation),
            ],
            dim=-1,
        )
        center: Float[Tensor, "1 3"] = torch.zeros_like(camera_position)
        up: Float[Tensor, "1 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32, device=self.device)[None]
        lookat: Float[Tensor, "1 3"] = F.normalize(center - camera_position, dim=-1)
        right: Float[Tensor, "1 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4: Float[Tensor, "1 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_position[:, :, None]],
            dim=-1,
        )
        c2w: Float[Tensor, "1 4 4"] = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
        )
        c2w[:, 3, 3] = 1.0
        return camera_position, c2w

    def deduce_bbox_from_mask(self, masks, padding_x=[0,0], padding_y=[0,0]):
        # masks: BHW1
        bboxs, new_masks, band_masks = [], [], []
        B, H, W, C = masks.shape
        for i in range(B):
            mask = masks[i, :, :, 0]
            coor = torch.nonzero(mask)
            if len(coor) == 0:
                new_mask = torch.zeros_like(masks[i])
                new_masks.append(new_mask)
                band_mask = torch.zeros_like(masks[i])
                band_masks.append(band_mask)
                bboxs.append([0, 0, 0, 0])
            else:
                aa = coor[:, 0].sort(descending=False)
                ymin = aa[0][0].clone()
                ymax = aa[0][-1].clone()
                aa = coor[:, 1].sort(descending=False)
                xmin = aa[0][0].clone()
                xmax = aa[0][-1].clone()
                # perform padding
                xmin = max(0, xmin-padding_x[0])
                xmax = min(W, xmax+padding_x[1])
                ymin = max(0, ymin-padding_y[0])
                ymax = min(H, ymax+padding_y[1])

                new_mask = torch.zeros_like(masks[i])
                new_mask[ymin:ymax+1, xmin:xmax+1] = 1
                new_masks.append(new_mask)

                band_mask = torch.zeros_like(masks[i])
                band_mask[ymin:ymax+1, :] = 1
                band_masks.append(band_mask)

                width = torch.abs(ymax - ymin)
                height = torch.abs(xmax - xmin)
                cx = (xmin + xmax) / 2
                cy = (ymin + ymax) / 2
                bboxs.append([cx, cy, width, height])
        new_masks = torch.stack(new_masks)
        band_masks = torch.stack(band_masks)
        return bboxs, new_masks, band_masks

    
    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        size_ind = bisect.bisect_right(self.resolution_milestones, global_step) - 1
        self.height = self.heights[size_ind]
        self.width = self.widths[size_ind]
        self.directions_unit_focal = self.directions_unit_focals[size_ind]
        if self.cfg.run_local_rendering:
            self.general_origins_local = self.general_origins_locals[size_ind]
        threestudio.debug(f"Training height: {self.height}, width: {self.width}")

    
    def __iter__(self):
        while True:
            yield {}

    
    def get_all_images(self):
        return self.rgb

    
    def collate(self, batch) -> Dict[str, Any]:
        # get directions by dividing directions_unit_focal by focal length
        directions: Float[Tensor, "1 H W 3"] = self.directions_unit_focal.clone()[None]
        
        # Importance note: the returned rays_d MUST be normalized!
        if self.cfg.use_orthograph:
            rays_o, rays_d = get_orthoghphic_rays(
                directions, self.c2w, keepdim=True, noise_scale=self.cfg.rays_noise_scale
            )
            rays_o_back, rays_d_back = get_orthoghphic_rays(
                directions, self.c2w_back, keepdim=True, noise_scale=self.cfg.rays_noise_scale
            )
        else:
            focal_length = 0.5 * self.height / torch.tan(0.5 * self.fovy).to(self.device)
            directions[:, :, :, :2] = directions[:, :, :, :2] / focal_length
            rays_o, rays_d = get_rays(
                directions, self.c2w, keepdim=True, noise_scale=self.cfg.rays_noise_scale
            )
            rays_o_back, rays_d_back = get_rays(
                directions, self.c2w_back, keepdim=True, noise_scale=self.cfg.rays_noise_scale
            )

        # proj_mtx: Float[Tensor, "4 4"] = get_projection_matrix(
        #     self.fovy, self.width / self.height, 0.1, 100.0
        # ).to(self.device)  # FIXME: hard-coded near and far
        # mvp_mtx: Float[Tensor, "4 4"] = get_mvp_matrix(self.c2w.to(self.device), proj_mtx)
        # mvp_mtx_back: Float[Tensor, "4 4"] = get_mvp_matrix(self.c2w_back.to(self.device), proj_mtx)
        batch_back = {
            "rays_o": rays_o_back,
            "rays_d": rays_d_back,
            # "mvp_mtx": mvp_mtx_back,
            # "camera_positions": self.camera_position_back,
            "light_positions": self.camera_position_back,
            "elevation": self.elevation_deg,
            "azimuth": self.azimuth_deg_back,
            "camera_distances": self.camera_distance,
            "height": self.height,
            "width": self.width,
            "c2w": self.c2w_back,
            "mask": self.mask_flip,
            "mask_back": self.mask_back,
            # "depth_back": self.depth_back,
            # "normal_back": self.normal_back,
            "normal_back_e": self.normal_back_e,
            "cloth_ref": self.cloth_flip,
            "cloth_ref_mask": self.mask_cloth_flip,
        }

        batch = {
            "rays_o": rays_o,
            "rays_d": rays_d,
            # "proj_mtx": proj_mtx,
            # "mvp_mtx": mvp_mtx,
            # "camera_positions": self.camera_position,
            "light_positions": self.light_position,
            "elevation": self.elevation_deg,
            "azimuth": self.azimuth_deg,
            "camera_distances": self.camera_distance,
            "height": self.height,
            "width": self.width,
            "c2w": self.c2w,
            "image": self.rgb,
            "mask": self.mask,
            "bbox_face": self.bbox_face,
            "band_face": self.band_face,
            "mask_front": self.mask_front,
            # "depth_front": self.depth,
            # "normal_front": self.normal,
            "normal_front_e": self.normal_e,
            # "mesh_render": self.mesh_render,
            "back_camera": batch_back,
            "mask_parts_ref": self.mask_parts_ref,
            "band_parts": self.band_parts,
        }
        if self.cfg.use_random_camera:
            batch["random_camera"] = self.random_pose_generator.collate(None)

        if self.cfg.run_local_rendering: 
            origins: Float[Tensor, "1 H W 3"] = self.general_origins_local.clone()[None]
            c2w_head = batch["random_camera"]["c2w"].clone()
            c2w_head[:, :3, -1] += torch.tensor([0., 0., self.cent_head_rate]).to(self.device)
            c2w_foot = batch["random_camera"]["c2w"].clone()
            c2w_foot[:, :3, -1] += torch.tensor([0., 0., -self.cent_foot_rate]).to(self.device)
            rays_o_head, rays_d_head = get_orthoghphic_rays(origins, c2w_head, keepdim=True, noise_scale=self.cfg.rays_noise_scale)
            rays_o_foot, rays_d_foot = get_orthoghphic_rays(origins, c2w_foot, keepdim=True, noise_scale=self.cfg.rays_noise_scale)
            batch["head_camera"] = {
                "rays_o": rays_o_head,
                "rays_d": rays_d_head,
                "light_positions": c2w_head[:, :3, -1],
                "elevation": batch["random_camera"]["elevation"],
                "azimuth": batch["random_camera"]["azimuth"],
                "camera_distances": batch["random_camera"]["camera_distances"],
                "height": batch["random_camera"]["height"],
                "width": batch["random_camera"]["width"],
                "c2w": c2w_head,
                "image": self.rgb_head,
                "mask": self.mask_head,
            }
            batch["foot_camera"] = {
                "rays_o": rays_o_foot,
                "rays_d": rays_d_foot,
                "light_positions": c2w_foot[:, :3, -1],
                "elevation": batch["random_camera"]["elevation"],
                "azimuth": batch["random_camera"]["azimuth"],
                "camera_distances": batch["random_camera"]["camera_distances"],
                "height": batch["random_camera"]["height"],
                "width": batch["random_camera"]["width"],
                "c2w": c2w_foot,
                "image": self.rgb_foot,
                "mask": self.mask_foot,
            }

        return batch


@register("humanref-datamodule")
class HumanDF_DataModule(pl.LightningDataModule):
    cfg: SingleImageDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(SingleImageDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = SingleImageIterableDataset(self.cfg)
        if stage in [None, "fit", "validate"]:
            self.val_dataset = SingleImageRandomDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = SingleImageRandomDataset(self.cfg, "test")

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
        return DataLoader(
            dataset,
            # very important to disable multi-processing if you want to change self attributes at runtime!
            # (for example setting self.width and self.height in update_step)
            num_workers=0,  # type: ignore
            batch_size=batch_size,
            collate_fn=collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.val_dataset, batch_size=1, collate_fn=self.val_dataset.collate
        )

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )

