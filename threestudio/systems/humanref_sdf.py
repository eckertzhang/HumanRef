import os
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from mesh_renderer.meshrender import MeshRender
import clip
import random
import imageio, cv2
import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.misc import cleanup, get_device
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *

# pip install apscheduler
# from apscheduler.schedulers.background import BackgroundScheduler


@threestudio.register("humanref-system-sdf")
class HumanRef(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        
        stage: str = "coarse"
        warm_up_iters: int = 500
        albedo_iters: int = 6000
        max_iters: int = 10000
        visualize_samples: bool = False
        with_clip_loss: bool = False
        with_clip_loss_ref: bool = False
        with_clip_loss_pidm: bool = False
        with_clip_loss_unclip: bool = False
        use_pidm_constrain: bool = False
        use_style_loss: bool = False
        use_multi_denoise: bool = True
        run_initial: bool = True
        run_local_rendering: bool = False
        attention_strategy: int = 1

    cfg: Config

    def configure(self) -> None:
        # set up geometry, material, background, renderer
        super().configure()

        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(self.cfg.prompt_processor)
        self.prompt_utils = self.prompt_processor()

        # CLIP model for calculating clip loss
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/16", device=self.device, jit=False, download_root=os.path.join(os.getenv('WEIGHT_PATH'), 'clip'))
        self.aug = T.Compose([
            T.Resize((224, 224)),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        for p in self.clip_model.parameters():
            p.requires_grad = False
        
        # perceptual loss
        from third_parties.lpips.lpips import LPIPS
        self.loss_fn_vgg = LPIPS(
            net='vgg', pretrained=True, 
            model_path='./third_parties/lpips/weights/v0.1/vgg.pth',
            net_path=os.path.join(os.getenv('WEIGHT_PATH'), 'LPIPS/vgg16-397923af.pth'),
            ).to(self.device)
        self.mseloss = nn.MSELoss()
        self.pearson = None
        self.std2mesh = None
        self.direction_prompts = [
            "in the front view", 
            "in the back view, no face", 
            "in the side view",]
        self.direction_embs = None
        self.image_emb = None
        self.text_latents = {}
        self.bg_white = torch.tensor([[1.,1.,1.]], device=self.device).expand(1, 512, 512, 3).contiguous()

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        # if self.cfg.stage == "geometry":
        #     render_out = self.renderer(**batch, render_normal=True, render_rgb=False)
        # else:
        render_out = self.renderer(**batch)
        return {
            **render_out,
        }
    
    def l2_loss(self, rgb1, rgb2, mask=None):
        if mask is None:
            l2_loss = self.mseloss(rgb1, rgb2)
        else:
            l2_loss = self.mseloss(rgb1[mask>0.5], rgb2[mask>0.5])
        return l2_loss
    
    def perceptual_loss(self, rgb1, rgb2, mask=None, normalize=False):
        "image should be RGB [B, H, W, 3], IMPORTANT: normalized to [-1,1]"
        if mask is None:
            loss = self.loss_fn_vgg(rgb1.permute(0, 3, 1, 2), rgb2.permute(0, 3, 1, 2), normalize=normalize)
        else:
            rgb1 = (rgb1*mask).permute(0, 3, 1, 2)
            rgb2 = (rgb2*mask).permute(0, 3, 1, 2)
            loss = self.loss_fn_vgg(rgb1, rgb2, retPerLayer=False, normalize=normalize)
        return loss
    
    def pearson_depth_loss(self, pred_depth, depth_gt, mask):
        if self.pearson is None:
            from torchmetrics import PearsonCorrCoef
            self.pearson = PearsonCorrCoef()
        pred_depth = torch.nan_to_num(pred_depth)
        co = self.pearson(pred_depth[mask>0.5], depth_gt[mask>0.5])
        return 1 - co 
    
    def neg_iou_loss(self, predict, target):
        dims = tuple(range(predict.ndimension())[1:])
        intersect = (predict * target).sum(dims)
        union = (predict + target - predict * target).sum(dims) + 1e-6
        return 1. - (intersect / union).sum() / intersect.nelement()
    
    def img_clip_loss(self, rgb1, rgb2):
        image_z_1 = self.clip_model.encode_image(self.aug(rgb1))
        image_z_2 = self.clip_model.encode_image(self.aug(rgb2))
        image_z_1 = image_z_1 / image_z_1.norm(dim=-1, keepdim=True) # normalize features
        image_z_2 = image_z_2 / image_z_2.norm(dim=-1, keepdim=True) # normalize features

        loss = - (image_z_1 * image_z_2).sum(-1).mean()
        return loss
    
    def img_text_clip_loss(self, rgb, prompt, view_dir=None):
        image_z_1 = self.clip_model.encode_image(self.aug(rgb))
        image_z_1 = image_z_1 / image_z_1.norm(dim=-1, keepdim=True) # normalize features

        if view_dir in self.text_latents:
            text_z = self.text_latents[view_dir]
        else:
            text = clip.tokenize(prompt).to(self.device)
            text_z = self.clip_model.encode_text(text)
            text_z = text_z / text_z.norm(dim=-1, keepdim=True)
            self.text_latents[view_dir] = text_z.detach()
        loss = - (image_z_1 * text_z).sum(-1).mean()
        return loss

    def view_matching(self, azimuth, elevation):
        if azimuth >= -15 and azimuth < 15:
            indx = 0
            dir_view = ', front view'
        elif (azimuth >= -180 and azimuth < -165) or (azimuth >= 165 and azimuth <= 180):
            indx = 1
            dir_view = ', back view, no face'
        elif azimuth >= 75 and azimuth < 105:
            indx = 2
            dir_view = ', left view'
        elif azimuth >= -105 and azimuth < -75:
            indx = 3
            dir_view = ', right view'
        elif azimuth >= 15 and azimuth < 75:
            indx = 4
            dir_view = ', left front view'
        elif azimuth >= 105 and azimuth < 165:
            indx = 5
            dir_view = ', left back view, no face'
        elif azimuth >= -75 and azimuth < -15:
            indx = 6
            dir_view = ', right front view'
        elif azimuth >= -165 and azimuth < -105:
            indx = 7
            dir_view = ', right back view, no face'
        return indx, dir_view

    def on_fit_start(self) -> None:
        super().on_fit_start()
        if 'implicit-sdf' in self.cfg.geometry_type:
            # initialize SDF
            self.geometry.initialize_shape(run_init=self.cfg.run_initial)
            # prune grid using initilized SDF
            self.renderer.update_step(0, 0, run_init=self.cfg.run_initial)

    def warp_mtx_on_std(self, ang_x_deg, ang_y_deg, ang_z_deg):
        ang_x = torch.tensor(ang_x_deg / 180, device=self.device) * torch.pi
        ang_y = torch.tensor(ang_y_deg / 180, device=self.device) * torch.pi 
        ang_z = torch.tensor(ang_z_deg / 180, device=self.device) * torch.pi
        r_x = torch.tensor([
            [1, 0, 0],
            [0, torch.cos(ang_x), torch.sin(ang_x)],
            [0, -torch.sin(ang_x), torch.cos(ang_x)]
        ], device=self.device)
        r_y = torch.tensor([
            [torch.cos(ang_y), 0, -torch.sin(ang_y)],
            [0, 1, 0],
            [torch.sin(ang_y), 0, torch.cos(ang_y)]
        ], device=self.device)
        r_z = torch.tensor([
            [torch.cos(ang_z), torch.sin(ang_z), 0],
            [-torch.sin(ang_z), torch.cos(ang_z), 0],
            [0, 0, 1]
        ], device=self.device)
        return torch.mm(r_z, torch.mm(r_y, r_x))

    def training_step(self, batch, batch_idx):
        rate_denomi = 20
        is_front_view, is_back_view, use_clip_loss, use_cloth_loss = False, False, False, False
        use_multi_denoise, use_diffusion_prior, use_local_rendering, run_local_rendering = False, False, False, False
        rendered_region = 'full'
        with_clip_text_loss = True
        w_normal = 1.0

        if self.global_step < 2000:
            use_clip_loss = True
            rate_denomi = 10
            # use_cloth_loss = True
        if self.global_step >= 3000:
            self.text_latents = {}
            w_normal = 10.
        if self.global_step >= int(0.6*self.cfg.max_iters):
            w_normal = 100.
            # use_local_rendering = self.cfg.run_local_rendering
        if self.global_step >= self.cfg.warm_up_iters:
            use_diffusion_prior = True
        
        if self.global_step >= int(0.8*self.cfg.max_iters):
            use_multi_denoise = self.cfg.use_multi_denoise

        rgb_ref_init = batch["image"]
        rgb_ref = rgb_ref_init.clone()
        mask_ref = batch["mask"]*1.
        bbox_face = batch["bbox_face"]
        band_face = batch["band_face"]
        mask_parts_ref = batch["mask_parts_ref"]
        band_parts = batch["band_parts"]
        if batch_idx % rate_denomi == 0:
            is_front_view = True
            normal_ref = batch["normal_front_e"]
            mask = mask_ref
            normal_ref = normal_ref
        elif batch_idx % rate_denomi == 1:
            is_back_view = True
            batch = batch["back_camera"]
            normal_ref = batch["normal_back_e"]
            mask = batch["mask"]*1.
            normal_ref = normal_ref
        else:
            if use_local_rendering and use_diffusion_prior:
                rand = random.random()
                if rand > 0.9: 
                    run_local_rendering = True
                    rendered_region = 'head'
                    batch = batch["head_camera"]
                    rgb_ref_init = batch["image"]
                    rgb_ref = rgb_ref_init.clone()
                    mask_ref = batch["mask"]*1.
                elif rand < 0.1:
                    run_local_rendering = True
                    rendered_region = 'foot'
                    batch = batch["foot_camera"]
                    rgb_ref_init = batch["image"]
                    rgb_ref = rgb_ref_init.clone()
                    mask_ref = batch["mask"]*1.
                else:
                    batch = batch["random_camera"]
            else:
                batch = batch["random_camera"]

        # randomly change background color
        bs = batch["rays_o"].shape[0]
        bg_color = torch.rand(bs, 3, device=self.device)
        bg_img = bg_color.expand(1, 512, 512, 3).contiguous()
        batch["bg_color"] = bg_color
        rgb_ref = rgb_ref * mask_ref + bg_img * (1 - mask_ref)

        # perform rendering
        batch["stage"] = self.cfg.stage
        out = self(batch)

        # prepare for loss calculation
        loss = 0.0
        guidance_inp = out["comp_rgb"]
        pred_rgb = F.interpolate(guidance_inp.permute(0,3,1,2), (512, 512), mode='bilinear', align_corners=True).permute(0,2,3,1)
        pred_mask = F.interpolate(out['opacity'].permute(0,3,1,2), (512, 512), mode='bilinear', align_corners=True).permute(0,2,3,1)
        pred_normal = F.interpolate((out['comp_normal']*2.-1.).permute(0,3,1,2), (512, 512), mode='bilinear', align_corners=True).permute(0,2,3,1)

        ## 01 front view: RGB reference loss (L2)
        if is_front_view:
            loss_ref = self.cfg.loss.lambda_img * self.l2_loss(pred_rgb, rgb_ref)
            # self.log("train/loss_ref", loss_ref.item())
            loss += loss_ref
        
        ## 02 back view: cloth RGB reference loss (optional, perceptual)
        if is_back_view and use_cloth_loss and self.cfg.loss.lambda_back_cloth>0:
            cloth_ref = batch["cloth_ref"]
            cloth_ref_mask = batch["cloth_ref_mask"]*1.
            loss_cloth_ref = self.cfg.loss.lambda_back_cloth * self.perceptual_loss(pred_rgb, cloth_ref, cloth_ref_mask, normalize=True)[0,0,0,0]
            loss += loss_cloth_ref

        ## 03 CLIP Loss
        if use_clip_loss and not is_front_view:
            # remove face influence
            rgb_ref_noface = rgb_ref * (1. - bbox_face) + bg_img * bbox_face
            band_face = pred_mask * band_face
            pred_rgb_nohead = pred_rgb * (1. - band_face) + bg_img * band_face
            loss_ref = self.cfg.loss.lambda_clip * self.img_clip_loss(pred_rgb_nohead.permute(0,3,1,2), rgb_ref_noface.permute(0,3,1,2)) 
            if with_clip_text_loss:
                direction_prompt, view_dirs = [''], ['']
                for d in self.prompt_utils.directions:
                    if d.condition(batch["elevation"], batch["azimuth"], batch["camera_distances"]):
                        direction_prompt.append(f', {d.name} view' if d.name != 'back' and d.name != 'side rear' else f', {d.name} view, no face')
                        view_dirs.append(d.name)
                text = self.cfg.prompt_processor.prompt + direction_prompt[-1]
                loss_ref += self.cfg.loss.lambda_clip * self.img_text_clip_loss(pred_rgb.permute(0,3,1,2), text, view_dirs[-1])
            # self.log("train/loss_ref", loss_ref.item())
            loss += loss_ref

        ## 04 Multiple steps denoise loss
        if use_multi_denoise and not is_front_view and not run_local_rendering:
            if self.cfg.attention_strategy == 0 or self.cfg.attention_strategy == 1:
                mask_image_ref = mask_parts_ref
                mask_image_ini = torch.cat([1. - pred_mask.detach(), pred_mask.detach(), pred_mask.detach(), pred_mask.detach(), pred_mask.detach()]).permute(0, 3, 1, 2)
                mask_image_ini *= band_parts
            elif self.cfg.attention_strategy == 2:
                mask_image_ref = torch.cat([1.-mask_ref, mask_ref]).permute(0, 3, 1, 2)
                mask_image_ini = torch.cat([1. - pred_mask.detach(), pred_mask.detach()]).permute(0, 3, 1, 2)
            else:
                mask_image_ref, mask_image_ini = None, None
            with torch.no_grad():
                img_denoised = self.guidance.multiple_step_denoise(
                    rgb=pred_rgb, 
                    ref_image=rgb_ref, 
                    t_max_rate=0.2,
                    num_steps_total=30,
                    prompt_utils=self.prompt_utils, **batch,
                    rendered_region=rendered_region,
                    mask_image_ref=mask_image_ref,
                    mask_image_ini=mask_image_ini,
                ).permute(0,2,3,1)
            loss_multi = self.cfg.loss.lambda_multi * self.perceptual_loss(pred_rgb, img_denoised, normalize=True)[0,0,0,0]
            # self.log("train/loss_multi", loss_multi.item())
            loss += loss_multi

        # 05 GT Normal guidance
        if is_front_view or is_back_view:
            loss_normal = w_normal * self.cfg.loss.lambda_normal * self.l2_loss(pred_normal, normal_ref, mask[:,:,:,0])
            # self.log("train/loss_normal", loss_normal.item())
            loss += loss_normal

        # 06 IoU Loss
        if is_front_view or is_back_view:
            loss_iou = self.cfg.loss.lambda_IoU * self.neg_iou_loss(pred_mask, mask)
            # self.log("train/loss_iou", loss_iou.item())
            loss += loss_iou
        
        # 08 SDF geometric prior loss
        if 'sdf_loss' in out and self.cfg.loss.lambda_sdf > 0:
            loss_sdf = self.cfg.loss.lambda_sdf * out['sdf_loss']
            # self.log("train/loss_sdf", loss_sdf.item())
            loss += loss_sdf
        
        # 09 SDF geometric smooth loss
        if 'sdf_smooth_loss' in out and self.cfg.loss.lambda_sdf_smooth > 0:
            loss_sdf_sm = self.cfg.loss.lambda_sdf_smooth * out['sdf_smooth_loss']
            # self.log("train/loss_sdf", loss_sdf_sm.item())
            loss += loss_sdf_sm

        # 2D normal smoothness
        if "comp_normal" in out:
            normal = out["comp_normal"]
            loss_smooth = (normal[:, 1:, :, :] - normal[:, :-1, :, :]).square().mean() \
                            + (normal[:, :, 1:, :] - normal[:, :, :-1, :]).square().mean()
            # self.log("train/loss_2Dnormal", loss_smooth)
            loss += loss_smooth * self.cfg.loss.lambda_2d_normal_smooth

        # 3D normal smooth
        if "normal" in out and "normal_perturb" in out:
            normals = out["normal"]
            normals_perturb = out["normal_perturb"]
            loss_smooth_3d = (normals - normals_perturb).abs().mean()
            # self.log("train/loss_3Dnormal", loss_smooth_3d)
            loss += loss_smooth_3d * self.cfg.loss.lambda_3d_normal_smooth
        
        ## Ref-SDS loss
        if not is_front_view and use_diffusion_prior:
            if (self.cfg.attention_strategy == 0 or self.cfg.attention_strategy == 1) and not run_local_rendering:
                mask_image_ref = mask_parts_ref
                mask_image_ini = torch.cat([1. - pred_mask.detach(), pred_mask.detach(), pred_mask.detach(), pred_mask.detach(), pred_mask.detach()]).permute(0, 3, 1, 2)
                mask_image_ini *= band_parts
            elif self.cfg.attention_strategy == 2 and not run_local_rendering:
                mask_image_ref = torch.cat([1.-mask_ref, mask_ref]).permute(0, 3, 1, 2)
                mask_image_ini = torch.cat([1. - pred_mask.detach(), pred_mask.detach()]).permute(0, 3, 1, 2)
            else:
                mask_image_ref, mask_image_ini = None, None
            guidance_out = self.guidance(
                rgb=pred_rgb, 
                ref_image=rgb_ref,
                prompt_utils=self.prompt_utils, **batch, 
                rgb_as_latents=False,
                ref_rgb=rgb_ref.permute(0,3,1,2), 
                ref_text=[text, view_dirs[-1]] if use_clip_loss and with_clip_text_loss else [self.cfg.prompt_processor.prompt, ''], 
                clip_model=self.clip_model,
                with_clip_loss=use_clip_loss,
                rendered_region=rendered_region,
                with_clip_text_loss=with_clip_text_loss,
                mask_image_ref=mask_image_ref,
                mask_image_ini=mask_image_ini,
            )
            for name, value in guidance_out.items():
                # self.log(f"train/{name}", value)
                if name.startswith("loss_"):
                    loss += value * self.cfg.loss[name.replace("loss_", "lambda_")]
    
        # Eikonal loss
        if 'sdf_eikonal_loss' in out and self.cfg.loss.lambda_eikonal > 0:
            loss_eikonal = self.cfg.loss.lambda_eikonal * out['sdf_eikonal_loss']
            loss += loss_eikonal
            self.log("train/loss_eikonal", loss_eikonal.item())

        # # sparsity (opacity) loss
        # if self.cfg.loss.lambda_sparsity > 0:
        #     loss_sparsity = (out["opacity"] ** 2 + 0.001).sqrt().mean()
        #     self.log("train/loss_sparsity", loss_sparsity)
        #     if abs(batch['azimuth']) > 135:
        #         loss += loss_sparsity * self.cfg.loss.lambda_sparsity * 10
        #     else:
        #         loss += loss_sparsity * self.cfg.loss.lambda_sparsity

        # # entropy (opaque) loss
        # if self.cfg.loss.lambda_opaque > 0:
        #     opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
        #     loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
        #     self.log("train/loss_opaque", loss_opaque)
        #     if self.global_step < self.cfg.warm_up_iters:
        #         loss += loss_opaque * self.cfg.loss.lambda_opaque
        #     else:
        #         loss += loss_opaque * self.cfg.loss.lambda_opaque * 10

        # # z variance loss
        # if "z_variance" in out and self.cfg.loss.lambda_z_variance > 0:
        #     # z variance loss: proposed in HiFA: http://arxiv.org/abs/2305.18766 helps reduce floaters and produce solid geometry
        #     loss_z_variance = out["z_variance"][out["opacity"] > 0.5].mean()
        #     self.log("train/loss_z_variance", loss_z_variance)
        #     loss += loss_z_variance * self.cfg.loss.lambda_z_variance
                
        # for name, value in self.cfg.loss.items():
        #     self.log(f"train_params/{name}", value)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        if "mesh_render" in batch:
            mesh_render = batch["mesh_render"]
            c2w = batch["c2w"]
            with torch.no_grad():
                depth_ref, normal_ref, mask = mesh_render.get_depth_normal(c2w[0], bg="gray")
            depth_ref, normal_ref, mask = depth_ref.unsqueeze(-1).to(self.device), ((normal_ref+1.)*0.5).permute(0,2,3,1).to(self.device), mask.unsqueeze(-1).to(self.device)
        bg_white = torch.tensor([[1.,1.,1.]], device=self.device).expand(out["comp_rgb"].shape).contiguous()
        bg_white = bg_white.to(out["comp_rgb"].device)
        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + (
                [
                    {
                        "type": "grayscale",
                        "img": out["opacity"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    },
                ]
                if "opacity" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": normal_ref[0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                ]
                if "mesh_render" in batch
                else []
            ),
            name="validation_step",
            step=self.true_global_step,
        )

        if self.cfg.visualize_samples:
            self.save_image_grid(
                f"it{self.true_global_step}-{batch['index'][0]}-sample.png",
                [
                    {
                        "type": "rgb",
                        "img": self.guidance.sample(
                            self.prompt_utils, **batch, seed=self.global_step
                        )[0],
                        "kwargs": {"data_format": "HWC"},
                    },
                    {
                        "type": "rgb",
                        "img": self.guidance.sample_lora(self.prompt_utils, **batch)[0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ],
                name="validation_step_samples",
                step=self.true_global_step,
            )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        out = self(batch)
        if "mesh_render" in batch:
            mesh_render = batch["mesh_render"]
            c2w = batch["c2w"]
            with torch.no_grad():
                depth_ref, normal_ref, mask = mesh_render.get_depth_normal(c2w[0], bg="gray")
            depth_ref, normal_ref, mask = depth_ref.unsqueeze(-1).to(self.device), ((normal_ref+1.)*0.5).permute(0,2,3,1).to(self.device), mask.unsqueeze(-1).to(self.device)
        self.bg_white = torch.tensor([[1.,1.,1.]], device=self.device).expand(out["comp_rgb"].shape).contiguous()
        self.bg_white = self.bg_white.to(out["comp_rgb"].device)
        self.save_image_grid(
            f"it{self.true_global_step}-test-rgb/{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": (out["comp_rgb"]*out["opacity"]+self.bg_white*(1.-out["opacity"]))[0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            ),
            name="test_step",
            step=self.true_global_step,
        )
        self.save_image_grid(
            f"it{self.true_global_step}-test-normal/{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": (out["comp_normal"]*out["opacity"]+self.bg_white*(1.-out["opacity"]))[0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            ),
            name="test_step",
            step=self.true_global_step,
        )
        if "mesh_render" in batch:
            self.save_image_grid(
                f"it{self.true_global_step}-test-smpl/{batch['index'][0]}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": normal_ref[0], # (normal_ref*mask+self.bg_white*(1.-mask))[0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        },
                    ]
                ),
                name="test_step",
                step=self.true_global_step,
            )
        self.save_image_grid(
            f"it{self.true_global_step}-test/{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            +(
                [
                    {
                        "type": "grayscale",
                        "img": out["opacity"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    },
                ]
                if "opacity" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": normal_ref[0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                ]
                if "mesh_render" in batch
                else []
            ),
            name="test_step",
            step=self.true_global_step,
        )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test-rgb",
            f"it{self.true_global_step}-test-rgb",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
        )
        self.save_img_sequence(
            f"it{self.true_global_step}-test-normal",
            f"it{self.true_global_step}-test-normal",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
        )
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )
