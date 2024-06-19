# This code is based on the implementation of 'reference-only':
# https://github.com/Mikubill/sd-webui-controlnet/discussions/1236

from dataclasses import dataclass, field
import clip, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from diffusers import DDIMScheduler, DDPMScheduler, UniPCMultistepScheduler, StableDiffusionPipeline
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import randn_tensor
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.attention_processor import Attention
from diffusers.models.unet_2d_blocks import CrossAttnDownBlock2D, CrossAttnUpBlock2D, DownBlock2D, UpBlock2D
from tqdm import tqdm

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.ops import perpendicular_component
from threestudio.utils.typing import *

CFG, MODE, uc_mask = None, None, None
strategy = 3

def torch_dfs(model: torch.nn.Module):
    result = [model]
    for child in model.children():
        result += torch_dfs(child)
    return result

@threestudio.register("stable-diffusion-clip-guidance-refsds")
class StableDiffusionReferenceGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5"
        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        guidance_scale: float = 7.0
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98
        max_step_percent_annealed: float = 0.5
        anneal_start_step: Optional[int] = None

        weighting_strategy: str = "sds"

        token_merging: bool = False
        token_merging_params: Optional[dict] = field(default_factory=dict)

        view_dependent_prompting: bool = True

        do_classifier_free_guidance: bool = True
        attention_auto_machine_weight: float = 1.0,
        gn_auto_machine_weight: float = 1.0,
        style_fidelity: float = 0.5
        reference_attn: bool = True
        reference_adain: bool = True
        attention_strategy: int = 3

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Stable Diffusion ...")

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        pipe_kwargs = {
            "tokenizer": None,
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
        }
        self.pipe = StableDiffusionPipeline.from_pretrained(
            os.path.join(os.getenv('WEIGHT_PATH'), self.cfg.pretrained_model_name_or_path),
            **pipe_kwargs,
        ).to(self.device)

        if self.cfg.enable_memory_efficient_attention:
            if parse_version(torch.__version__) >= parse_version("2"):
                threestudio.info(
                    "PyTorch2.0 uses memory efficient attention by default."
                )
            elif not is_xformers_available():
                threestudio.warn(
                    "xformers is not available, memory efficient attention is not enabled."
                )
            else:
                self.pipe.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)

        del self.pipe.text_encoder
        cleanup()

        # Create model
        self.vae = self.pipe.vae.eval()
        self.unet = self.pipe.unet.eval()

        ##### !!!! Modify self attention and group norm
        global CFG, MODE, uc_mask, strategy
        CFG = self.cfg
        MODE = "write"
        uc_mask = (
            torch.Tensor([0] + [1]).to(self.device)
            .bool()
        )
        strategy = self.cfg.attention_strategy

        def hacked_basic_transformer_inner_forward(
            self,
            hidden_states: torch.FloatTensor,  # image latent
            attention_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,  # text_emb
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            class_labels: Optional[torch.LongTensor] = None,
        ):
            
            # (Note! Eckert) resampling attention_mask, its shape should be [B, 1, key_tokens], here the 'key_tokens' should 
            # be same as the number of 'key_tokens' in hidden_states. However, hidden_states (image latent) will be 
            # down/upsampling during Unet process.
            # Thus, we adaptively adjust the 'key_tokens' of attention_mask according to: key_tokens = h*w
            # Eckert: Here, the 0 channel of attention_mask is defaultly assumed as background channel.
            if attention_mask is not None:
                target_length = hidden_states.shape[1]
                height, width = int(np.sqrt(target_length)), int(np.sqrt(target_length))
                batch = attention_mask.shape[0]
                key_tokens = attention_mask.shape[-1]
                min_v = attention_mask.min()
                h, w = int(np.sqrt(key_tokens)), int(np.sqrt(key_tokens))
                attention_mask = attention_mask.reshape(batch, 1, h, w) / min_v
                attention_mask = F.interpolate(attention_mask, size=(height, width)).view(batch, 1, -1)
                # attention_mask[:1] = (1. - 1. * (attention_mask[1:].sum(dim=0, keepdim=True)>=0.5)).to(hidden_states.dtype)
                attention_mask = (1. * (attention_mask>=0.5) * min_v).to(hidden_states.dtype)
            
            
            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
            else:
                norm_hidden_states = self.norm1(hidden_states)

            # 1. Self-Attention
            cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
            if MODE == "write":
                self.bank.append(norm_hidden_states.detach().clone())
                attn_output = self.attn1(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                )
                if attention_mask is not None:
                    self.bankmask.append(attention_mask.detach().clone())
            if MODE == "read":
                if CFG.attention_auto_machine_weight > self.attn_weight:
                    attn_output_uc = self.attn1(
                        norm_hidden_states,
                        encoder_hidden_states=torch.cat([norm_hidden_states] + self.bank, dim=1),
                        attention_mask=torch.cat([attention_mask] + self.bankmask, dim=-1) if attention_mask is not None else attention_mask,
                        **cross_attention_kwargs,
                    )
                    attn_output_c = attn_output_uc.clone()
                    if CFG.do_classifier_free_guidance and CFG.style_fidelity > 0:
                        # Eckert: same as self-attention, Here implement a seld-attn in the 「unconditional」 chennel 
                        attn_output_c[uc_mask] = self.attn1(
                            norm_hidden_states[uc_mask],
                            encoder_hidden_states=norm_hidden_states[uc_mask],
                            attention_mask=attention_mask,
                            **cross_attention_kwargs,
                        )       
                    attn_output = CFG.style_fidelity * attn_output_c + (1.0 - CFG.style_fidelity) * attn_output_uc
                    self.bank.clear()
                    self.bankmask.clear()
                else:
                    attn_output = self.attn1(
                        norm_hidden_states,
                        encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                        attention_mask=attention_mask,
                        **cross_attention_kwargs,
                    )
                    self.bank.clear()
                    self.bankmask.clear()
            if self.use_ada_layer_norm_zero:
                attn_output = gate_msa.unsqueeze(1) * attn_output
            hidden_states = attn_output + hidden_states

            # 2. Cross-Attention
            if self.attn2 is not None:
                norm_hidden_states = (
                    self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
                )

                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                hidden_states = attn_output + hidden_states

            # 3. Feed-forward
            norm_hidden_states = self.norm3(hidden_states)

            if self.use_ada_layer_norm_zero:
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

            ff_output = self.ff(norm_hidden_states)

            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output

            hidden_states = ff_output + hidden_states

            return hidden_states

        def hacked_get_attention_scores_inner_attention(
            self, query: torch.Tensor, key: torch.Tensor, attention_mask: torch.Tensor = None
        ) -> torch.Tensor:
            r"""
            Compute the attention scores.

            Args:
                query (`torch.Tensor`): The query tensor.
                key (`torch.Tensor`): The key tensor.
                attention_mask (`torch.Tensor`, *optional*): The attention mask to use. If `None`, no mask is applied.

            Returns:
                `torch.Tensor`: The attention probabilities/scores.
            """
            dtype = query.dtype
            if self.upcast_attention:
                query = query.float()
                key = key.float()

            baddbmm_input = torch.empty(
                query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
            )
            beta = 0

            attention_scores = torch.baddbmm(
                baddbmm_input,
                query,
                key.transpose(-1, -2),
                beta=beta,
                alpha=self.scale,
            )
            del baddbmm_input

            if self.upcast_softmax:
                attention_scores = attention_scores.float()

            attention_probs = attention_scores.softmax(dim=-1)
            del attention_scores
            # import imageio
            # imageio.imwrite('test.png', ((attention_probs[2]/attention_probs[2].max()).detach().cpu().numpy()*255.).astype(np.uint8)[:,:,np.newaxis].repeat(3, -1))

            ## we perform mask-aware calculation after softmax
            if attention_mask is not None:             
                batch, _, ch, length = attention_mask.shape
                heads, l_q, l_k = attention_probs.shape
                assert length>=l_q and length==l_k
                if strategy == 0:
                    mask_q = attention_mask[..., :l_q]   # self mask
                    mask_k = attention_mask[..., -l_k:]  # ref mask
                    mask = torch.zeros_like(attention_probs).reshape(heads, -1)
                    for i in range(batch):
                        indx_q = (mask_q[i] == 1).nonzero()[:, -1]
                        indx_k = (mask_k[i] == 1).nonzero()[:, -1]
                        ind_mask = torch.stack(torch.meshgrid(indx_q, indx_k))
                        ind_mask = ind_mask.permute(1,2,0).reshape(-1, 2)
                        ind_mask = ind_mask[:, 0] * l_q + ind_mask[:, 1]
                        mask[:, ind_mask] = 1.
                        del indx_q, indx_k, ind_mask
                    mask = mask.reshape(attention_probs.shape).to(dtype)
                    mask[mask==0] = 0.3
                    attention_probs *= mask
                    del mask
                    attention_probs /= (attention_probs.sum(dim=-1, keepdim=True) + 1e-6)
                
                elif strategy == 1 or strategy == 2:
                    mask_q = attention_mask[..., :l_q]   # self mask
                    mask_k = attention_mask[..., -l_q:]  # ref mask
                    if l_q == l_k:
                        probs_ref = attention_probs
                    else:
                        probs_self, probs_ref = attention_probs.split([l_q, l_q], dim=-1)
                    mask = torch.zeros_like(probs_ref).reshape(heads, -1)
                    for i in range(batch):
                        try:
                            indx_q = (mask_q[i] == 1).nonzero()[:, -1]
                            # if i == 0:
                            #     indx_k = (mask_k[i] <= 1).nonzero()[:, -1]
                            # else:
                            indx_k = (mask_k[i] == 1).nonzero()[:, -1]
                            ind_mask = torch.stack(torch.meshgrid(indx_q, indx_k))
                            ind_mask = ind_mask.permute(1,2,0).reshape(-1, 2)
                            ind_mask = ind_mask[:, 0] * l_q + ind_mask[:, 1]
                            mask[:, ind_mask] = 1.
                            del indx_q, indx_k, ind_mask
                        except:
                            continue
                    mask = 3. * mask.reshape(probs_ref.shape).to(dtype)
                    probs_ref *= mask
                    # import imageio
                    # imageio.imwrite('test_msk.png', ((mask[2]).detach().cpu().numpy()*255.).astype(np.uint8)[:,:,np.newaxis].repeat(3, -1))
                    del mask

                    # Normalize scaling
                    if l_q == l_k:
                        attention_probs = probs_ref
                    else:
                        attention_probs = torch.cat([probs_self, probs_ref], dim=-1)
                    attention_probs /= (attention_probs.sum(dim=-1, keepdim=True) + 1e-6)

            attention_probs = torch.nan_to_num(attention_probs)
            attention_probs = attention_probs.to(dtype)

            return attention_probs
       
        def hacked_prepare_attention_mask_inner_attention(
            self, attention_mask: torch.Tensor, target_length: int, batch_size: int, out_dim: int = 3
        ) -> torch.Tensor:
            r"""
            Prepare the attention mask for the attention computation. Assume that attention_mask is a attention_bias.
            Here, we convert bias to the normal mask. With it, we will perform region-aware attention in function 'get_attention_scores'

            Args:
                attention_mask (`torch.Tensor`):
                    The attention mask to prepare. shape: [batch, 1, tokens]
                target_length (`int`):
                    The target length of the attention mask. This is the length of the attention mask after padding.
                batch_size (`int`):
                    The batch size, which is used to repeat the attention mask.
                    Here, batch_size is the batch size of target tokens.
                out_dim (`int`, *optional*, defaults to `3`):
                    The output dimension of the attention mask. Can be either `3` or `4`.

            Returns:
                `torch.Tensor`: The prepared attention mask.
            """
            head_size = self.heads
            if attention_mask is None:
                return attention_mask

            current_length: int = attention_mask.shape[-1]
            intype = attention_mask.dtype
            # convert it to normal mask
            attention_mask = 1. - attention_mask / attention_mask.min()  
            assert current_length==target_length, 'Note, the num of tokens between attention_mask and hidden_states should be same!'
            # if current_length != target_length:
            #     # (Note! Eckert) resampling attention_mask, its shape should be [B, 1, key_tokens], here the 'key_tokens' should 
            #     # be same as the number of 'key_tokens' in hidden_states. However, hidden_states (image latent) will be 
            #     # down/upsampling during Unet process.
            #     # Thus, we adaptively adjust the 'key_tokens' of attention_mask according to: key_tokens = h*w
            #     height, width = int(np.sqrt(target_length)), int(np.sqrt(target_length))
            #     batch = attention_mask.shape[0]
            #     key_tokens = attention_mask.shape[-1]
            #     h, w = int(np.sqrt(key_tokens)), int(np.sqrt(key_tokens))
            #     attention_mask = attention_mask.reshape(batch, 1, h, w)
            #     attention_mask = F.interpolate(attention_mask, size=(height, width)).view(batch, 1, -1)
            #     attention_mask = (1. * (attention_mask>=0.5)).to(intype)

            attention_mask = attention_mask.unsqueeze(1)
            # attention_mask = attention_mask.repeat_interleave(batch_size*head_size, dim=1)

            return attention_mask

        if self.cfg.reference_attn:
            attn_modules = [module for module in torch_dfs(self.unet) if isinstance(module, BasicTransformerBlock)]
            attn_modules = sorted(attn_modules, key=lambda x: -x.norm1.normalized_shape[0])

            for i, module in enumerate(attn_modules):
                module._original_inner_forward = module.forward
                module.forward = hacked_basic_transformer_inner_forward.__get__(module, BasicTransformerBlock)
                module.bank = []
                module.bankmask = []
                module.attn_weight = float(i) / float(len(attn_modules))

            attn_modules = [module for module in torch_dfs(self.unet) if isinstance(module, Attention)]
            for i, module in enumerate(attn_modules):
                module._original_get_attention_scores = module.get_attention_scores
                module._original_prepare_attention_mask = module.prepare_attention_mask
                module.get_attention_scores = hacked_get_attention_scores_inner_attention.__get__(module, Attention)
                module.prepare_attention_mask = hacked_prepare_attention_mask_inner_attention.__get__(module, Attention)

        # freeze the network
        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)

        # self.scheduler = DDIMScheduler.from_pretrained(
        #     self.cfg.pretrained_model_name_or_path,
        #     subfolder="scheduler",
        #     torch_dtype=self.weights_dtype,
        # )
        self.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * self.cfg.min_step_percent)
        self.max_step = int(self.num_train_timesteps * self.cfg.max_step_percent)
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)
        self.scheduler.alpha_t = self.scheduler.alpha_t.to(self.device)
        self.scheduler.sigma_t = self.scheduler.sigma_t.to(self.device)
        self.scheduler.lambda_t = self.scheduler.lambda_t.to(self.device)

        self.aug = T.Compose([
            T.Resize((224, 224)),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        self.text_latents = {}

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )

        self.grad_clip_val: Optional[float] = None

        threestudio.info(f"Loaded Stable Diffusion!")

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)
    
    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
        attention_mask=None,
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return self.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            return_dict=False,
            attention_mask=attention_mask,
        )[0].to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 512 512"]
    ) -> Float[Tensor, "B 4 64 64"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self,
        latents: Float[Tensor, "B 4 H W"],
        latent_height: int = 64,
        latent_width: int = 64,
    ) -> Float[Tensor, "B 3 512 512"]:
        input_dtype = latents.dtype
        latents = F.interpolate(
            latents, (latent_height, latent_width), mode="bilinear", align_corners=False
        )
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    def img_clip_loss(self, clip_model, rgb1, rgb2):
        image_z_1 = clip_model.encode_image(self.aug(rgb1))
        image_z_2 = clip_model.encode_image(self.aug(rgb2))
        image_z_1 = image_z_1 / image_z_1.norm(dim=-1, keepdim=True) # normalize features
        image_z_2 = image_z_2 / image_z_2.norm(dim=-1, keepdim=True) # normalize features

        loss = - (image_z_1 * image_z_2).sum(-1).mean()
        return loss
    
    def img_text_clip_loss(self, clip_model, rgb, prompt, view_dir=None):
        image_z_1 = clip_model.encode_image(self.aug(rgb))
        image_z_1 = image_z_1 / image_z_1.norm(dim=-1, keepdim=True) # normalize features

        if view_dir in self.text_latents:
            text_z = self.text_latents[view_dir]
        else:
            text = clip.tokenize(prompt).to(self.device)
            text_z = clip_model.encode_text(text)
            text_z = text_z / text_z.norm(dim=-1, keepdim=True)
            self.text_latents[view_dir] = text_z.detach()
        loss = - (image_z_1 * text_z).sum(-1).mean()
        return loss
    
    def multiple_step_denoise(
        self, 
        rgb: Float[Tensor, "B H W C"],
        ref_image: Float[Tensor, "B H W C"],
        t_max_rate,  #[0,1]
        num_steps_total,
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        rendered_region='full',
        mask_image_ref=None,  # B1HW, B=0 indicate background, B!=0 indicate foreground
        mask_image_ini=None,
        **kwargs,
        ):
        global MODE
        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        ref_image = ref_image.permute(0, 3, 1, 2) # BHWC --> BCHW, range[0, 1]

        # rgb_BCHW_512 = F.interpolate(
        #     rgb_BCHW, (512, 512), mode="bilinear", align_corners=False
        # )
        # encode image into latents with vae
        latents = self.encode_images(rgb_BCHW)

        # Prepare reference latent variables
        ref_image_latents = self.encode_images(ref_image)

        # attention mask
        mask_image_ref, mask_image_ini = self.preprocess_attention_mask(mask_image_ref, mask_image_ini, latents.dtype)

        text_embeddings = prompt_utils.get_local_text_embeddings(
            rendered_region, elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
        )

        self.scheduler.set_timesteps(num_steps_total, device=self.device)
        timesteps = self.scheduler.timesteps
        num_inference_steps = int(t_max_rate*num_steps_total)
        timesteps = timesteps[-num_inference_steps:]

        noise = randn_tensor(latents.shape, device=self.device, dtype=ref_image_latents.dtype)
        latents = self.scheduler.add_noise(latents, noise, timesteps[0])

        for t in timesteps:
            # reference
            noise = randn_tensor(ref_image_latents.shape, device=self.device, dtype=ref_image_latents.dtype)
            ref_xt = self.scheduler.add_noise(ref_image_latents, noise, t)
            ref_xt = torch.cat([ref_xt] * 2)
            ref_xt = self.scheduler.scale_model_input(ref_xt, t)
            MODE = "write"
            self.unet(
                    ref_xt.to(self.weights_dtype),
                    t.to(self.weights_dtype),
                    encoder_hidden_states=text_embeddings.to(self.weights_dtype),
                    return_dict=False,
                    attention_mask=mask_image_ref,
                )
            
            # pred noise          
            latent_model_input = torch.cat([latents] * 2, dim=0)
            MODE = "read"
            noise_pred = self.forward_unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings,
                    attention_mask=mask_image_ini,
                )  # (4B, 3, 64, 64)
            
            # perform guidance
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_text + self.cfg.guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            
        image = self.decode_latents(latents)
        # image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        # image = (image * 0.5 + 0.5).clamp(0, 1)
        return image

    def compute_grad_sds(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        ref_image_latents: Float[Tensor, "B 4 64 64"],
        t: Int[Tensor, "B"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        rendered_region='full',
        mask_image_ref=None,
        mask_image_ini=None,
    ):
        global MODE, uc_mask
        batch_size = latents.shape[0]

        if prompt_utils.use_perp_neg:
            (text_embeddings, neg_guidance_weights,) = prompt_utils.get_local_text_embeddings_perp_neg(rendered_region, elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting)
            uc_mask = (
                torch.Tensor([0] + [1] + [0]*2).to(self.device)
                .bool()
            )
            with torch.no_grad():
                # add noise
                noise = randn_tensor(ref_image_latents.shape, device=self.device, dtype=ref_image_latents.dtype)
                latents_noisy = self.scheduler.add_noise(latents, noise, t)
                ref_xt = self.scheduler.add_noise(ref_image_latents, noise, t)

                # reference
                ref_xt = torch.cat([ref_xt] * 4)
                ref_xt = self.scheduler.scale_model_input(ref_xt, t)
                MODE = "write"
                self.unet(
                    ref_xt.to(self.weights_dtype),
                    torch.cat([t] * 4).to(self.weights_dtype),
                    encoder_hidden_states=text_embeddings.to(self.weights_dtype),
                    return_dict=False,
                    attention_mask=mask_image_ref,
                )

                # pred noise      
                latent_model_input = torch.cat([latents_noisy] * 4, dim=0)
                MODE = "read"
                noise_pred = self.forward_unet(
                    latent_model_input,
                    torch.cat([t] * 4),
                    encoder_hidden_states=text_embeddings,
                    attention_mask=mask_image_ini,
                )  # (4B, 3, 64, 64)
            
            noise_pred_text = noise_pred[:batch_size]
            noise_pred_uncond = noise_pred[batch_size : batch_size * 2]
            noise_pred_neg = noise_pred[batch_size * 2 :]
            e_pos = noise_pred_text - noise_pred_uncond
            accum_grad = 0
            n_negative_prompts = neg_guidance_weights.shape[-1]
            for i in range(n_negative_prompts):
                e_i_neg = noise_pred_neg[i::n_negative_prompts] - noise_pred_uncond
                accum_grad += neg_guidance_weights[:, i].view(
                    -1, 1, 1, 1
                ) * perpendicular_component(e_i_neg, e_pos)

            noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (e_pos + accum_grad)
        else:
            neg_guidance_weights = None
            text_embeddings = prompt_utils.get_local_text_embeddings(
                rendered_region, elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )
            # predict the noise residual with unet, NO grad!
            with torch.no_grad():
                # add noise
                noise = randn_tensor(ref_image_latents.shape, device=self.device, dtype=ref_image_latents.dtype)
                latents_noisy = self.scheduler.add_noise(latents, noise, t)
                ref_xt = self.scheduler.add_noise(ref_image_latents, noise, t)
                
                # reference
                ref_xt = torch.cat([ref_xt] * 2)
                ref_xt = self.scheduler.scale_model_input(ref_xt, t)
                MODE = "write"
                self.unet(
                        ref_xt.to(self.weights_dtype),
                        torch.cat([t] * 2).to(self.weights_dtype),
                        encoder_hidden_states=text_embeddings.to(self.weights_dtype),
                        return_dict=False,
                        attention_mask=mask_image_ref, #expects mask of shape: [batch, key_tokens]
                    )
                
                # pred noise          
                latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
                MODE = "read"
                noise_pred = self.forward_unet(
                        latent_model_input,
                        torch.cat([t] * 2),
                        encoder_hidden_states=text_embeddings,
                        attention_mask=mask_image_ini,
                    )  # (4B, 3, 64, 64)

            # perform guidance
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_text + self.cfg.guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        # latents = self.scheduler.step(noise_pred, t, latents_noisy, return_dict=False)[0]

        if self.cfg.weighting_strategy == "sds":
            # w(t), sigma_t^2
            w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        elif self.cfg.weighting_strategy == "uniform":
            w = 1
        elif self.cfg.weighting_strategy == "fantasia3d":
            w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1)
        else:
            raise ValueError(
                f"Unknown weighting strategy: {self.cfg.weighting_strategy}"
            )

        grad = w * (noise_pred - noise)

        guidance_eval_utils = {
            "use_perp_neg": prompt_utils.use_perp_neg,
            "text_embeddings": text_embeddings,
            "t_orig": t,
            "latents_noisy": latents_noisy,
            "noise_pred": noise_pred,
        }

        return grad, guidance_eval_utils
    
    def compute_grad_sds_clip(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        ref_image_latents: Float[Tensor, "B 4 64 64"],
        t: Int[Tensor, "B"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        ref_rgb=None, 
        ref_text=None, 
        clip_model=None,
        rendered_region='full',
        with_clip_text_loss=True,
        mask_image_ref=None,
        mask_image_ini=None,
    ):
        out_grad = True
        global MODE
        neg_guidance_weights = None
        text_embeddings = prompt_utils.get_local_text_embeddings(
            rendered_region, elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
        )
        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = randn_tensor(ref_image_latents.shape, device=self.device, dtype=ref_image_latents.dtype)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            ref_xt = self.scheduler.add_noise(ref_image_latents, noise, t)
            
            # reference
            ref_xt = torch.cat([ref_xt] * 2)
            ref_xt = self.scheduler.scale_model_input(ref_xt, t)
            MODE = "write"
            self.unet(
                    ref_xt.to(self.weights_dtype),
                    torch.cat([t] * 2).to(self.weights_dtype),
                    encoder_hidden_states=text_embeddings.to(self.weights_dtype),
                    return_dict=False,
                    attention_mask=mask_image_ref,
                )
            
            # pred noise          
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            MODE = "read"
            noise_pred = self.forward_unet(
                    latent_model_input,
                    torch.cat([t] * 2),
                    encoder_hidden_states=text_embeddings,
                    attention_mask=mask_image_ini,
                )  # (4B, 3, 64, 64)

        # perform guidance (high scale from paper!)
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_text + self.cfg.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )
        
        # use sds loss OR clip loss
        if abs(azimuth[0])<torch.tensor(30.) and (t / self.num_train_timesteps) <= 0.4:
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            self.scheduler.set_timesteps(self.num_train_timesteps, device=self.device)
            de_latents = self.scheduler.step(noise_pred, t, latents_noisy)['prev_sample']
            imgs = self.decode_latents(de_latents)
            grad = 10.*self.img_clip_loss(clip_model, imgs, ref_rgb) 
            if with_clip_text_loss:
                grad += 10.*self.img_text_clip_loss(clip_model, imgs, ref_text[0], ref_text[1])
            out_grad = False
        else:
            if self.cfg.weighting_strategy == "sds":
                # w(t), sigma_t^2
                w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
            elif self.cfg.weighting_strategy == "uniform":
                w = 1
            elif self.cfg.weighting_strategy == "fantasia3d":
                w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1)
            else:
                raise ValueError(
                    f"Unknown weighting strategy: {self.cfg.weighting_strategy}"
                )

            grad = w * (noise_pred - noise)

        guidance_eval_utils = {
            "use_perp_neg": prompt_utils.use_perp_neg,
            "neg_guidance_weights": neg_guidance_weights,
            "text_embeddings": text_embeddings,
            "t_orig": t,
            "latents_noisy": latents_noisy,
            "noise_pred": noise_pred,
            "out_grad": out_grad,
        }

        return grad, guidance_eval_utils

    def preprocess_attention_mask(self, mask_image_ref, mask_image_ini, intype):
        # attention mask
        if self.cfg.attention_strategy == 3:
            mask_image_ref, mask_image_ini = None, None
        else:
            if mask_image_ref is not None and mask_image_ini is not None:  # converting to expected mask shape: [batch, key_tokens]
                b_ref, b_ini = mask_image_ref.shape[0], mask_image_ini.shape[0]
                assert b_ref == b_ini, 'the batch size of mask_image_ref and mask_image_ini should be same.'
                mask_image_ref = F.interpolate(mask_image_ref, size=(64, 64))
                mask_image_ref = 1. * (mask_image_ref>=0.5)
                # mask_image_ref[:1] = (1. - 1. * (mask_image_ref[1:].sum(dim=0, keepdim=True)>=0.5)).to(intype)
                mask_image_ref = mask_image_ref.view(b_ref, -1).to(intype)

                mask_image_ini = F.interpolate(mask_image_ini, size=(64, 64))
                mask_image_ini = 1. * (mask_image_ini>=0.5)
                # mask_image_ini[:1] = (1. - 1. * (mask_image_ini[1:].sum(dim=0, keepdim=True)>=0.5)).to(intype)
                mask_image_ini = mask_image_ini.view(b_ini, -1).to(intype)
            else:
                mask_image_ref, mask_image_ini = None, None
        return mask_image_ref, mask_image_ini
    
    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        ref_image: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        guidance_eval=False,
        ref_rgb=None, 
        ref_text=None, 
        clip_model=None,
        with_clip_loss=False,
        with_clip_text_loss=True,
        rendered_region='full',
        mask_image_ref=None,  # B1HW, B=0 indicate background, B!=0 indicate foreground
        mask_image_ini=None,
        **kwargs,
    ):
        batch_size = rgb.shape[0]
        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        ref_image = ref_image.permute(0, 3, 1, 2) # BHWC --> BCHW, range[0, 1]

        # rgb_BCHW_512 = F.interpolate(
        #     rgb_BCHW, (512, 512), mode="bilinear", align_corners=False
        # )
        # encode image into latents with vae
        latents = self.encode_images(rgb_BCHW)
        
        # Prepare reference latent variables
        ref_image_latents = self.encode_images(ref_image)

        # attention mask
        mask_image_ref, mask_image_ini = self.preprocess_attention_mask(mask_image_ref, mask_image_ini, latents.dtype)

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.min_step,
            self.max_step + 1,
            [batch_size],
            dtype=torch.long,
            device=self.device,
        )

        if with_clip_loss:
            grad, guidance_eval_utils = self.compute_grad_sds_clip(
                latents, ref_image_latents, t, prompt_utils, elevation, azimuth, camera_distances, ref_rgb=ref_rgb, ref_text=ref_text, 
                clip_model=clip_model, rendered_region=rendered_region, with_clip_text_loss=with_clip_text_loss,
                mask_image_ref=mask_image_ref, mask_image_ini=mask_image_ini,
            )
        else:
            grad, guidance_eval_utils = self.compute_grad_sds(
                latents, ref_image_latents, t, prompt_utils, elevation, azimuth, camera_distances, 
                rendered_region=rendered_region, mask_image_ref=mask_image_ref, mask_image_ini=mask_image_ini,
            )
        grad = torch.nan_to_num(grad)

        if "out_grad" in guidance_eval_utils and not guidance_eval_utils["out_grad"]:
            loss_clip = grad
            loss_sds = 0.
        else:
            # clip grad for stable training?
            if self.grad_clip_val is not None:
                grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

            # loss = SpecifyGradient.apply(latents, grad)
            # SpecifyGradient is not straghtforward, use a reparameterization trick instead
            target = (latents - grad).detach()
            # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
            loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size
            loss_clip = 0.

        guidance_out = {
            "loss_ref_sds": loss_sds,
            "loss_clip": loss_clip,
            "grad_norm": grad.norm(),
        }

        if guidance_eval:
            guidance_eval_out = self.guidance_eval(**guidance_eval_utils)
            texts = []
            for n, e, a, c in zip(
                guidance_eval_out["noise_levels"], elevation, azimuth, camera_distances
            ):
                texts.append(
                    f"n{n:.02f}\ne{e.item():.01f}\na{a.item():.01f}\nc{c.item():.02f}"
                )
            guidance_eval_out.update({"texts": texts})
            guidance_out.update({"eval": guidance_eval_out})

        return guidance_out

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def get_noise_pred(
        self,
        latents_noisy,
        t,
        text_embeddings,
        use_perp_neg=False,
        neg_guidance_weights=None,
    ):
        batch_size = latents_noisy.shape[0]

        if use_perp_neg:
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 4, dim=0)
            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t.reshape(1)] * 4).to(self.device),
                encoder_hidden_states=text_embeddings,
            )  # (4B, 3, 64, 64)

            noise_pred_text = noise_pred[:batch_size]
            noise_pred_uncond = noise_pred[batch_size : batch_size * 2]
            noise_pred_neg = noise_pred[batch_size * 2 :]

            e_pos = noise_pred_text - noise_pred_uncond
            accum_grad = 0
            n_negative_prompts = neg_guidance_weights.shape[-1]
            for i in range(n_negative_prompts):
                e_i_neg = noise_pred_neg[i::n_negative_prompts] - noise_pred_uncond
                accum_grad += neg_guidance_weights[:, i].view(
                    -1, 1, 1, 1
                ) * perpendicular_component(e_i_neg, e_pos)

            noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
                e_pos + accum_grad
            )
        else:
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t.reshape(1)] * 2).to(self.device),
                encoder_hidden_states=text_embeddings,
            )
            # perform guidance (high scale from paper!)
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_text + self.cfg.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        return noise_pred

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def guidance_eval(
        self,
        t_orig,
        text_embeddings,
        latents_noisy,
        noise_pred,
        use_perp_neg=False,
        neg_guidance_weights=None,
    ):
        # use only 50 timesteps, and find nearest of those to t
        self.scheduler.set_timesteps(50)
        self.scheduler.timesteps_gpu = self.scheduler.timesteps.to(self.device)
        bs = latents_noisy.shape[0]  # batch size
        large_enough_idxs = self.scheduler.timesteps_gpu.expand(
            [bs, -1]
        ) > t_orig.unsqueeze(
            -1
        )  # sized [bs,50] > [bs,1]
        idxs = torch.min(large_enough_idxs, dim=1)[1]
        t = self.scheduler.timesteps_gpu[idxs]

        fracs = list((t / self.scheduler.config.num_train_timesteps).cpu().numpy())
        imgs_noisy = self.decode_latents(latents_noisy).permute(0, 2, 3, 1)

        # get prev latent
        latents_1step = []
        pred_1orig = []
        for b in range(len(t)):
            step_output = self.scheduler.step(
                noise_pred[b : b + 1], t[b], latents_noisy[b : b + 1], eta=1
            )
            latents_1step.append(step_output["prev_sample"])
            pred_1orig.append(step_output["pred_original_sample"])
        latents_1step = torch.cat(latents_1step)
        pred_1orig = torch.cat(pred_1orig)
        imgs_1step = self.decode_latents(latents_1step).permute(0, 2, 3, 1)
        imgs_1orig = self.decode_latents(pred_1orig).permute(0, 2, 3, 1)

        latents_final = []
        for b, i in enumerate(idxs):
            latents = latents_1step[b : b + 1]
            text_emb = (
                text_embeddings[
                    [b, b + len(idxs), b + 2 * len(idxs), b + 3 * len(idxs)], ...
                ]
                if use_perp_neg
                else text_embeddings[[b, b + len(idxs)], ...]
            )
            neg_guid = neg_guidance_weights[b : b + 1] if use_perp_neg else None
            for t in tqdm(self.scheduler.timesteps[i + 1 :], leave=False):
                # pred noise
                noise_pred = self.get_noise_pred(
                    latents, t, text_emb, use_perp_neg, neg_guid
                )
                # get prev latent
                latents = self.scheduler.step(noise_pred, t, latents, eta=1)[
                    "prev_sample"
                ]
            latents_final.append(latents)

        latents_final = torch.cat(latents_final)
        imgs_final = self.decode_latents(latents_final).permute(0, 2, 3, 1)

        return {
            "noise_levels": fracs,
            "imgs_noisy": imgs_noisy,
            "imgs_1step": imgs_1step,
            "imgs_1orig": imgs_1orig,
            "imgs_final": imgs_final,
        }

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        # t annealing from ProlificDreamer
        if (
            self.cfg.anneal_start_step is not None
            and global_step > self.cfg.anneal_start_step
        ):
            self.max_step = int(
                self.num_train_timesteps * self.cfg.max_step_percent_annealed
            )
