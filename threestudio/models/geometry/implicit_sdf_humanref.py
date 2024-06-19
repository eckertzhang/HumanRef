import os, random
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.geometry.base import BaseImplicitGeometry, contract_to_unisphere
from threestudio.models.mesh import Mesh
from threestudio.models.networks import get_encoding, get_mlp
from threestudio.utils.misc import broadcast, get_rank
from threestudio.utils.typing import *


@threestudio.register("implicit-sdf-humanref")
class ImplicitSDF(BaseImplicitGeometry):
    @dataclass
    class Config(BaseImplicitGeometry.Config):
        n_input_dims: int = 3
        n_feature_dims: int = 3  # for outputing color
        n_common_dims: int = 256
        predict_offset: bool = False
        pos_encoding_config: dict = field(
            default_factory=lambda: {
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 1.447269237440378,
            }
        )
        mlp_network_config: dict = field(
            default_factory=lambda: {
                "otype": "VanillaMLP",
                "activation": "ReLU",
                "output_activation": "none",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            }
        )
        normal_type: Optional[
            str
        ] = "finite_difference"  # in ['pred', 'finite_difference', 'finite_difference_laplacian']
        finite_difference_normal_eps: Union[
            float, str
        ] = 0.01  # in [float, "progressive"]
        shape_init: Optional[str] = None
        shape_init_params: Optional[Any] = None
        shape_init_mesh_up: str = "+z"
        shape_init_mesh_front: str = "+x"
        force_shape_init: bool = False
        sdf_bias: Union[float, str] = 0.0
        sdf_bias_params: Optional[Any] = None
        fov: Optional[float] = 60
        geo_prior_type: Optional[str] = None
        image_path: Optional[str] = None

        # set for different config
        out_mlp_network_config: Optional[dict] = None
        rgb_pos_encoding_config: Optional[dict] = None
        sdf_select_level: int = 8
        run_individual: bool = False
        run_sharing: bool = False

        # geometric loss
        out_eikonal_loss: bool = False
        out_sdf_loss: bool = False
        out_sdf_smooth_loss: bool = False
        use_orthograph: bool = False

        # no need to removal outlier for SDF
        isosurface_remove_outliers: bool = False

    cfg: Config

    def configure(self) -> None:
        super().configure()
        self.encoding = get_encoding(
            self.cfg.n_input_dims, self.cfg.pos_encoding_config
        )
        self.run_common_frame = False
        self.run_hierhashgrid = False
        self.run_individual = self.cfg.run_individual
        self.run_sharing = self.cfg.run_sharing
        self.encoding_rgb = None

        if self.cfg.pos_encoding_config.otype == 'HashGrid' or self.cfg.pos_encoding_config.otype == 'ProgressiveBandHashGrid':
            if self.run_sharing:
                self.sdf_network = get_mlp(
                    self.encoding.n_output_dims, 1+self.cfg.n_feature_dims, self.cfg.mlp_network_config
                )
            else:
                self.sdf_network = get_mlp(
                    self.encoding.n_output_dims, 1, self.cfg.mlp_network_config
                )
                if self.cfg.n_feature_dims > 0:  # to output appearance
                    if self.run_individual:
                        self.encoding_rgb = get_encoding(
                            self.cfg.n_input_dims, self.cfg.rgb_pos_encoding_config
                        )
                        self.feature_network = get_mlp(
                            self.encoding_rgb.n_output_dims,
                            self.cfg.n_feature_dims,
                            self.cfg.mlp_network_config,
                        )
                    else:
                        self.feature_network = get_mlp(
                            self.encoding.n_output_dims,
                            self.cfg.n_feature_dims,
                            self.cfg.mlp_network_config,
                        )
                if self.cfg.normal_type == "pred":
                    self.normal_network = get_mlp(
                        self.encoding.n_output_dims, 3, self.cfg.mlp_network_config
                    )
                if self.cfg.isosurface_deformable_grid:
                    assert (
                        self.cfg.isosurface_method == "mt"
                    ), "isosurface_deformable_grid only works with mt"
                    self.deformation_network = get_mlp(
                        self.encoding.n_output_dims, 3, self.cfg.mlp_network_config
                    )
        elif self.cfg.pos_encoding_config.otype == 'HierHashGrid':
            self.run_hierhashgrid = True
            self.in_dim_sdf = self.cfg.pos_encoding_config.n_features_per_level * self.cfg.sdf_select_level
            self.sdf_network = get_mlp(
                self.in_dim_sdf, 1, self.cfg.mlp_network_config
            )
            if self.cfg.n_feature_dims > 0:  # to output appearance
                self.feature_network = get_mlp(
                    self.encoding.n_output_dims,
                    self.cfg.n_feature_dims,
                    self.cfg.mlp_network_config,
                )
        else:
            self.run_common_frame = True
            self.common_net = get_mlp(
                self.encoding.n_output_dims, self.cfg.n_common_dims, self.cfg.mlp_network_config
            )
            self.common_act = nn.ReLU(inplace=True)
            self.sdf_network = get_mlp(self.cfg.n_common_dims, 1, self.cfg.out_mlp_network_config)
            if self.cfg.n_feature_dims > 0:  # to output appearance
                self.feature_network = get_mlp(
                    self.cfg.n_common_dims,
                    self.cfg.n_feature_dims,
                    self.cfg.out_mlp_network_config,
                )
            if self.cfg.normal_type == "pred":
                self.normal_network = get_mlp(
                    self.cfg.n_common_dims, 3, self.cfg.out_mlp_network_config
                )
            if self.cfg.isosurface_deformable_grid:
                assert (
                    self.cfg.isosurface_method == "mt"
                ), "isosurface_deformable_grid only works with mt"
                self.deformation_network = get_mlp(
                    self.cfg.n_common_dims, 3, self.cfg.out_mlp_network_config
                )

        
        if self.cfg.geo_prior_type == 'econ_smpl' or self.cfg.geo_prior_type == 'econ' or self.cfg.geo_prior_type == 'cape_smpl' or self.cfg.geo_prior_type == 'thuman2_smpl':
            self.cfg.shape_init_mesh_up = '+y'
            self.cfg.shape_init_mesh_front = '+z'
        elif self.cfg.geo_prior_type == 'hybrik':
            self.cfg.shape_init_mesh_up = '+y'
            self.cfg.shape_init_mesh_front = '-z'
        self.load_ref_shape()

        self.finite_difference_normal_eps: Optional[float] = self.cfg.finite_difference_normal_eps

    def load_ref_shape(self):
        img_name = os.path.basename(self.cfg.image_path).split('.')[0].replace('_0_rgba', '')
        if self.cfg.geo_prior_type == 'econ_smpl':
            mesh_path = os.path.join(os.getenv('ECON_PATH'), img_name, f'econ/obj/{img_name}_smpl_00.obj')
        elif self.cfg.geo_prior_type == 'cape_smpl':
            mesh_path = os.path.join(os.getenv('Human3D_PATH'), 'cape_test_data/results_econ', img_name, f'econ/obj/{img_name}_smpl_00_prior.obj')
        elif self.cfg.geo_prior_type == 'thuman2_smpl':
            mesh_path = os.path.join(os.getenv('Human3D_PATH'),'thuman2/results_econ', img_name, f'econ/obj/{img_name}_smpl_00_prior.obj')
        else:
            mesh_path = self.cfg.shape_init[5:]
        if not os.path.exists(mesh_path):
            raise ValueError(f"Mesh file {mesh_path} does not exist.")
        import trimesh
        mesh = trimesh.load(mesh_path)
        # align to up-z and front-x
        dirs = ["+x", "+y", "+z", "-x", "-y", "-z"]
        dir2vec = {
            "+x": np.array([1, 0, 0]),
            "+y": np.array([0, 1, 0]),
            "+z": np.array([0, 0, 1]),
            "-x": np.array([-1, 0, 0]),
            "-y": np.array([0, -1, 0]),
            "-z": np.array([0, 0, -1]),
        }
        if (
            self.cfg.shape_init_mesh_up not in dirs
            or self.cfg.shape_init_mesh_front not in dirs
        ):
            raise ValueError(f"shape_init_mesh_up and shape_init_mesh_front must be one of {dirs}.")
        if self.cfg.shape_init_mesh_up[1] == self.cfg.shape_init_mesh_front[1]:
            raise ValueError("shape_init_mesh_up and shape_init_mesh_front must be orthogonal.")
        z_, x_ = (
            dir2vec[self.cfg.shape_init_mesh_up],
            dir2vec[self.cfg.shape_init_mesh_front],
        )
        y_ = np.cross(z_, x_)
        std2mesh = np.stack([x_, y_, z_], axis=0).T
        mesh2std = np.linalg.inv(std2mesh)

        # scaling
        if self.cfg.geo_prior_type == 'econ_smpl' or self.cfg.geo_prior_type == 'cape_smpl' or self.cfg.geo_prior_type == 'thuman2_smpl':
            if self.cfg.use_orthograph:
                scale = 1.
            else:
                fov = self.cfg.fov
                scale = 1. / np.tan(np.deg2rad(fov/2.))
            mesh.vertices = mesh.vertices / scale * self.cfg.shape_init_params
        mesh.vertices = np.dot(mesh2std, mesh.vertices.T).T

        from pysdf import SDF

        sdf = SDF(mesh.vertices, mesh.faces)

        def func(points_rand: Float[Tensor, "N 3"]) -> Float[Tensor, "N 1"]:
            # add a negative signed here
            # as in pysdf the inside of the shape has positive signed distance
            return torch.from_numpy(-sdf(points_rand.cpu().numpy())).to(
                points_rand
            )[..., None]

        self.get_gt_sdf = func

    def initialize_shape(self, run_init=True) -> None:
        if self.cfg.shape_init is None and not self.cfg.force_shape_init:
            return

        # do not initialize shape if weights are provided
        if self.cfg.weights is not None and not self.cfg.force_shape_init:
            return

        if self.cfg.sdf_bias != 0.0:
            threestudio.warn(
                "shape_init and sdf_bias are both specified, which may lead to unexpected results."
            )

        self.get_gt_sdf: Callable[[Float[Tensor, "N 3"]], Float[Tensor, "N 1"]]
        assert isinstance(self.cfg.shape_init, str)
        if self.cfg.shape_init == "ellipsoid":
            assert (
                isinstance(self.cfg.shape_init_params, Sized)
                and len(self.cfg.shape_init_params) == 3
            )
            size = torch.as_tensor(self.cfg.shape_init_params).to(self.device)

            def func(points_rand: Float[Tensor, "N 3"]) -> Float[Tensor, "N 1"]:
                return ((points_rand / size) ** 2).sum(
                    dim=-1, keepdim=True
                ).sqrt() - 1.0  # pseudo signed distance of an ellipsoid

            self.get_gt_sdf = func
        elif self.cfg.shape_init == "sphere":
            assert isinstance(self.cfg.shape_init_params, float)
            radius = self.cfg.shape_init_params

            def func(points_rand: Float[Tensor, "N 3"]) -> Float[Tensor, "N 1"]:
                return (points_rand**2).sum(dim=-1, keepdim=True).sqrt() - radius

            self.get_gt_sdf = func
        elif self.cfg.shape_init.startswith("mesh"):
            assert isinstance(self.cfg.shape_init_params, float)
            assert isinstance(self.cfg.geo_prior_type, str)

            self.load_ref_shape()

        else:
            raise ValueError(
                f"Unknown shape initialization type: {self.cfg.shape_init}"
            )

        # Initialize SDF to a given shape when no weights are provided or force_shape_init is True
        if run_init and self.training:
            optim = torch.optim.Adam(self.parameters(), lr=1e-3)
            from tqdm import tqdm

            for ii in tqdm(
                range(1000),
                desc=f"Initializing SDF to a(n) {self.cfg.shape_init}:",
                disable=get_rank() != 0,
            ):
                points_rand = (
                    torch.rand((20000, 3), dtype=torch.float32, device=self.device) * 2.0 - 1.0
                )
                
                sdf_pred = self.forward_sdf(points_rand, debug_using_gt=False)
                sdf_gt = self.get_gt_sdf(points_rand)

                if self.cfg.predict_offset:
                    loss = F.mse_loss(sdf_pred, sdf_gt)
                else:
                    loss = F.l1_loss(sdf_pred, sdf_gt)
                optim.zero_grad()
                loss.backward()
                optim.step()

            print('initialization loss:', loss.item())
            # explicit broadcast to ensure param consistency across ranks
            for param in self.parameters():
                broadcast(param, src=0)

    def get_shifted_sdf(
        self, points: Float[Tensor, "*N Di"], sdf: Float[Tensor, "*N 1"]
    ) -> Float[Tensor, "*N 1"]:
        sdf_bias: Union[float, Float[Tensor, "*N 1"]]
        if self.cfg.sdf_bias == "ellipsoid":
            assert (
                isinstance(self.cfg.sdf_bias_params, Sized)
                and len(self.cfg.sdf_bias_params) == 3
            )
            size = torch.as_tensor(self.cfg.sdf_bias_params).to(points)
            sdf_bias = ((points / size) ** 2).sum(
                dim=-1, keepdim=True
            ).sqrt() - 1.0  # pseudo signed distance of an ellipsoid
        elif self.cfg.sdf_bias == "sphere":
            assert isinstance(self.cfg.sdf_bias_params, float)
            radius = self.cfg.sdf_bias_params
            sdf_bias = (points**2).sum(dim=-1, keepdim=True).sqrt() - radius
        elif isinstance(self.cfg.sdf_bias, float):
            sdf_bias = self.cfg.sdf_bias
        else:
            raise ValueError(f"Unknown sdf bias {self.cfg.sdf_bias}")
        return sdf + sdf_bias

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.forward_sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)
    
    def clone_hash_grid(self):
        import copy
        self.encoding_rgb = copy.deepcopy(self.encoding)

    def forward(
        self, points: Float[Tensor, "*N Di"], output_normal: bool = False
    ) -> Dict[str, Float[Tensor, "..."]]:
        
        grad_enabled = torch.is_grad_enabled()
        if output_normal and self.cfg.normal_type == "analytic":
            torch.set_grad_enabled(True)
            points.requires_grad_(True)

        points_unscaled = points  # points in the original scale
        points = contract_to_unisphere(
            points, self.bbox, self.unbounded
        )  # points normalized to (0, 1)

        enc = self.encoding(points.view(-1, self.cfg.n_input_dims))
        if self.run_common_frame:
            enc = self.common_net(enc)
        if self.run_hierhashgrid:
            sdf = self.sdf_network(enc[:, :self.in_dim_sdf]).view(*points.shape[:-1], 1)
        else:
            if self.run_sharing:
                sdf_fea = self.sdf_network(enc).view(*points.shape[:-1], 4)
                sdf = sdf_fea[..., :1]
            else:
                sdf = self.sdf_network(enc).view(*points.shape[:-1], 1)
        sdf = self.get_shifted_sdf(points_unscaled, sdf)

        if self.cfg.predict_offset:
            with torch.no_grad():
                sdf_base = self.get_gt_sdf(points_unscaled)
            sdf = sdf_base + sdf

        output = {"sdf": sdf}

        # Eikonal loss
        if self.training and self.cfg.out_eikonal_loss:
            gradients = self.gradient(points_unscaled).squeeze()
            gradient_error = (torch.linalg.norm(gradients.view(*points.shape[:-1], 3), ord=2, dim=-1) - 1.0) ** 2
            pts_norm = torch.linalg.norm(points_unscaled, ord=2, dim=-1, keepdim=True).view(*points.shape[:-1])
            relax_inside_sphere = (pts_norm < 1.2).float().detach()
            gradient_error = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)

            output.update({"sdf_eikonal_loss": gradient_error})

        plot_coutour = False
        if self.training and plot_coutour:
            size = 256
            i, j = torch.meshgrid(
                torch.arange(size, dtype=torch.float32) + 0.5,
                torch.arange(size, dtype=torch.float32) + 0.5,
                indexing="xy",
            )
            points_surface = torch.stack(
                [torch.zeros_like(i), i/size * 2.-1, j/size * 2.-1], -1
            ).reshape(-1, 3)
            with torch.no_grad():
                sdf_surf = self.forward_sdf(points_surface.to(self.device)).reshape(size, size)
            plt_coutour = {"i": i, "j": j, "sdf_surfx": sdf_surf}
            points_surface = torch.stack(
                [i/size * 2.-1, torch.zeros_like(i), j/size * 2.-1], -1
            ).reshape(-1, 3)
            with torch.no_grad():
                sdf_surf = self.forward_sdf(points_surface.to(self.device)).reshape(size, size)
            plt_coutour.update({"sdf_surfy": sdf_surf})
            points_surface = torch.stack(
                [i/size * 2.-1, j/size * 2.-1, torch.zeros_like(i)], -1
            ).reshape(-1, 3)
            with torch.no_grad():
                sdf_surf = self.forward_sdf(points_surface.to(self.device)).reshape(size, size)
            plt_coutour.update({"sdf_surfz": sdf_surf})
            output.update({"plt_coutour": plt_coutour})

        # SDF Loss
        if self.training and self.cfg.out_sdf_loss:
            random_sdf = False
            if self.cfg.predict_offset:
                with torch.no_grad():
                    sdf_gt = self.get_gt_sdf(points_unscaled)
                geo_loss = F.mse_loss(sdf, sdf_gt)
            else:
                coarse2fine = False
                num_sub_sample = 20
                rad_sub_sample = 1e-2
                if random_sdf:
                    points_rand = (
                            torch.rand((10000, 3), dtype=torch.float32).to(self.device) * 2.0 - 1.0
                        )
                    if coarse2fine:
                        num_sampling = min(points_rand.shape[0],10000)
                        index = random.sample(range(points_rand.shape[0]), num_sampling)
                        b = points_rand[index]
                        b = b.unsqueeze(1).repeat([1, num_sub_sample, 1]).reshape(-1, 3)
                        b = b + torch.randn_like(b, device=b.device) * rad_sub_sample
                        sdf_pred = self.forward_sdf(b).reshape(num_sampling, -1, 1).mean(dim=1)
                        with torch.no_grad():
                            sdf_gt = self.get_gt_sdf(points_rand[index])
                        geo_loss = F.l1_loss(sdf_pred, sdf_gt)
                    else:
                        sdf_pred = self.forward_sdf(points_rand, debug_using_gt=False)
                        # sdf_pred.register_hook(print)
                        with torch.no_grad():
                            sdf_gt = self.get_gt_sdf(points_rand)
                        geo_loss = F.l1_loss(sdf_pred, sdf_gt)
                else:
                    if coarse2fine:
                        num_sampling = min(points_unscaled.shape[0],10000)
                        index = random.sample(range(points_unscaled.shape[0]), num_sampling)
                        b = points_unscaled[index]
                        b = b.unsqueeze(1).repeat([1, num_sub_sample, 1]).reshape(-1, 3)
                        b = b + torch.randn_like(b, device=b.device) * rad_sub_sample
                        sdf_pred = self.forward_sdf(b).reshape(num_sampling, -1, 1).mean(dim=1)
                        with torch.no_grad():
                            sdf_gt = self.get_gt_sdf(points_unscaled[index])
                        geo_loss = F.l1_loss(sdf_pred, sdf_gt)
                    else:
                        with torch.no_grad():
                            sdf_gt = self.get_gt_sdf(points_unscaled)
                        geo_loss = F.l1_loss(sdf, sdf_gt)
            output.update({"sdf_loss": geo_loss})

        # SDF Smooth Loss
        if self.training and self.cfg.out_sdf_smooth_loss:
            points_perturb = points_unscaled + torch.randn_like(points_unscaled) * 1e-2
            sdf_prerturb = self.forward_sdf(points_perturb)
            sdf_smooth_loss = (sdf - sdf_prerturb).abs().mean()
            output.update({"sdf_smooth_loss": sdf_smooth_loss})

        # predicting appearance
        if self.cfg.n_feature_dims > 0: 
            if self.encoding_rgb is not None:
                enc_rgb = self.encoding_rgb(points.view(-1, self.cfg.n_input_dims))
                features = self.feature_network(enc_rgb).view(
                    *points.shape[:-1], self.cfg.n_feature_dims
                )
            else:
                if self.run_sharing:
                    features = sdf_fea[..., 1:]
                else:
                    features = self.feature_network(enc).view(
                        *points.shape[:-1], self.cfg.n_feature_dims
                    )
            output.update({"features": features})

        if output_normal:
            if (
                self.cfg.normal_type == "finite_difference"
                or self.cfg.normal_type == "finite_difference_laplacian"
            ):
                assert self.finite_difference_normal_eps is not None
                eps: float = self.finite_difference_normal_eps
                if self.cfg.normal_type == "finite_difference_laplacian":
                    offsets: Float[Tensor, "6 3"] = torch.as_tensor(
                        [
                            [eps, 0.0, 0.0],
                            [-eps, 0.0, 0.0],
                            [0.0, eps, 0.0],
                            [0.0, -eps, 0.0],
                            [0.0, 0.0, eps],
                            [0.0, 0.0, -eps],
                        ]
                    ).to(points_unscaled)
                    points_offset: Float[Tensor, "... 6 3"] = (
                        points_unscaled[..., None, :] + offsets
                    ).clamp(-self.cfg.radius, self.cfg.radius)
                    sdf_offset: Float[Tensor, "... 6 1"] = self.forward_sdf(
                        points_offset
                    )
                    sdf_grad = (
                        0.5
                        * (sdf_offset[..., 0::2, 0] - sdf_offset[..., 1::2, 0])
                        / eps
                    )
                else:
                    offsets: Float[Tensor, "3 3"] = torch.as_tensor(
                        [[eps, 0.0, 0.0], [0.0, eps, 0.0], [0.0, 0.0, eps]]
                    ).to(points_unscaled)
                    points_offset: Float[Tensor, "... 3 3"] = (
                        points_unscaled[..., None, :] + offsets
                    ).clamp(-self.cfg.radius, self.cfg.radius)
                    B = points_offset.shape[0]
                    sdf_offset: Float[Tensor, "... 3 1"] = self.forward_sdf(
                        points_offset.reshape(-1, 3)
                    ).reshape(B, -1, 1)
                    sdf_grad = (sdf_offset[..., 0::1, 0] - sdf) / eps
                normal = F.normalize(sdf_grad, dim=-1)
            elif self.cfg.normal_type == "pred":
                normal = self.normal_network(enc).view(*points.shape[:-1], 3)
                normal = F.normalize(normal, dim=-1)
                sdf_grad = normal
            elif self.cfg.normal_type == "analytic":
                sdf_grad = -torch.autograd.grad(
                    sdf,
                    points_unscaled,
                    grad_outputs=torch.ones_like(sdf),
                    create_graph=True,
                )[0]
                normal = F.normalize(sdf_grad, dim=-1)
                if not grad_enabled:
                    sdf_grad = sdf_grad.detach()
                    normal = normal.detach()
            else:
                raise AttributeError(f"Unknown normal type {self.cfg.normal_type}")
            output.update(
                {"normal": normal, "shading_normal": normal, "sdf_grad": sdf_grad}
            )
        return output

    def forward_sdf(self, points: Float[Tensor, "*N Di"], debug_using_gt=False) -> Float[Tensor, "*N 1"]:
        points_unscaled = points
        points = contract_to_unisphere(points_unscaled, self.bbox, self.unbounded)

        if debug_using_gt:
            sdf = self.get_gt_sdf(points_unscaled)   # Eckert: debug using gt sdf
        else:
            enc = self.encoding(points.view(-1, self.cfg.n_input_dims))
            if self.run_common_frame:
                enc = self.common_net(enc)
            if self.run_hierhashgrid:
                sdf = self.sdf_network(enc[:, :self.in_dim_sdf]).view(*points.shape[:-1], 1)
            else:
                if self.run_sharing:
                    sdf_fea = self.sdf_network(enc).view(*points.shape[:-1], 4)
                    sdf = sdf_fea[..., :1]
                else:
                    sdf = self.sdf_network(enc).view(*points.shape[:-1], 1)
            sdf = self.get_shifted_sdf(points_unscaled, sdf)
            if self.cfg.predict_offset:
                with torch.no_grad():
                    sdf_base = self.get_gt_sdf(points_unscaled)
                sdf = sdf_base + sdf
        return sdf

    def forward_field(
        self, points: Float[Tensor, "*N Di"]
    ) -> Tuple[Float[Tensor, "*N 1"], Optional[Float[Tensor, "*N 3"]]]:
        points_unscaled = points
        points = contract_to_unisphere(points_unscaled, self.bbox, self.unbounded)
        enc = self.encoding(points.reshape(-1, self.cfg.n_input_dims))
        if self.run_common_frame:
            enc = self.common_net(enc)
        if self.run_hierhashgrid:
            sdf = self.sdf_network(enc[:, :self.in_dim_sdf]).view(*points.shape[:-1], 1)
        else:
            if self.run_sharing:
                sdf_fea = self.sdf_network(enc).view(*points.shape[:-1], 4)
                sdf = sdf_fea[..., :1]
            else:
                sdf = self.sdf_network(enc).view(*points.shape[:-1], 1)
        sdf = self.get_shifted_sdf(points_unscaled, sdf)
        if self.cfg.predict_offset:
            with torch.no_grad():
                sdf_base = self.get_gt_sdf(points_unscaled)
            sdf = sdf_base + sdf
        deformation: Optional[Float[Tensor, "*N 3"]] = None
        if self.cfg.isosurface_deformable_grid:
            deformation = self.deformation_network(enc).reshape(*points.shape[:-1], 3)
        return sdf, deformation

    def forward_level(
        self, field: Float[Tensor, "*N 1"], threshold: float
    ) -> Float[Tensor, "*N 1"]:
        return field - threshold
    
    def export(self, points: Float[Tensor, "*N Di"], **kwargs) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.cfg.n_feature_dims == 0:
            return out
        points_unscaled = points
        points = contract_to_unisphere(points_unscaled, self.bbox, self.unbounded)
        if self.run_individual:
            enc_rgb = self.encoding_rgb(points.view(-1, self.cfg.n_input_dims))
            features = self.feature_network(enc_rgb).view(
                *points.shape[:-1], self.cfg.n_feature_dims
            )
        else:
            enc = self.encoding(points.reshape(-1, self.cfg.n_input_dims))
            if self.run_common_frame:
                enc = self.common_net(enc)
            if self.run_sharing:
                sdf_fea = self.sdf_network(enc).view(*points.shape[:-1], 4)
                features = sdf_fea[..., 1:]
            else:
                features = self.feature_network(enc).view(
                    *points.shape[:-1], self.cfg.n_feature_dims
                )
        out.update(
            {
                "features": features,
            }
        )
        return out

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        if (
            self.cfg.normal_type == "finite_difference"
            or self.cfg.normal_type == "finite_difference_laplacian"
        ):
            if isinstance(self.cfg.finite_difference_normal_eps, float):
                self.finite_difference_normal_eps = (
                    self.cfg.finite_difference_normal_eps
                )
            elif self.cfg.finite_difference_normal_eps == "progressive":
                # progressive finite difference eps from Neuralangelo
                # https://arxiv.org/abs/2306.03092
                hg_conf: Any = self.cfg.pos_encoding_config
                assert (
                    hg_conf.otype == "ProgressiveBandHashGrid"
                ), "finite_difference_normal_eps=progressive only works with ProgressiveBandHashGrid"
                current_level = min(
                    hg_conf.start_level
                    + max(global_step - hg_conf.start_step, 0) // hg_conf.update_steps,
                    hg_conf.n_levels,
                )
                grid_res = hg_conf.base_resolution * hg_conf.per_level_scale ** (
                    current_level - 1
                )
                grid_size = 2 * self.cfg.radius / grid_res
                if grid_size != self.finite_difference_normal_eps:
                    threestudio.info(
                        f"Update finite_difference_normal_eps to {grid_size}"
                    )
                self.finite_difference_normal_eps = grid_size
            else:
                raise ValueError(
                    f"Unknown finite_difference_normal_eps={self.cfg.finite_difference_normal_eps}"
                )
