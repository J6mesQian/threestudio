# start here
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.geometry.base import BaseImplicitGeometry, contract_to_unisphere
from threestudio.utils.ops import get_activation
from threestudio.utils.typing import *


@threestudio.register("triplane-volume")
class TriplaneVolume(BaseImplicitGeometry):
    @dataclass
    class Config(BaseImplicitGeometry.Config):
        # Size for each of the three feature planes
        plane_resolution: Tuple[int, int] = field(default_factory=lambda: (256, 256))
        # Total feature dimensions to store per plane
        n_feature_dims: int = 3
        density_activation: Optional[str] = "softplus"
        density_bias: Union[float, str] = "blob"
        density_blob_scale: float = 5.0
        density_blob_std: float = 0.5
        normal_type: Optional[
            str
        ] = "finite_difference"  # in ['pred', 'finite_difference', 'finite_difference_laplacian']

        # automatically determine the threshold
        isosurface_threshold: Union[float, str] = "auto"
        
        # MLP decoder configuration
        use_feature_decoder: bool = False
        feature_decoder_layers: int = 2
        feature_decoder_hidden_dim: int = 64
        feature_decoder_output_dim: int = 3  # Final feature dimension after MLP decoding

    cfg: Config

    def configure(self) -> None:
        super().configure()
        # Define the three planes (xy, xz, yz)
        # Each plane stores density (1 channel) + features
        # Dimensions are [batch, features, height, width]
        n_dims_per_plane = self.cfg.n_feature_dims + 1
        
        # Create the three planes - each has density + feature channels
        self.xy_plane = nn.Parameter(
            torch.zeros(1, n_dims_per_plane, *self.cfg.plane_resolution)
        )
        self.xz_plane = nn.Parameter(
            torch.zeros(1, n_dims_per_plane, *self.cfg.plane_resolution)
        )
        self.yz_plane = nn.Parameter(
            torch.zeros(1, n_dims_per_plane, *self.cfg.plane_resolution)
        )
        
        # Define a scaling parameter for density
        if self.cfg.density_bias == "blob":
            self.register_buffer("density_scale", torch.tensor(0.0))
        else:
            self.density_scale = nn.Parameter(torch.tensor(0.0))

        # If using predicted normals, we need three more planes for normal directions
        if self.cfg.normal_type == "pred":
            self.xy_normal_plane = nn.Parameter(torch.zeros(1, 3, *self.cfg.plane_resolution))
            self.xz_normal_plane = nn.Parameter(torch.zeros(1, 3, *self.cfg.plane_resolution))
            self.yz_normal_plane = nn.Parameter(torch.zeros(1, 3, *self.cfg.plane_resolution))
        
        # Initialize the MLP decoder if specified
        if self.cfg.use_feature_decoder:
            self.feature_decoder = self.create_mlp_decoder(
                input_dim=self.cfg.n_feature_dims,
                hidden_dim=self.cfg.feature_decoder_hidden_dim,
                output_dim=self.cfg.feature_decoder_output_dim,
                num_layers=self.cfg.feature_decoder_layers
            )

    def create_mlp_decoder(self, input_dim, hidden_dim, output_dim, num_layers):
        """Create an MLP decoder for processing features"""
        layers = []
        
        # First layer: input_dim to hidden_dim
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        # Final layer: hidden_dim to output_dim
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        return nn.Sequential(*layers)

    def get_density_bias(self, points: Float[Tensor, "*N Di"]):
        if self.cfg.density_bias == "blob":
            # Similar to volume grid implementation
            density_bias: Float[Tensor, "*N 1"] = (
                self.cfg.density_blob_scale
                * (
                    1
                    - torch.sqrt((points.detach() ** 2).sum(dim=-1))
                    / self.cfg.density_blob_std
                )[..., None]
            )
            return density_bias
        elif isinstance(self.cfg.density_bias, float):
            return self.cfg.density_bias
        else:
            raise AttributeError(f"Unknown density bias {self.cfg.density_bias}")

    def sample_plane(
        self, points: Float[Tensor, "*N 2"], plane: Float[Tensor, "1 C H W"]
    ) -> Float[Tensor, "*N C"]:
        """Sample features from a plane using bilinear interpolation"""
        # points should be in range [-1, 1] for grid_sample
        points_shape = points.shape[:-1]
        c = plane.shape[1]
        
        # Reshape for grid_sample which expects 4D or 5D input
        # Add batch dim and height dim (set to 1)
        points_grid = points.view(1, 1, -1, 2)
        
        # Sample from the plane
        sampled = F.grid_sample(
            plane, points_grid, align_corners=False, mode="bilinear"
        )
        
        # Reshape to match original shape
        sampled = sampled.reshape(c, -1).T.reshape(*points_shape, c)
        return sampled

    def triplane_feature(
        self, points: Float[Tensor, "*N 3"]
    ) -> Float[Tensor, "*N C"]:
        """Sample features from all three planes and combine them"""
        # Extract 2D coordinates for each plane
        xy_coords = points[..., :2]  # Extract x, y coordinates
        xz_coords = torch.stack([points[..., 0], points[..., 2]], dim=-1)  # Extract x, z coordinates
        yz_coords = points[..., 1:]  # Extract y, z coordinates

        # Sample from each plane
        xy_features = self.sample_plane(xy_coords, self.xy_plane)
        xz_features = self.sample_plane(xz_coords, self.xz_plane)
        yz_features = self.sample_plane(yz_coords, self.yz_plane)

        # Combine features (sum or mean)
        # Using sum as it's common in NeRF-like triplane implementations
        features = xy_features + xz_features + yz_features
        
        return features

    def triplane_normal(
        self, points: Float[Tensor, "*N 3"]
    ) -> Float[Tensor, "*N 3"]:
        """Sample normal vectors from all three normal planes, if available"""
        if self.cfg.normal_type != "pred":
            raise ValueError("Normal planes are only used with normal_type='pred'")
            
        # Extract 2D coordinates for each plane
        xy_coords = points[..., :2]  # Extract x, y coordinates
        xz_coords = torch.stack([points[..., 0], points[..., 2]], dim=-1)  # Extract x, z coordinates
        yz_coords = points[..., 1:]  # Extract y, z coordinates

        # Sample from each normal plane
        xy_normal = self.sample_plane(xy_coords, self.xy_normal_plane)
        xz_normal = self.sample_plane(xz_coords, self.xz_normal_plane)
        yz_normal = self.sample_plane(yz_coords, self.yz_normal_plane)

        # Combine normal vectors
        normal = xy_normal + xz_normal + yz_normal
        
        return normal

    def forward(
        self, points: Float[Tensor, "*N Di"], output_normal: bool = False
    ) -> Dict[str, Float[Tensor, "..."]]:
        points_unscaled = points  # points in the original scale
        points = contract_to_unisphere(
            points, self.bbox, self.unbounded
        )  # points normalized to (0, 1)
        points = points * 2 - 1  # convert to [-1, 1] for grid sample

        # Get features from triplane representation
        out = self.triplane_feature(points)
        density, features = out[..., 0:1], out[..., 1:]
        
        # Apply density scaling
        density = density * torch.exp(self.density_scale)  # exp scaling like in DreamFusion

        # Apply MLP decoder to features if specified
        if self.cfg.use_feature_decoder and hasattr(self, 'feature_decoder'):
            features = self.feature_decoder(features)

        # Apply density activation with bias
        density = get_activation(self.cfg.density_activation)(
            density + self.get_density_bias(points_unscaled)
        )

        output = {
            "density": density,
            "features": features,
        }

        if output_normal:
            if (
                self.cfg.normal_type == "finite_difference"
                or self.cfg.normal_type == "finite_difference_laplacian"
            ):
                eps = 1.0e-3
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
                    density_offset: Float[Tensor, "... 6 1"] = self.forward_density(
                        points_offset
                    )
                    normal = (
                        -0.5
                        * (density_offset[..., 0::2, 0] - density_offset[..., 1::2, 0])
                        / eps
                    )
                else:
                    offsets: Float[Tensor, "3 3"] = torch.as_tensor(
                        [[eps, 0.0, 0.0], [0.0, eps, 0.0], [0.0, 0.0, eps]]
                    ).to(points_unscaled)
                    points_offset: Float[Tensor, "... 3 3"] = (
                        points_unscaled[..., None, :] + offsets
                    ).clamp(-self.cfg.radius, self.cfg.radius)
                    density_offset: Float[Tensor, "... 3 1"] = self.forward_density(
                        points_offset
                    )
                    normal = -(density_offset[..., 0::1, 0] - density) / eps
                normal = F.normalize(normal, dim=-1)
            elif self.cfg.normal_type == "pred":
                normal = self.triplane_normal(points)
                normal = F.normalize(normal, dim=-1)
            else:
                raise AttributeError(f"Unknown normal type {self.cfg.normal_type}")
            output.update({"normal": normal, "shading_normal": normal})
        
        return output

    def forward_density(self, points: Float[Tensor, "*N Di"]) -> Float[Tensor, "*N 1"]:
        points_unscaled = points
        points = contract_to_unisphere(points_unscaled, self.bbox, self.unbounded)
        points = points * 2 - 1  # convert to [-1, 1] for grid sample

        # Sample from triplane to get density
        out = self.triplane_feature(points)
        density = out[..., 0:1]  # Just extract density channel
        
        # Apply density scaling
        density = density * torch.exp(self.density_scale)

        # Apply density activation with bias
        density = get_activation(self.cfg.density_activation)(
            density + self.get_density_bias(points_unscaled)
        )
        
        return density

    def forward_field(
        self, points: Float[Tensor, "*N Di"]
    ) -> Tuple[Float[Tensor, "*N 1"], Optional[Float[Tensor, "*N 3"]]]:
        if getattr(self.cfg, "isosurface_deformable_grid", False):
            threestudio.warn(
                f"{self.__class__.__name__} does not support isosurface_deformable_grid. Ignoring."
            )
        density = self.forward_density(points)
        return density, None

    def forward_level(
        self, field: Float[Tensor, "*N 1"], threshold: float
    ) -> Float[Tensor, "*N 1"]:
        return -(field - threshold)

    def export(self, points: Float[Tensor, "*N Di"], **kwargs) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.cfg.n_feature_dims == 0:
            return out
            
        points_unscaled = points
        points = contract_to_unisphere(points, self.bbox, self.unbounded)
        points = points * 2 - 1  # convert to [-1, 1] for grid sample
        
        # Get features from triplane
        features = self.triplane_feature(points)[..., 1:]
        
        # Apply MLP decoder to features if specified
        if self.cfg.use_feature_decoder and hasattr(self, 'feature_decoder'):
            features = self.feature_decoder(features)
        
        out.update(
            {
                "features": features,
            }
        )
        return out