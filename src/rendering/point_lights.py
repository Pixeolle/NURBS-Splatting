
import torch
import torch.nn as nn

from jaxtyping import jaxtyped, Float32
from beartype import beartype

class PointLights(nn.Module):

    @jaxtyped(typechecker=beartype)
    def __init__(
            self,
            positions: Float32[torch.Tensor, "L 3"],
            colors: Float32[torch.Tensor, "L 3"],
            intensities: Float32[torch.Tensor, "L"],
            sharpness: Float32[torch.Tensor, "L"]
    ):
        super().__init__()

        self.register_buffer('positions', positions)
        self.register_buffer('colors', colors)
        self.register_buffer('intensities', intensities)
        self.register_buffer('sharpness', sharpness)

    @property
    @jaxtyped(typechecker=beartype)
    def device(self) -> torch.device:
        """Device of the instance"""
        return self.positions.device

    @jaxtyped(typechecker=beartype)
    def shade(
            self,
            positions,
            normals,
            albedo,
            roughness,
            metalness,
            cameras_positions
    ):

        positions = positions.unsqueeze(1)
        normals = normals.unsqueeze(1)
        albedo = albedo.unsqueeze(1)
        roughness = roughness.unsqueeze(1)
        metalness = metalness.unsqueeze(1)
        cameras_positions = cameras_positions[:, None, None, None, :]

        cameras_direction = torch.nn.functional.normalize(cameras_positions - positions, dim=-1)
        lights_direction = torch.nn.functional.normalize(self.positions[None, :, None, None] - positions, dim=-1)

        half_vector = torch.nn.functional.normalize(cameras_direction + lights_direction, dim=-1)

        f0 = (1 - metalness) * 0.04 + metalness * albedo

        def soft_clamp(x, min_val=0.0, sharpness=10.0):
            return torch.nn.functional.softplus((x - min_val) * sharpness) / sharpness + min_val


        n_dot_l = soft_clamp(torch.einsum("...i, ...i -> ...", normals, lights_direction)).unsqueeze(-1)
        n_dot_v = soft_clamp(torch.einsum("...i, ...i -> ...", normals, cameras_direction)).unsqueeze(-1)
        n_dot_h = soft_clamp(torch.einsum("...i, ...i -> ...", normals, half_vector)).unsqueeze(-1)
        v_dot_h = soft_clamp(torch.einsum("...i, ...i -> ...", cameras_direction, half_vector)).unsqueeze(-1)

        a_r_2 = roughness**4
        a_r_2 = a_r_2

        d = a_r_2 / (torch.pi * (n_dot_h**2 * (a_r_2 - 1) + 1)**2)

        f = f0 + (1 - f0) * torch.pow(1 - v_dot_h, 5)

        k = (roughness + 1) ** 2 / 8

        def g_1(x):
            return x / (x * (1 - k) + k)

        g = g_1(n_dot_l) * g_1(n_dot_v)

        f_spec = d * f * g / (4 * n_dot_l * n_dot_v + 1e-4)

        k_d = (1 - f) * (1 - metalness)
        f_diff = k_d * albedo / torch.pi

        attenuation = self.intensities[None, :, None, None, None] / (torch.norm(self.positions[None, :, None, None] - positions, dim=-1)**1.5)[..., None]

        a = (f_diff + f_spec) * self.colors[None, :, None, None, :]
        b = attenuation * n_dot_l

        l_out = a * b
        l_out = l_out.sum(dim=1)
        return l_out