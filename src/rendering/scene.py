
import torch
import torch.nn as nn

from kaolin.render.camera import Camera
import nvdiffrast.torch as dr

from jaxtyping import jaxtyped, Int32, Float32
from beartype import beartype
from typing import cast

from .point_lights import PointLights


from src.geometry import NURBS

class Scene(nn.Module):
    """
    A differentiable Scene composed of NURBS rendered by Cameras.

    Attributes:
        nurbs (nn.ModuleList): Collection of NURBS modules that describe the
            geometry and albedo of the scene objects.

        cameras (Camera): Instance of Camera that represent the batched point of view
            where the scene can be seen.
            Shape: (Batch_Size,)

        lights (SgLightingParameters): Light of the scene encoded in Spherical Gaussian
            Shape: (Batch_Size,)

        background_rgb (Tensor): RGB tensor that represents the background use for rendering
            images (not use in loss).
            Shape: (3,)

    """

    @jaxtyped(typechecker=beartype)
    def __init__(
            self,
            nurbs: list[NURBS] | nn.ModuleList,
            cameras: Camera,
            lights: PointLights,
            background_rgb: Float32[torch.Tensor, "3"],
    ) -> None:
        super().__init__()

        self._validate_integrity(nurbs, cameras, background_rgb)

        # --- Learnable Parameters ---
        if isinstance(nurbs, nn.ModuleList):
            self.nurbs = nurbs
        else:
            self.nurbs = nn.ModuleList(nurbs)

        # --- Rendering Tools ---
        self.cameras = cameras
        self.lights = lights
        self.register_buffer('background_rgb', background_rgb)
        self.ctx = dr.RasterizeCudaContext()

    @jaxtyped(typechecker=beartype)
    def _validate_integrity(
            self,
            nurbs: list[NURBS],
            cameras: Camera,
            background_rgb: Float32[torch.Tensor, "3"],
    )-> None:
        """
        Ensures the validity of the components

        Args:
            nurbs (nn.ModuleList): NURBS instance.

            cameras (Camera): Instance of Kaolin Camera batched.
                Shape: (Batch_Size,)

            background_rgb (Tensor): RGB tensor that represents the background use for rendering
                images (not use in loss).
                Shape: (3,)
        Raises:
            ValueError: If any object is empty or has a length of 0
            RuntimeError: If all the components are not on the same device
        """

        if len(nurbs) == 0:
            raise ValueError("NURBS list is empty")

        if len(cameras) == 0:
            raise ValueError("Camera batch is empty.")

        if cameras.device != nurbs[0].device:
            raise RuntimeError(
                f"Device Mismatch: NURBS are on {nurbs[0].device} but Cameras are on {cameras.device}. "
                "All components must be on the same device for rendering."
            )

        if background_rgb.shape != (3,):
            raise ValueError("Background should be encoded with 3 channels (RGB).")

        if torch.any(background_rgb > 255) or torch.any(background_rgb < 0):
            raise ValueError("Background should be in RGB in 8 bits that implies values "
                             "between 0 and 255 (included)."
            )

    @jaxtyped(typechecker=beartype)
    def forward(
            self,
            discretization_u,
            discretization_v,
            #epsilon: float,
            cameras_idx: Int32[torch.Tensor, "num_camera"]
    ):
        (
            vertices,
            faces,
            obj_indices,
            normals,
            albedo_palette,
            roughness_palette,
            metalness_palette
        ) = self._aggregate_geometry(discretization_u, discretization_v)

        cameras = self.cameras[cameras_idx]
        clip_space_vertices = self._view_transform(cameras, vertices)

        cameras_space_vertices = cameras.extrinsics.transform(vertices)
        depth_camera_space = cameras_space_vertices[..., 2:3]

        rast_out, rast_db = dr.rasterize(
            self.ctx,
            clip_space_vertices,
            faces,
            resolution=(cameras.height, cameras.width)
        )

        ones = torch.ones(
            (rast_out.shape[0], rast_out.shape[1], rast_out.shape[2], 1),
            dtype=torch.float32, device=self.device
        )
        mask = dr.antialias(ones, rast_out, clip_space_vertices, faces)

        features = torch.cat([
            vertices.unsqueeze(0).expand(len(cameras), -1, -1),
            albedo_palette[obj_indices].unsqueeze(0).expand(len(cameras), -1, -1),
            roughness_palette[obj_indices].unsqueeze(0).expand(len(cameras), -1, -1),
            metalness_palette[obj_indices].unsqueeze(0).expand(len(cameras), -1, -1),
            normals.unsqueeze(0).expand(len(cameras), -1, -1),
            depth_camera_space
        ], dim=-1)

        interpolated_features, _ = dr.interpolate(features.contiguous(), rast_out, faces)

        interpolated_vertices = interpolated_features[..., :3]
        interpolated_albedo = interpolated_features[..., 3:6]
        interpolated_roughness = interpolated_features[..., 6:7]
        interpolated_metalness = interpolated_features[..., 7:8]
        interpolated_normals = torch.nn.functional.normalize(interpolated_features[..., 8:11], dim=-1)
        interpolated_depth = interpolated_features[..., 11:14]

        shaded = self.lights.shade(
            interpolated_vertices,
            interpolated_normals,
            interpolated_albedo,
            interpolated_roughness,
            interpolated_metalness,
            cameras.extrinsics.cam_pos().squeeze(-1)
        )

        def aces_tone_mapping(x):
            a = 2.51
            b = 0.03
            c = 2.43
            d = 0.59
            e = 0.14
            return (x * (a * x + b)) / (x * (c * x + d) + e)

        img = shaded * mask + (1 - mask) * self.background_rgb
        img = aces_tone_mapping(img)
        img = torch.pow(img.clamp(0.001, 1.0), 1.0/2.2)
        img = dr.antialias(img.contiguous(), rast_out, clip_space_vertices, faces)
        return img, interpolated_depth, rast_out


    @property
    @jaxtyped(typechecker=beartype)
    def device(self) -> torch.device:
        """Device of the instance"""
        return self.cameras.device


    def to(self, *args, **kwargs) -> "Scene":
        """
        Moves and/or casts the parameters and buffers of the scene.

        This overrides the default nn.Module.to to ensure that the Kaolin Camera
        object, which is not a nn.Module, is correctly moved to the destination
        device and cast to the desired dtype.

        Args:
            *args: Floating point numbers, torch.device, or torch.dtype to cast/move tensors.

            **kwargs: Keyword arguments such as 'device', 'dtype', or 'non_blocking'.

        Returns:
            self: The instance with moved/cast components.
        """

        super().to(*args, **kwargs)
        self.cameras = self.cameras.to(*args, **kwargs)

        return self


    @jaxtyped(typechecker=beartype)
    def _aggregate_geometry(
            self,
            discretization_u,
            discretization_v
            #epsilon
    ) -> tuple[
        Float32[torch.Tensor, "vertices 3"],
        Int32[torch.Tensor, "triangles 3"],
        Int32[torch.Tensor, "vertices"],
        Float32[torch.Tensor, "vertices 3"],
        Float32[torch.Tensor, "num_objects 3"],
        Float32[torch.Tensor, "num_objects 1"],
        Float32[torch.Tensor, "num_objects 1"]
    ]:
        """
        Discretizes all NURBS in the scene and merges them into tensors of vertices, indices and features.
        Args:
            discretization_u (Tensor): Tensor of points to evaluate along the u axis.
                Shape: (Num_U,).

            discretization_v: Tensor of points to evaluate along the v axis.
                Shape: (Num_V,).

        Returns:
            tuple: Composed of (vertices, faces, obj_indices, face_normals, albedo_palette) containing:

            - **vertices** (Tensor):Geometry vertices in World Space.
                Shape: (Vertices, 3).

            - **faces** (Tensor): Global topology indices for triangles.
                Shape: (Triangles, 3).

            - **obj_indices** (Tensor): Object index for each vertice (for deferred shading).
                Shape: (Triangles,).

            - **face_normals** (Tensor): Geometric normal of each triangle (Flat Shading).
                Shape: (Triangles, 3).

            - **albedo_palette** (Tensor): RGB color palette for the objects.
                Shape: (N_obj, 3).
        """

        # Lists that will contain the data and then convert into tensors
        vertices = []
        faces = []
        normals = []
        indices = []
        albedo = []
        roughness = []
        metalness = []

        offset = 0

        # Iterate over scene objects
        for index, module in enumerate(self.nurbs):
            nurbs = cast(NURBS, module)
            vertices_k, faces_k, normals_k = nurbs.forward(discretization_u, discretization_v)

            vertices.append(vertices_k)
            faces.append(faces_k + offset)
            normals.append(normals_k)
            indices.append(torch.full((vertices_k.shape[0],), index, dtype=torch.int32, device=self.device))
            albedo.append(nurbs.albedo_rgb)
            roughness.append(nurbs.roughness)
            metalness.append(nurbs.metalness)

            offset += vertices_k.shape[0]

        # Convert lists into tensors
        vertices = torch.cat(vertices, dim=0)
        faces = torch.cat(faces, dim=0)
        normals = torch.cat(normals, dim=0)
        indices = torch.cat(indices, dim=0)
        albedo = torch.stack(albedo, dim=0)
        roughness = torch.stack(roughness, dim=0)
        metalness = torch.stack(metalness, dim=0)


        return vertices, faces, indices, normals, albedo, roughness, metalness


    @jaxtyped(typechecker=beartype)
    def _view_transform(
            self,
            cameras: Camera,
            vertices: Float32[torch.Tensor, "vertices 3"]
    ) ->  Float32[torch.Tensor, "num_camera vertices 4"]:
        """
        Project vertices and normals into the camera's world.

        Args:
            cameras (Camera): Batch of camera where the points are going to be projected.
                Shape: (Batch_Size,).

            vertices (Tensor): Vertices.
                Shape: (Vertices, 3).

        Returns:
            tuple: Composed of (projected_vertices, projected_normals) containing:

            - **projected_vertices** (Tensor): Vertices projected on the batch of cameras.
                Shape: (Batch_Size, Vertices, 3).
        """

        cameras_space_vertices = cameras.extrinsics.transform(vertices)
        clip_space_vertices = cameras.intrinsics.project(cameras_space_vertices)

        if clip_space_vertices.ndim == 2:
            clip_space_vertices = clip_space_vertices.unsqueeze(0)

        return clip_space_vertices
