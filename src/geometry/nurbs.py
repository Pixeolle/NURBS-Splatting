import torch
import torch.nn as nn

from jaxtyping import jaxtyped, Float32, Int64, Int32
from beartype import beartype
from typing import Union

class NURBS(nn.Module):
    """
    A differentiable NURBS surface layer with cyclic topology support.

    The U-dimension is treated as the primary axis (longitudinal),
    while the V-dimension represents the auxiliary layers (transversal).

    The knot vector is reconstructed using a 'Softmax Sandwich' strategy:
    internal intervals are learned as raw deltas, normalized via softmax
    to ensure monotonicity, and then clamped between (p+1) zeros and ones.

    Attributes:
        control_points (Tensor): Unique XYZ coordinates (Rational B-Splines).
            Shape: (N_points, 3).

        index_map (Tensor): Topology map linking unique points to the (U, V) grid.
            Shape: (U, V).

        weights_map (Tensor): Tensor linking unique points to a weight.
            Shape: (U, V).

        knot_deltas_u (Tensor): Interval parameters for the U-axis. Vector used
            to compute internal knot positions.
            Shape: (U - degree + 1,).

        knot_deltas_v (Tensor): Same as `knot_deltas_u` but for the V-axis.
            Shape: (V - degree,).

        albedo_rgb (Tensor): Surface color in RGB [0, 255].
            Shape: (3,).

        degree (int): Polynomial degree of the surface.
    """

    @jaxtyped(typechecker=beartype)
    def __init__(
            self,
            control_points: Float32[torch.Tensor, "N 3"],
            index_map: Int64[torch.Tensor, "num_u num_v"],
            weights_map: Float32[torch.Tensor, "num_u num_v"],
            knot_deltas_u: Float32[torch.Tensor, "num_intervals_u"],
            knot_deltas_v: Float32[torch.Tensor, "num_intervals_v"],
            albedo_rgb: Float32[torch.Tensor, "3"],
            roughness: float | Float32[torch.Tensor, "1"],
            metalness: float | Float32[torch.Tensor, "1"],
            degree: int = 2,
    ) -> None:
        super().__init__()

        self._validate_integrity(
            control_points,
            index_map,
            weights_map,
            knot_deltas_u,
            knot_deltas_v,
            albedo_rgb,
            degree
        )

        # --- Geometric Parameters ---
        self.control_points = nn.Parameter(control_points)
        self.weights_map = nn.Parameter(weights_map)
        self.knot_deltas_u = nn.Parameter(knot_deltas_u)
        self.knot_deltas_v = nn.Parameter(knot_deltas_v)

        # --- Buffers & Constants ---
        self.register_buffer('index_map', index_map)
        self.degree = degree

        # --- Appearance ---
        self.albedo_rgb = nn.Parameter(albedo_rgb)
        if isinstance(roughness, torch.Tensor):
            self.register_buffer('_roughness', roughness)
        else:
            self.register_buffer('_roughness', torch.tensor(roughness))

        if isinstance(metalness, torch.Tensor):
            self.register_buffer('_metalness', metalness)
        else:
            self.register_buffer('_metalness', torch.tensor(metalness))

    @jaxtyped(typechecker=beartype)
    def _validate_integrity(
            self,
            control_points: Float32[torch.Tensor, "N 3"],
            index_map: Int64[torch.Tensor, "num_u num_v"],
            weights_map: Float32[torch.Tensor, "num_u num_v"],
            knot_deltas_u: Float32[torch.Tensor, "num_intervals_u"],
            knot_deltas_v: Float32[torch.Tensor, "num_intervals_v"],
            albedo_rgb: Float32[torch.Tensor, "3"],
            degree: int
    ) -> None:
        """
        Ensures the mathematical and structural coherence of the NURBS parameters.

        Args:
            control_points (Tensor): Unique rational coordinates (XYZ).
                Shape: (N_points, 3).

            index_map (Tensor): Grid mapping used to reconstruct the control net.
                Shape: (U, V).

            weights_map (Tensor): Tensor linking unique points to a weight.
                Shape: (U, V).

            knot_deltas_u (Tensor): Internal knot intervals for the U axis.
                Shape: (U - p + 1,).

            knot_deltas_v (Tensor): Internal knot intervals for the V axis.
                Shape: (V - p,).

            albedo_rgb (Tensor): Surface color vector in 8-bit RGB.
                Shape: (3,).

            degree (int): Polynomial degree of the basis functions.

        Raises:
            ValueError: If any mathematical constraint (like m = n + p + 1) is violated.
            IndexError: If the index_map points to non-existent control points.
        """

        if degree < 1:
            raise ValueError(f"Degree must be at least 1, got {degree}.")

        if len(control_points.shape) != 2:
            raise ValueError(f"Control points should be a 2-rank tensor (N, 3), "
                             f"got {control_points.shape}")

        if control_points.shape[-1] != 3:
            raise ValueError(f"Control points should be encoded with 3 channels (XYZ), "
                             f"got {control_points.shape[-1]}."
            )

        if torch.any(index_map >= control_points.shape[0]):
            raise IndexError(
                f"Index map should not contain index above or equal to the number of "
                f"control points ({control_points.shape[0]}), got {index_map.max().item()}."
            )

        expected_u = index_map.shape[0] - degree + 1
        if knot_deltas_u.shape[0] != expected_u:
            raise ValueError(
                f"U knot intervals mismatch. Expected {expected_u} (num_u - p), "
                f"got {knot_deltas_u.shape[-1]}"
            )

        expected_v = index_map.shape[1] - degree
        if knot_deltas_v.shape[0] != expected_v:
            raise ValueError(
                f"V knot intervals mismatch. Expected {expected_v} (num_v - p), "
                f"got {knot_deltas_v.shape[-1]}"
            )

        if weights_map.shape != index_map.shape:
            raise ValueError(
                f"Index map and weights map should have the same shape "
                f"({weights_map.shape} != {index_map.shape})"
            )

        if albedo_rgb.shape != (3,):
            raise ValueError("Albedo should be encoded with 3 channels (RGB).")

        if torch.any(albedo_rgb > 255) or torch.any(albedo_rgb < 0):
            raise ValueError("Albedo should be in RGB in 8 bits that implies values "
                             "between 0 and 255 (included)."
            )

    @jaxtyped(typechecker=beartype)
    def forward(
            self,
            discretization_u,
            discretization_v
            #epsilon: float
    ) -> tuple[Float32[torch.Tensor, "num_points 3"], Int32[torch.Tensor, "triangles 3"], Float32[torch.Tensor, "num_points 3"]]:
        """
        Tessellate the NURBS according to the discretization given.

        Args:
            discretization_u (Tensor): Tensor of values between 0.0 and 1.0 included that
                represents the resolution of the tessellation in the U-dimension (longitudinal).
                Shape: (Resolution_U,).

            discretization_v (Tensor): Same as `discretization_u` but for the V-dimension (transversal).
                Shape: (Resolution_V,).

        Returns:
            tuple: A tuple containing the mesh data:

            * **vertices** (Tensor): The 3D coordinates of the mesh.
                Shape: (Res_U * Res_V, 3).
            * **faces** (Tensor): The triangle topology indices.
                Shape: ((Res_U-1)*(Res_V-1)*2, 3).
        """

        #discretization_u, discretization_v = self._compute_resolutions(epsilon)

        # Reconstruct knots vector from deltas.
        knots_u = self._get_clamped_knots(self.knot_deltas_u)
        knots_v = self._get_clamped_knots(self.knot_deltas_v)

        # Compute basis matrices.
        basis_u, d_basis_u = self._compute_basis_functions(discretization_u, knots_u, compute_derivatives=True)
        basis_v, d_basis_v = self._compute_basis_functions(discretization_v, knots_v, compute_derivatives=True)

        # Reconstruct the controls points matrix and apply weights to
        # the coordinates of controls points.
        weighted_xyz = self.grid_control_points * self.cyclic_weights_map
        weighted_control_points = torch.cat([weighted_xyz, self.cyclic_weights_map], dim=-1)

        # Compute the tensor product along U and V and then divide by the weight previously multiply
        # by the basis matrices to obtain the surface describes by the NURBS.
        coordinate_4d = torch.einsum('iu, jv, uvw -> ijw', basis_u, basis_v, weighted_control_points)
        weights = coordinate_4d[..., 3:4]
        vertices = self._nurbs_divide(coordinate_4d[..., :3], weights)

        # Compute the triangles indices
        triangles = self._compute_triangles_indices(int(discretization_u.shape[0]), int(discretization_v.shape[0]))

        # Compute the derivative
        d_coord_4d_du = torch.einsum('iu, jv, uvw -> ijw', d_basis_u, basis_v, weighted_control_points)
        dS_du = self._nurbs_divide(d_coord_4d_du[..., :3] - vertices * d_coord_4d_du[..., 3:4], weights)

        d_coord_4d_dv = torch.einsum('iu, jv, uvw -> ijw', basis_u, d_basis_v, weighted_control_points)
        dS_dv = self._nurbs_divide(d_coord_4d_dv[..., :3] - vertices * d_coord_4d_dv[..., 3:4], weights)

        normals = torch.linalg.cross(dS_du, dS_dv, dim=-1)
        normals = torch.nn.functional.normalize(normals, dim=-1)

        normals = normals.reshape(-1, 3)
        vertices = vertices.reshape(-1, 3)

        return vertices, triangles, normals

    @property
    @jaxtyped(typechecker=beartype)
    def device(self) -> torch.device:
        """Device of the instance"""
        return self.control_points.device

    @property
    @jaxtyped(typechecker=beartype)
    def grid_control_points(self) -> Float32[torch.Tensor, "U V 3"]:
        """Dynamically reconstructs the control point grid from the 1D parameter tensor."""
        return self.control_points[self.cyclic_index_map]

    @property
    @jaxtyped(typechecker=beartype)
    def cyclic_index_map(self) -> Int64[torch.Tensor, "num_u_cyclic num_v"]:
        """Return the cyclic index map by copying the first row at the end."""
        return torch.cat([self.index_map, self.index_map[None, 0, :]], dim=0)

    @property
    @jaxtyped(typechecker=beartype)
    def cyclic_weights_map(self) -> Float32[torch.Tensor, "num_u_cyclic num_v 1"]:
        """Return the cyclic weights_map by copying the first row at the end."""
        return torch.cat([self.weights_map, self.weights_map[None, 0, :]], dim=0)[..., None]

    @property
    @jaxtyped(typechecker=beartype)
    def roughness(self) -> Float32[torch.Tensor, "1"]:
        """Return the roughness scaled on [0, 1]"""
        return torch.sigmoid(self._roughness)

    @property
    @jaxtyped(typechecker=beartype)
    def metalness(self) -> Float32[torch.Tensor, "1"]:
        """Return the metalness scaled on [0, 1]"""
        return torch.sigmoid(self._metalness)

    @jaxtyped(typechecker=beartype)
    def _get_clamped_knots(
            self,
            deltas: Float32[torch.Tensor, "num_intervals"]
    ) -> Float32[torch.Tensor, "total_knots"]:
        """
        Constructs a clamped NURBS knot vector from intervals.

        This method applies a 'Sandwich' strategy: it normalizes the deltas
        to [0, 1], computes the cumulative sum for internal knots, and
        pads both ends with (p + 1) repeated values to ensure the surface
        is clamped (interpolates the boundary control points).

        Args:
            deltas (Tensor): Raw interval parameters. Will be normalized via softmax internally
                to ensure monotonicity and range constraints.
                Shape: (Intervals,).

        Returns:
            Tensor: A full knot vector starting with zeros and ending with ones.
                Shape: (num_cp + degree + 1,).
        """

        # Ensure intervals are positive and their sum is equal to 1.0.
        activated = torch.nn.functional.softplus(deltas, beta=10)
        intervals = activated / activated.sum()

        # Reconstruct internal knot timings.
        knot_intervals = torch.cumsum(intervals[:-1], dim=0)

        # Add the clamped composant at the beginning and at the end.
        padding = self.degree + 1
        zeros = torch.zeros(padding, device=self.device)
        ones = torch.ones(padding, device=self.device)

        return torch.cat([zeros, knot_intervals, ones], dim=0)

    @jaxtyped(typechecker=beartype)
    def _compute_triangles_indices(
            self,
            resolution_u: int,
            resolution_v: int
    ) -> Int32[torch.Tensor, "triangles 3"]:
        """
        Compute indices to render the surface.

        Args:
            resolution_u (int): Resolution along the U-dimension.

            resolution_v (int): Resolution along the V-dimension.

        Returns:
            Tensor: Indices that indicate how to render triangles.
                Shape: (Triangles, 3)
        """

        # Create tensor indices in each dimension
        u = torch.arange(resolution_u - 1, dtype=torch.int32, device=self.device)
        v = torch.arange(resolution_v - 1, dtype=torch.int32, device=self.device)

        # Create grid that link u and v
        uu, vv = torch.meshgrid(u, v, indexing='ij')

        # Compute indices for each corner of the square
        bottom_left = uu * resolution_v + vv
        top_left = bottom_left + 1
        bottom_right = bottom_left + resolution_v
        top_right = bottom_right + 1

        # Group corner to create two triangles
        triangle_sup = torch.stack([bottom_left, bottom_right, top_left], dim=-1)
        triangle_inf = torch.stack([bottom_right, top_right, top_left], dim=-1)

        triangles = torch.cat([triangle_inf, triangle_sup], dim=-1).reshape(-1, 3)

        return triangles

    @jaxtyped(typechecker=beartype)
    def _compute_basis_functions(
            self,
            discretization: Float32[torch.Tensor, "res"],
            knots: Float32[torch.Tensor, "num_knots"],
            compute_derivatives: bool = False
    ) -> tuple[
        Float32[torch.Tensor, "res num_control_points"],
        Union[Float32[torch.Tensor, "res derivative"], None]
    ]:
        """
        Compute the basis matrix with the Cox-de-Boor algorithm.

        Args:
            discretization (Tensor): Parametric coordinates to evaluate.
                Shape: (Resolution,).

            knots (Tensor): Real values with which we can compute Cox-de-Boor algorithm.
                Shape: (Num_Knots,).

        Returns:
            Tensor: Contains the basis values interpolated over the knot vector.
                Shape: (Resolution, Num_Control_points)
        """

        # Expands dimensions to enable broadcasting across the knot intervals.
        discretization = discretization[..., None]

        lower_bound = knots[..., :-1]
        upper_bound = knots[..., 1:]

        # Compute degree 0 matrix with the boundary condition.
        is_in_interval = (discretization >= lower_bound) & (discretization < upper_bound)
        is_at_end = (discretization == knots.max()) & (discretization > lower_bound) & (discretization <= upper_bound)

        basis = (is_in_interval | is_at_end).float()

        # Increase the degree of the matrix by evaluating the previous degree.
        for degree in range(1, self.degree + 1):

            if compute_derivatives and degree == self.degree:
                basis_p_minus_1 = basis

            ti = knots[..., :-(degree + 1)]
            tip = knots[..., degree:-1]
            ti1 = knots[..., 1:-degree]
            tip1 = knots[..., degree + 1:]

            left_weight = self._nurbs_divide(discretization - ti, tip - ti)
            right_weight = self._nurbs_divide(tip1 - discretization, tip1 - ti1)

            basis = left_weight * basis[..., :-1] + right_weight * basis[..., 1:]

        if not compute_derivatives:
            return basis, None

        n = basis.shape[-1]

        delta_left = knots[self.degree:self.degree + n] - knots[:n]
        delta_right = knots[self.degree + 1:self.degree + 1 + n] - knots[1:n + 1]

        d_basis = self.degree * (
                self._nurbs_divide(basis_p_minus_1[..., :n], delta_left)
                - self._nurbs_divide(basis_p_minus_1[..., 1:n + 1], delta_right)
        )
        return basis, d_basis


    @jaxtyped(typechecker=beartype)
    def _nurbs_divide(
            self,
            numerator: Float32[torch.Tensor, "..."],
            denominator: Float32[torch.Tensor, "..."],
    ) -> Float32[torch.Tensor, "..."]:
        """
        Compute the safe element-wise quotient of numerator over denominator.

        Handles the singularity 0/0 by returning 0, ensuring numerical stability
        for homogeneous coordinate division.

        Args:
            numerator (Tensor): Tensor of real values to be divided.
                Shape: (...).

            denominator (Tensor): Tensor of real values to divide the numerator.
                Shape: (...).

        Returns:
            Tensor: Computed quotient, with zeros where the divisor was near-zero.
                Shape: (...).
        """

        is_zero = denominator.abs() <= torch.finfo(denominator.dtype).eps
        safe_denominator = torch.where(is_zero, 1, denominator)
        quotient = numerator / safe_denominator

        return torch.where(is_zero, torch.zeros_like(numerator), quotient)

    @jaxtyped(typechecker=beartype)
    def _compute_resolutions(self, epsilon):

        knots_u = self._get_clamped_knots(self.knot_deltas_u)
        knots_v = self._get_clamped_knots(self.knot_deltas_v)

        eval_points_u, deltas_u, start_u, end_u = self._extract_interval(knots_u)
        eval_points_v, deltas_v, start_v, end_v = self._extract_interval(knots_v)

        basis_u, d2_basis_u = self._compute_second_derivative(eval_points_u, knots_u)
        basis_v, d2_basis_v = self._compute_second_derivative(eval_points_v, knots_v)

        weighted_xyz = self.grid_control_points * self.cyclic_weights_map
        weighted_control_points = torch.cat([weighted_xyz, self.cyclic_weights_map], dim=-1)

        coordinate_4d_u = torch.einsum('iu, jv, uvw -> ijw', d2_basis_u, basis_v, weighted_control_points)
        coordinate_4d_v = torch.einsum('iu, jv, uvw -> ijw', basis_u, d2_basis_v, weighted_control_points)

        kappa_u = coordinate_4d_u[..., :3].norm(dim=-1)
        kappa_u = kappa_u.max(dim=1).values

        kappa_v = coordinate_4d_v[..., :3].norm(dim=-1)
        kappa_v = kappa_v.max(dim=0).values

        resolution_u = self._compute_directional_resolution(kappa_u, deltas_u, start_u, end_u, epsilon)
        resolution_v = self._compute_directional_resolution(kappa_v, deltas_v, start_v, end_v, epsilon)

        return resolution_u, resolution_v

    @jaxtyped(typechecker=beartype)
    def _extract_interval(self, knots):
        unique_values = knots[self.degree :-self.degree]

        t = torch.linspace(0.0, 1.0, self.degree - 1, device=self.device)
        start = unique_values[:-1] + torch.finfo(unique_values.dtype).eps
        end = unique_values[1:] - torch.finfo(unique_values.dtype).eps
        deltas = end - start
        eval_points = start + t * (end - start)

        return eval_points, deltas, start, end

    @jaxtyped(typechecker=beartype)
    def _compute_directional_resolution(
            self,
            d2_values,
            deltas,
            start,
            end,
            epsilon: float
    ) -> Float32[torch.Tensor, "res"]:

        n = torch.maximum(torch.ones_like(d2_values), torch.round(deltas * torch.sqrt(d2_values / (8 * epsilon)))).long()
        resolution = []

        print(n)

        for i in range(n.shape[0]):
            resolution.append(start[i] + torch.linspace(0.0, 1.0, n[i].item(), device=self.device) * (end[i] - start[i]))

        resolution = torch.cat(resolution, dim=0)
        return resolution


    @jaxtyped(typechecker=beartype)
    def _compute_second_derivative(
            self,
            discretization,
            knots
    ):
        discretization = discretization[..., None]

        lower_bound = knots[..., :-1]
        upper_bound = knots[..., 1:]

        # Compute degree 0 matrix with the boundary condition.
        is_in_interval = (discretization >= lower_bound) & (discretization < upper_bound)
        is_at_end = (discretization == knots.max()) & (discretization > lower_bound) & (discretization <= upper_bound)

        basis = (is_in_interval | is_at_end).float()
        d2_basis = basis.clone()

        # Increase the degree of the matrix by evaluating the previous degree.
        for degree in range(1, self.degree + 1):

            if degree == self.degree - 1:
                d2_basis = basis

            ti = knots[..., :-(degree + 1)]
            tip = knots[..., degree:-1]
            ti1 = knots[..., 1:-degree]
            tip1 = knots[..., degree + 1:]

            left_weight = self._nurbs_divide(discretization - ti, tip - ti)
            right_weight = self._nurbs_divide(tip1 - discretization, tip1 - ti1)

            basis = left_weight * basis[..., :-1] + right_weight * basis[..., 1:]

        n = knots.shape[0] - self.degree - 1

        delta_1 = (knots[self.degree:self.degree + n] - knots[:n]) * (knots[self.degree - 1: self.degree - 1 + n] - knots[:n])
        delta_2 = (knots[self.degree:self.degree + n] - knots[:n]) * (knots[self.degree:self.degree + n] - knots[1: n + 1])
        delta_3 = (knots[self.degree + 1: self.degree + 1 + n] - knots[1:n+1]) * (knots[self.degree: self.degree + n] - knots[1:n+1])
        delta_4 = (knots[self.degree + 1: self.degree + 1 + n] - knots[1:n+1]) * (knots[self.degree + 1: self.degree + 1 + n] - knots[2:n+2])

        d2_basis = self.degree * (
                self._nurbs_divide(d2_basis[..., :n], delta_1)
                - self._nurbs_divide(d2_basis[..., 1:n + 1], delta_2)
                - self._nurbs_divide(d2_basis[..., 1:n + 1], delta_3)
                + self._nurbs_divide(d2_basis[..., 2:n + 2], delta_4)
        )

        return basis, d2_basis

    @classmethod
    @jaxtyped(typechecker=beartype)
    def from_args(
            cls,
            control_points: Float32[torch.Tensor, "N 3"],
            index_map: Int64[torch.Tensor, "num_u num_v"],
            weights_map: Float32[torch.Tensor, "num_u num_v"],
            knot_deltas_u: Float32[torch.Tensor, "num_intervals_u"],
            knot_deltas_v: Float32[torch.Tensor, "num_intervals_v"],
            albedo_rgb: Float32[torch.Tensor, "3"],
            roughness: float |  Float32[torch.Tensor, "1"],
            metalness: float | Float32[torch.Tensor, "1"],
            degree: int,
            device: Union[str, torch.device]
    ) -> "NURBS":
        """
        Creates a NURBS instance directly from its constituent tensors.

        Args:
            control_points (Tensor): Unique XYZ coordinates (Rational B-Splines).
                Shape: (N_points, 3).

            index_map (Tensor): Topology map linking unique points to the (U, V) grid.
                Shape: (U, V).

            weights_map (Tensor): Tensor linking unique points to a weight.
                Shape: (U, V).

            knot_deltas_u (Tensor): Interval parameters for the U-axis. Vector used
                to compute internal knot positions.
                Shape: (U - degree + 1,).

            knot_deltas_v (Tensor): Same as `knot_deltas_u` but for the V-axis.
                Shape: (V - degree,).

            albedo_rgb (Tensor): Surface color in RGB [0, 255].
                Shape: (3,).

            degree (int): Polynomial degree of the surface.

            device (str | torch.device): The target device for the tensors.

        Returns:
            Instance of NURBS class.
        """

        return cls(
            control_points.to(device),
            index_map.to(device),
            weights_map.to(device),
            knot_deltas_u.to(device),
            knot_deltas_v.to(device),
            albedo_rgb.to(device),
            roughness.to(device),
            metalness.to(device),
            degree
        )
