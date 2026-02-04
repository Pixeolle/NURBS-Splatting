
import torch
import torch.nn as nn

from jaxtyping import jaxtyped, Float32, Int64
from beartype import beartype

class NURBS(nn.Module):
    """
    A differentiable NURBS surface layer with cyclic topology support.

    The U-dimension is treated as the primary axis (longitudinal),
    while the V-dimension represents the auxiliary layers (transversal).

    The knot vector is reconstructed using a 'Softmax Sandwich' strategy:
    internal intervals are learned as raw deltas, normalized via softmax
    to ensure monotonicity, and then clamped between (p+1) zeros and ones.

    Attributes:
        control_points: Unique XYZ coordinates (Rational B-Splines).
        index_map: Topology map linking unique points to the (U, V) grid.
        weights_map: Tensor linking unique points to a weight.
        knot_deltas_u: Interval parameters for the U-axis. Vector of size (num_u - p) used to compute internal knot positions.
        knot_deltas_v: Same as `knot_deltas_u` but for the V-axis. Vector of size (num_v - p).
        albedo_rgb: Surface color in RGB [0, 255].
        degree: Polynomial degree of the surface.
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
            degree: int = 2,
    ) -> None:
        super().__init__()

        self._validate_integrity(control_points, index_map, weights_map, knot_deltas_u, knot_deltas_v, albedo_rgb, degree)

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
            control_points: Unique rational coordinates (XYZW).
            index_map: Grid mapping used to reconstruct the control net.
            knot_deltas_u: Internal knot intervals for the U axis.
            knot_deltas_v: Internal knot intervals for the V axis.
            albedo_rgb: Surface color vector in 8-bit RGB.
            degree: Polynomial degree of the basis functions.

        Raises:
            ValueError: If any mathematical constraint (like m = n + p + 1) is violated.
            IndexError: If the index_map points to non-existent control points.
        """

        if degree < 1:
            raise ValueError(f"Degree must be at least 1, got {degree}.")

        if len(control_points.shape) != 2:
            raise ValueError(f"Control points should be a 2-rank tensor (N, 3), got {control_points.shape}")

        if control_points.shape[-1] != 3:
            raise ValueError(f"Control points should be encoded with 3 channels (XYZ), got {control_points.shape[-1]}.")

        if torch.any(index_map >= control_points.shape[0]):
            raise IndexError(f"Index map should not contain index above or equal to the number of control points ({control_points.shape[0]}), got {index_map.max().item()}.")

        expected_u = index_map.shape[0] - degree + 1
        if knot_deltas_u.shape[0] != expected_u:
            raise ValueError(f"U knot intervals mismatch. Expected {expected_u} (num_u - p), got {knot_deltas_u.shape[-1]}")

        expected_v = index_map.shape[1] - degree
        if knot_deltas_v.shape[0] != expected_v:
            raise ValueError(f"V knot intervals mismatch. Expected {expected_v} (num_v - p), got {knot_deltas_v.shape[-1]}")

        if weights_map.shape != index_map.shape:
            raise ValueError(f"Index map and weights map should have the same shape ({weights_map.shape} != {index_map.shape})")

        if albedo_rgb.shape != (3,):
            raise ValueError("Albedo should be encoded with 3 channels (RGB).")

        if torch.any(albedo_rgb > 255) or torch.any(albedo_rgb < 0):
            raise ValueError("Albedo should be in RGB in 8 bits that implies values between 0 and 255 (included).")


    @jaxtyped(typechecker=beartype)
    def forward(
            self,
            discretization_u: Float32[torch.Tensor, "res_u"],
            discretization_v: Float32[torch.Tensor, "res_v"]
    ) -> tuple[Float32[torch.Tensor, "num_points 3"], Int64[torch.Tensor, "triangles 3"]]:
        """
        Tessellate the NURBS according to the discretization given.

        Args:
            discretization_u: Tensor of values between 0.0 and 1.0 included that represents the resolution of the tessellation in the U-dimension (longitudinal).
            discretization_v: Same as `discretization_u` but for the V-dimension (transversal).

        Returns:
            Tuple with a tensor that contains all the point in format (XYZ) and another tensor that contain the three indices to build the triangles.
        """

        # Put input tensor on the right device to avoid error during tessellate.
        discretization_u = discretization_u.to(self.device)
        discretization_v = discretization_v.to(self.device)

        # Clamp input tensor to avoid error during tessellate.
        discretization_u = torch.clamp(discretization_u, 0.0, 1.0)
        discretization_v = torch.clamp(discretization_v, 0.0, 1.0)

        # Reconstruct knots vector from deltas.
        knots_u = self._get_clamped_knots(self.knot_deltas_u)
        knots_v = self._get_clamped_knots(self.knot_deltas_v)

        # Compute basis matrices.
        basis_u = self._compute_basis_functions(discretization_u, knots_u)
        basis_v = self._compute_basis_functions(discretization_v, knots_v)

        # Reconstruct the controls points matrix and apply weights to the coordinates of controls points.
        print(self.control_points.shape)
        weighted_xyz = self.control_points * self.cyclic_weights_map

        weighted_control_points = torch.cat([weighted_xyz, self.cyclic_weights_map], dim=-1)

        # Compute the tensor product along U and V and then divide by the weight previously multiply by the basis matrices to obtain the surface describes by the NURBS.
        coordinate_4d = torch.einsum('iu, jv, uvw -> ijw', basis_u, basis_v, weighted_control_points)
        coordinate_3d = self._nurbs_divide(coordinate_4d[..., :3], coordinate_4d[..., 3:4])

        # Flatten the coordinate to correspond to the rasterization format.
        coordinate_3d = coordinate_3d.reshape(-1, 3)

        # Compute the triangles indices
        triangles = self._compute_triangles_indices(discretization_u.shape[0], discretization_v.shape[0])

        return coordinate_3d, triangles


    @property
    @jaxtyped(typechecker=beartype)
    def device(self) -> torch.device:
        return self.control_points.device


    @property
    @jaxtyped(typechecker=beartype)
    def grid_control_points(self) -> Float32[torch.Tensor, "U V 4"]:
        """Dynamically reconstructs the control point grid from the 1D parameter tensor."""
        return self.control_points[self.cyclic_index_map]

    @property
    @jaxtyped(typechecker=beartype)
    def cyclic_index_map(self) -> Int64[torch.Tensor, "num_u_cyclic num_v"]:
        """Return the cyclic index map by copying the first row at the end."""
        return torch.cat([self.index_map, self.index_map[None, 0, :]], dim=0)

    @property
    @jaxtyped(typechecker=beartype)
    def cyclic_weights_map(self) -> Float32[torch.Tensor, "num_u_cyclic num_v"]:
        """Return the cyclic weights_map by copying the first row at the end."""
        return torch.cat([self.weights_map, self.weights_map[None, 0, :]], dim=0)

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
            deltas: Raw interval parameters. Will be normalized via softmax internally to ensure monotonicity and range constraints.

        Return:
            A full knot vector of shape (num_control_points + degree + 1), starting with zeros and ending with ones.
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
    ) -> Int64[torch.Tensor, "triangles 3"]:
        """
        Compute indices to render the surface.

        Args:
            resolution_u: Resolution along the U-dimension.
            resolution_v: Resolution along the V-dimension.

        Returns:
            Tensor of 3 indices that indicate triangles to render.
        """

        # Create tensor indices in each dimension
        u = torch.arange(resolution_u - 1, device=self.device)
        v = torch.arange(resolution_v - 1, device=self.device)

        # Create grid that link u and v
        uu, vv = torch.meshgrid(u, v, indexing='ij')

        # Compute indices for each corner of the square
        bottom_left =  uu * resolution_v + vv
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
            knots: Float32[torch.Tensor, "num_knots"]
    ) -> Float32[torch.Tensor, "res num_control_points"]:
        """
        Compute the basis matrix with the Cox-de-Boor algorithm.

        Args:
            discretization: Tensor of real values where we evaluate the basis values.
            knots: Tensor of real values with which we can compute Cox-de-Boor algorithm.

        Returns:
            2D Matrix that contains the basis values interpolated over the knot vector.
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
            ti = knots[..., :-(degree + 1)]
            tip = knots[..., degree:-1]
            ti1 = knots[..., 1:-degree]
            tip1 = knots[..., degree + 1:]

            left_weight = self._nurbs_divide(discretization - ti, tip - ti)
            right_weight = self._nurbs_divide(tip1 - discretization, tip1 - ti1)

            basis = left_weight * basis[..., :-1] + right_weight * basis[..., 1:]

        return basis


    @jaxtyped(typechecker=beartype)
    def _nurbs_divide(
            self,
            numerator: Float32[torch.Tensor, "... column"],
            denominator: Float32[torch.Tensor, "... column"],
    ) -> Float32[torch.Tensor, "... row column"]:
        """
        Compute the safe element-wise quotient of numerator over denominator.

        Args:
            numerator: Tensor of real values to be divided.
            denominator: Tensor of real values to divide the numerator.

        Returns:
            The computed quotient, with zeros where the divisor was near-zero.
        """

        is_zero = denominator.abs() <= torch.finfo(denominator.dtype).eps
        safe_denominator = torch.where(is_zero, 1, denominator)
        quotient = numerator / safe_denominator

        return torch.where(is_zero, torch.zeros_like(numerator), quotient)


    @classmethod
    @jaxtyped(typechecker=beartype)
    def from_components(
            cls,
            control_points: Float32[torch.Tensor, "N 3"],
            index_map: Int64[torch.Tensor, "num_u num_v"],
            weights_map: Float32[torch.Tensor, "num_u num_v"],
            knot_deltas_u: Float32[torch.Tensor, "num_intervals_u"],
            knot_deltas_v: Float32[torch.Tensor, "num_intervals_v"],
            albedo_rgb: Float32[torch.Tensor, "3"],
            degree: int,
    ) -> "NURBS":
        """
        Creates a NURBS instance directly from its constituent tensors.

        Args:
            control_points: Unique XYZW coordinates (Rational B-Splines).
            index_map: Topology map linking unique points to the (U, V) grid.
            weights_map: Tensor linking unique points to a weight.
            knot_deltas_u: Interval parameters for the U-axis. Vector of size (num_u - p) used to compute internal knot positions.
            knot_deltas_v: Same as `knot_deltas_u` but for the V-axis. Vector of size (num_v - p).
            albedo_rgb: Surface color in RGB [0, 255].
            degree: Polynomial degree of the surface.
            device: Device where the model is stored.

        Returns:
            Instance of NURBS class.

        """
        return cls(
            control_points,
            index_map,
            weights_map,
            knot_deltas_u,
            knot_deltas_v,
            albedo_rgb,
            degree
        )

    @classmethod
    @jaxtyped(typechecker=beartype)
    def from_random(
            cls,
            device: torch.device
    ) -> "NURBS":
        """
        Creates a cyclic NURBS with random components and a resolution of 4x4 by get two random points and interpolate them .
        Args:
            device: Device where the model is stored.

        Returns:
            Instance of NURBS class
        """

        # Get the two opposites points, for nomenclature convention we consider the first point as [-1, -1, -1] and the second [1, 1, 1]
        first_point = torch.rand(3)[..., None]
        second_point = torch.rand(3)[..., None]

        center = torch.mean(first_point + second_point, dim=0)

        center_bottom = center
        center_bottom[2] = first_point[2]

        center_top = center
        center_top[2] = second_point[2]

        coordinates_available = torch.cat([first_point, second_point], dim=-1)
        x, y, z = torch.meshgrid(coordinates_available[0], coordinates_available[1], coordinates_available[2], indexing='ij')

        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        z = z.reshape(-1, 1)

        points = torch.cat([x, y, z], dim=-1)
        points = torch.cat([points, center_top, center_bottom], dim=0)






