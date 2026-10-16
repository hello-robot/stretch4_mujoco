"""Give Stretch 4's head mesh UVs and paint its sensor face.

`head_link.STL` is a single solid shell with no texture coordinates, so the
camera face and the two lidar domes all render in body white. This module
unwraps the mesh and bakes a texture that colors them, so the simulated head
reads like the real one.

It also builds the two LED ring meshes that surround the head cameras. Those
are separate geometry rather than part of the texture, because their colour is
run-time state: see `Stretch4MujocoSimulator.set_led_ring`.

The unwrap is an orthographic projection along the head's forward axis (+X).
A flat projection alone would be ambiguous -- every surface sharing a (y, z)
shares a texel, so a decal meant for the sensor face also lands on the head's
side walls and underside behind it. The fix is a three-row atlas: each face is
assigned a row up front, in 3D, and the rows cannot bleed into each other.

    FACE   the sensor face: forward-facing and near the frontmost point.
           Carries the camera plate, the lens pupils and the centre camera,
           all placed from the sensors' own URDF coordinates.
    DOME   the two lidar caps, picked by distance to the lidar origins.
           Filled flat, so no (y, z) decal can stray onto the shell.
    SHELL  everything else, left body white.
"""

from __future__ import annotations

import numpy as np
import trimesh
from PIL import Image

# The head sensors whose positions the decals are placed from.
SENSOR_LINKS = (
    "camera_left_link",
    "camera_right_link",
    "camera_center_link",
    "lidar_left_link",
    "lidar_right_link",
)

# Sensor positions are read from the URDF; everything here is a size or color
# that the URDF does not carry, measured off the real robot's head.
PIXELS_PER_METRE = 4000

# Body shell, matching the off-white the rest of the robot is drawn in.
SHELL_RGB = (230, 230, 224)
# The glass strip the three head cameras look through.
FACE_PLATE_RGB = (24, 24, 28)
# Camera pupil. The lit ring around it is geometry, not texture.
PUPIL_RGB = (12, 12, 16)
# The lidars' glossy black domes.
DOME_RGB = (30, 30, 34)

# The face plate, measured off a head-on photo of the real robot and given
# relative to where the camera lenses sit ON the shell -- not to the camera
# links, which are ~20 mm behind the plate down a raked axis and so project
# 13 mm too high. Scale comes from the camera spacing the URDF gives. The
# lenses sit high in the plate, just under its top edge, with most of the
# black below them.
FACE_PLATE_ABOVE_LENS = 0.0125
FACE_PLATE_BELOW_LENS = 0.0253
FACE_PLATE_BEYOND_LENS = 0.0155
FACE_PLATE_CORNER_R = 0.012

PUPIL_R = 0.0095
CENTRE_PUPIL_R = 0.0045

# The LED ring around each head camera, measured off the real robot: an
# annulus just outside the pupil, standing a hair proud of the face plate so
# it never z-fights with it.
LED_RING_INNER_R = 0.0100
LED_RING_OUTER_R = 0.0140
LED_RING_THICKNESS = 0.0012
LED_RING_STANDOFF = 0.0006
LED_RING_SEGMENTS = 64
LED_RING_LINKS = ("camera_left_link", "camera_right_link")

# Hello Robot's default head colour, sampled from the real robot's rings.
LED_RING_DEFAULT_RGBA = (0.76, 0.42, 0.95, 1.0)
# The spherical cap's silhouette in this projection reaches 36 mm; past that
# the mesh blends back into the shell, which stays white.
DOME_R = 0.036

# Padding around the mesh bounds so no triangle lands on the atlas border.
MARGIN = 0.004

# A face joins the front row only if it faces forward by at least this much.
# Faces closer to edge-on project to a sliver of the atlas while covering a
# lot of surface, so a decal crossing one smears down the head's side walls;
# sending them to the plain back row keeps the side walls clean.
FRONT_COS = 0.35

# ...and only if it lies within this depth of the frontmost point. A flat
# projection gives one texel to every surface at the same (y, z), and the head
# has forward-facing surfaces behind the sensor face -- the underside lip and
# the inner rear shell -- that would otherwise pick up the face plate. Those
# sit at least 25 mm behind it, so this cleanly keeps the decals on the face.
FRONT_DEPTH = 0.04

# Atlas rows, top to bottom, as `pixel` and `uv` take them.
FACE, DOME, SHELL = 0, 1, 2
ROWS = 3

# The lidar caps are found by flooding outwards from each lidar origin over
# smooth edges, stopping at the crease where the cap meets the shell. A plain
# distance cut leaves the dome's rim ragged, because the seam is not a sphere
# centred on the origin; the crease is exactly where the real one is. The
# search is bounded to DOME_SEARCH_R so a fill can never escape along the
# shell, and the result is insensitive to DOME_CREASE_DEG anywhere in 25-45.
DOME_CREASE_DEG = 35
DOME_SEARCH_R = 0.060


class HeadAtlas:
    """The (y, z) -> pixel map shared by the unwrap and the painter."""

    def __init__(self, bounds):
        (_, y0, z0), (_, y1, z1) = bounds
        self.y0, self.y1 = y0 - MARGIN, y1 + MARGIN
        self.z0, self.z1 = z0 - MARGIN, z1 + MARGIN
        self.width = int(round((self.y1 - self.y0) * PIXELS_PER_METRE))
        self.row_height = int(round((self.z1 - self.z0) * PIXELS_PER_METRE))
        self.height = ROWS * self.row_height

    def row_slice(self, row):
        """The image rows belonging to one atlas row."""
        return slice(row * self.row_height, (row + 1) * self.row_height)

    def pixel(self, y, z, row):
        """Pixel coordinates in the atlas for a point, in the given row.

        `row` is one of FACE / DOME / SHELL. Rows run down the image and +Z is
        up within each. This is the only place the layout is defined; `uv` is
        derived from it, so the unwrap and the painted decals cannot drift
        apart.
        """
        px = (y - self.y0) / (self.y1 - self.y0) * self.width
        t = (self.z1 - z) / (self.z1 - self.z0)
        return px, (t + np.asarray(row)) * self.row_height

    def uv(self, y, z, row):
        """Texture coordinates for a point, OpenGL convention.

        MuJoCo samples textures with v = 0 at the BOTTOM of the image, so v is
        the pixel row measured from the bottom.
        """
        px, py = self.pixel(y, z, row)
        return px / self.width, 1.0 - py / self.height


def _disc(img, atlas, y, z, r, rgb, row, inner=0.0):
    """Paint a filled disc (or annulus) of radius r metres, anti-aliased."""
    cx, cy = atlas.pixel(y, z, row)
    rp = r * PIXELS_PER_METRE
    ip = inner * PIXELS_PER_METRE
    x0, x1 = int(max(0, cx - rp - 2)), int(min(img.shape[1], cx + rp + 2))
    y0, y1 = int(max(0, cy - rp - 2)), int(min(img.shape[0], cy + rp + 2))
    if x0 >= x1 or y0 >= y1:
        return
    xs = np.arange(x0, x1) + 0.5
    ys = np.arange(y0, y1) + 0.5
    d = np.hypot(xs[None, :] - cx, ys[:, None] - cy)
    # One-pixel feather on both edges keeps the decals from stair-stepping.
    # A solid disc gets no inner edge, or the feather would dim its centre.
    a = np.clip(rp - d + 0.5, 0, 1)
    if inner > 0:
        a = a * np.clip(d - ip + 0.5, 0, 1)
    patch = img[y0:y1, x0:x1]
    img[y0:y1, x0:x1] = patch * (1 - a[..., None]) + np.array(rgb) * a[..., None]


def _rounded_rect(img, atlas, y_span, z_span, r, rgb, row):
    (ya, yb), (za, zb) = y_span, z_span
    x0, y1 = atlas.pixel(ya, za, row)
    x1, y0 = atlas.pixel(yb, zb, row)
    rp = r * PIXELS_PER_METRE
    xs = np.arange(img.shape[1]) + 0.5
    ys = np.arange(img.shape[0]) + 0.5
    # Signed distance to the rounded rectangle.
    dx = np.maximum(np.maximum(x0 + rp - xs, xs - (x1 - rp)), 0)
    dy = np.maximum(np.maximum(y0 + rp - ys, ys - (y1 - rp)), 0)
    d = np.hypot(dx[None, :], dy[:, None])
    inside = (xs[None, :] >= x0) & (xs[None, :] <= x1) & (ys[:, None] >= y0) & (ys[:, None] <= y1)
    a = np.clip(rp - d + 0.5, 0, 1) * inside
    img[:] = img * (1 - a[..., None]) + np.array(rgb) * a[..., None]


def write_led_ring_obj(path):
    """Write the LED ring: a thin washer in the local XY plane, facing +Z.

    Closed rather than a bare annulus, so MuJoCo can take its inertia without
    complaint even though the geom is visual-only.
    """
    a = np.linspace(0, 2 * np.pi, LED_RING_SEGMENTS, endpoint=False)
    ring = np.stack([np.cos(a), np.sin(a)], 1)
    half = LED_RING_THICKNESS / 2
    # Four rings of vertices: inner/outer x back/front.
    verts = np.concatenate([
        np.c_[ring * r, np.full(len(ring), z)]
        for r in (LED_RING_INNER_R, LED_RING_OUTER_R)
        for z in (-half, half)
    ])
    n = LED_RING_SEGMENTS
    IB, IF, OB, OF = 0, n, 2 * n, 3 * n      # inner/outer, back/front

    faces = []
    for i in range(n):
        j = (i + 1) % n
        # Front annulus, back annulus, then the outer and inner walls. Winding
        # is chosen so every normal points away from the washer's body.
        faces += [
            (IF + i, OF + i, OF + j), (IF + i, OF + j, IF + j),
            (IB + i, OB + j, OB + i), (IB + i, IB + j, OB + j),
            (OB + i, OB + j, OF + j), (OB + i, OF + j, OF + i),
            (IB + j, IB + i, IF + i), (IB + j, IF + i, IF + j),
        ]
    with open(path, "w") as f:
        f.write("# Stretch 4 head LED ring, generated by head_texture.py\n")
        np.savetxt(f, verts, fmt="v %.6f %.6f %.6f")
        np.savetxt(f, np.asarray(faces) + 1, fmt="f %d %d %d")


def lens_on_shell(mesh, lens):
    """Where a camera's lens meets the face plate, and the plate's normal there.

    The camera links sit well behind the plate along their optical axis, which
    is raked ~39 degrees off +X. Projecting a link straight down the unwrap
    axis would put its decal 13 mm too high, so every face decal and the LED
    rings are anchored here instead.
    """
    centres, normals = mesh.triangles_center, mesh.face_normals
    lens = np.asarray(lens)
    # Plate faces in an annulus around the lens boss: clear of the boss, still
    # on the flat of the plate.
    d = np.linalg.norm(centres[:, 1:] - lens[1:], axis=1)
    on_plate = (
        (d > LED_RING_INNER_R)
        & (d < LED_RING_OUTER_R + 0.006)
        & (normals[:, 0] > FRONT_COS)
        & (centres[:, 0] > mesh.bounds[1][0] - FRONT_DEPTH)
    )
    if not on_plate.any():
        raise ValueError(f"found no face-plate geometry around lens at {lens}")

    normal = normals[on_plate].mean(0)
    normal /= np.linalg.norm(normal)
    # Slide the lens forward along the plate normal onto the plate itself.
    # Taking only the plane from the mesh keeps the result centred on the
    # URDF's lens axis, which an average of the faces would not be: the
    # plate's top edge truncates the annulus and drags the mean downwards.
    plane_point = centres[on_plate].mean(0)
    return lens + normal * np.dot(plane_point - lens, normal), normal


def _quat_from_z_to(normal):
    """MuJoCo w-x-y-z quaternion rotating +Z onto `normal`."""
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z, normal)
    sin = np.linalg.norm(axis)
    if sin < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    angle = np.arctan2(sin, float(np.dot(z, normal)))
    return np.concatenate([[np.cos(angle / 2)], axis / sin * np.sin(angle / 2)])


def led_ring_placements(mesh, sensors):
    """Where each LED ring sits: concentric with its lens, flat on the plate.

    Returns link name -> (pos, quat) in the head frame, quat in MuJoCo's
    w-x-y-z order.
    """
    placements = {}
    for link in LED_RING_LINKS:
        surface, normal = lens_on_shell(mesh, sensors[link])
        pos = surface + normal * (LED_RING_STANDOFF + LED_RING_THICKNESS / 2)
        placements[link] = (pos, _quat_from_z_to(normal))
    return placements


def _cap_faces(mesh, origin):
    """Faces of the lidar cap at `origin`, bounded by the crease around it."""
    centres = mesh.triangles_center
    d = np.linalg.norm(centres - np.asarray(origin), axis=1)

    smooth = np.degrees(mesh.face_adjacency_angles) < DOME_CREASE_DEG
    edges = mesh.face_adjacency[smooth]
    neighbours = [[] for _ in range(len(mesh.faces))]
    for a, b in edges:
        neighbours[a].append(b)
        neighbours[b].append(a)

    # Seed on the cap itself rather than the nearest face: the origin sits
    # inside the dome, so the nearest face could be anything behind it.
    seed = int(np.argmin(np.abs(d - DOME_SEARCH_R / 2)))
    seen = {seed}
    stack = [seed]
    while stack:
        for nxt in neighbours[stack.pop()]:
            if nxt not in seen and d[nxt] < DOME_SEARCH_R:
                seen.add(nxt)
                stack.append(nxt)
    return np.fromiter(seen, int, len(seen))


def paint(atlas, lenses):
    """Bake the head texture.

    `lenses` maps each camera link to where its lens sits on the shell, as
    `lens_on_shell` returns; the plate is positioned relative to those, which
    is how it was measured off the robot.
    """
    img = np.empty((atlas.height, atlas.width, 3), float)
    img[:] = SHELL_RGB

    # The DOME row holds only the lidar caps, so it is simply flat black; its
    # faces were already picked out in 3D.
    img[atlas.row_slice(DOME)] = DOME_RGB

    ys = [lenses[link][1] for link in LED_RING_LINKS]
    zs = [lenses[link][2] for link in LED_RING_LINKS]
    plate_y = (min(ys) - FACE_PLATE_BEYOND_LENS, max(ys) + FACE_PLATE_BEYOND_LENS)
    lens_z = sum(zs) / len(zs)
    plate_z = (lens_z - FACE_PLATE_BELOW_LENS, lens_z + FACE_PLATE_ABOVE_LENS)
    _rounded_rect(img, atlas, plate_y, plate_z, FACE_PLATE_CORNER_R, FACE_PLATE_RGB, FACE)

    for link in LED_RING_LINKS:
        _, y, z = lenses[link]
        _disc(img, atlas, y, z, PUPIL_R, PUPIL_RGB, FACE)

    _, y, z = lenses["camera_center_link"]
    _disc(img, atlas, y, z, CENTRE_PUPIL_R, PUPIL_RGB, FACE)

    return np.clip(img, 0, 255).astype(np.uint8)


def write_obj(path, vertices, faces, face_rows, atlas):
    """Write the mesh with per-face texture coordinates from the atlas."""
    n = len(vertices)
    # One vt block per atlas row, so a face picks its row by offsetting its
    # vertex indices. OBJ indexes vertices and texcoords separately, so this
    # needs no duplicated geometry.
    coords = [np.stack(atlas.uv(vertices[:, 1], vertices[:, 2], row), 1)
              for row in range(ROWS)]
    vt = faces + 1 + (face_rows * n)[:, None]
    with open(path, "w") as f:
        f.write("# Stretch 4 head, unwrapped by head_texture.py\n")
        np.savetxt(f, vertices, fmt="v %.6f %.6f %.6f")
        for block in coords:
            np.savetxt(f, block, fmt="vt %.6f %.6f")
        idx = np.concatenate([faces + 1, vt], 1)[:, [0, 3, 1, 4, 2, 5]]
        np.savetxt(f, idx, fmt="f %d/%d %d/%d %d/%d")


def generate(stl_path, obj_path, png_path, sensors, ring_obj_path=None):
    """Unwrap `stl_path` into `obj_path` and bake its texture to `png_path`.

    With `ring_obj_path`, also writes the LED ring mesh and returns where the
    two rings sit; otherwise returns an empty placement map.

    `sensors` maps the URDF link name of each head sensor to its (x, y, z) in
    the head frame, so the decals follow the URDF rather than hard-coded
    offsets.
    """
    mesh = trimesh.load(stl_path, process=True)
    atlas = HeadAtlas(mesh.bounds)
    centres = mesh.triangles_center

    rows = np.full(len(mesh.faces), SHELL)
    # DOME first: a cap face can also face forward, and the cap wins.
    for link in ("lidar_left_link", "lidar_right_link"):
        rows[_cap_faces(mesh, sensors[link])] = DOME
    face = (
        (rows == SHELL)
        & (mesh.face_normals[:, 0] > FRONT_COS)
        & (centres[:, 0] > mesh.bounds[1][0] - FRONT_DEPTH)
    )
    rows[face] = FACE

    # Every face decal is anchored where its lens meets the shell, not at the
    # camera link, which sits behind the raked plate.
    lenses = {
        link: lens_on_shell(mesh, sensors[link])[0]
        for link in (*LED_RING_LINKS, "camera_center_link")
    }

    write_obj(obj_path, np.asarray(mesh.vertices), np.asarray(mesh.faces), rows, atlas)
    Image.fromarray(paint(atlas, lenses)).save(png_path)

    placements = {}
    if ring_obj_path is not None:
        write_led_ring_obj(ring_obj_path)
        placements = led_ring_placements(mesh, sensors)
    return atlas, placements
