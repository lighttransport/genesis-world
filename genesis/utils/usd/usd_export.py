"""Export a built Genesis scene to a USD stage (the inverse of the USD importer in this package).

``export_scene_to_usd(scene, path)`` writes every ``RigidEntity`` to a Z-up / meters USD stage that can be
re-imported with :func:`genesis.utils.usd.parse_usd_stage` (i.e. ``scene.add_stage(gs.morphs.USD(file=...))``):

* links -> prims with ``UsdPhysics.RigidBodyAPI`` + ``MassAPI`` (kinematic flag for fixed roots),
* joints -> ``UsdPhysics.Revolute/Prismatic/Spherical/FixedJoint`` with axis / limits / drive,
* collision geoms -> ``Collision_*`` prims (analytic ``Cube``/``Sphere``/``Capsule``/``Plane`` where possible,
  otherwise ``UsdGeom.Mesh``) with ``UsdPhysics.CollisionAPI``,
* visual geoms -> ``Visual_*`` ``UsdGeom.Mesh`` prims with a display color.

The canonical stage is authored directly in Genesis' native Z-up / meters / unit-scale frame, so re-import
performs no axis conversion. For an exact-fidelity, device-independent round-trip (collision meshes + SDF +
simulation state) use :func:`genesis.save_compiled_scene` instead; this exporter targets USD interop and a
human-readable / viewable scene description.

Scope (v1): rigid entities only. Materials are exported as display color (best-effort), not full UsdShade.
"""

import os
import re

import numpy as np

import genesis as gs
import genesis.utils.geom as gu


def _require_pxr():
    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics  # noqa: F401

        return Gf, Sdf, Usd, UsdGeom, UsdPhysics
    except ImportError as e:
        gs.raise_exception_from("USD export requires the 'usd-core' package (pip install 'genesis-world[usd]').", e)


_INVALID_PRIM_CHARS = re.compile(r"[^a-zA-Z0-9_]")


def _sanitize(name, fallback="prim"):
    """Turn an arbitrary Genesis name into a valid USD prim name."""
    name = _INVALID_PRIM_CHARS.sub("_", str(name))
    if not name or not (name[0].isalpha() or name[0] == "_"):
        name = "_" + name
    return name or fallback


def _set_local_xform(UsdGeom, Gf, prim, pos, quat, scale=None):
    """Author translate / orient (w,x,y,z) / optional scale xformOps on a prim."""
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
    q = np.asarray(quat, dtype=float)
    xform.AddOrientOp().Set(Gf.Quatf(float(q[0]), Gf.Vec3f(float(q[1]), float(q[2]), float(q[3]))))
    if scale is not None:
        xform.AddScaleOp().Set(Gf.Vec3f(float(scale[0]), float(scale[1]), float(scale[2])))


def _write_mass(UsdPhysics, Gf, prim, link):
    """Author UsdPhysics.MassAPI from a link's inertial properties (eigendecomposed to USD's diag form)."""
    if link.inertial_mass is None or link.inertial_mass <= 0.0:
        return
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(float(link.inertial_mass))
    com = np.asarray(link.inertial_pos if link.inertial_pos is not None else (0.0, 0.0, 0.0), dtype=float)
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(float(com[0]), float(com[1]), float(com[2])))

    if link.inertial_i is not None:
        # Inertia expressed in the link frame: I_link = R(inertial_quat) @ inertial_i @ R(inertial_quat)^T.
        # USD stores a diagonal inertia + principalAxes quaternion, so eigendecompose.
        inertial_i = np.asarray(link.inertial_i, dtype=float)
        quat = np.asarray(link.inertial_quat if link.inertial_quat is not None else (1.0, 0.0, 0.0, 0.0), dtype=float)
        R = gu.quat_to_R(quat)
        I_link = R @ inertial_i @ R.T
        eigvals, eigvecs = np.linalg.eigh(0.5 * (I_link + I_link.T))
        eigvals = np.clip(eigvals, 0.0, None)
        if np.linalg.det(eigvecs) < 0:
            eigvecs[:, 0] = -eigvecs[:, 0]
        paxes = gu.R_to_quat(eigvecs)
        mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*(float(v) for v in eigvals)))
        mass_api.CreatePrincipalAxesAttr(Gf.Quatf(float(paxes[0]), Gf.Vec3f(*(float(v) for v in paxes[1:]))))


def _set_display_color(UsdGeom, Gf, mesh_geom, color):
    if color is None:
        return
    color = np.asarray(color, dtype=float).reshape(-1)[:3]
    mesh_geom.CreateDisplayColorAttr([Gf.Vec3f(*(float(c) for c in color))])


def _author_mesh(UsdGeom, Gf, prim, verts, faces, normals=None, uvs=None):
    mesh = UsdGeom.Mesh(prim)
    mesh.CreatePointsAttr([Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in np.asarray(verts)])
    faces = np.asarray(faces, dtype=np.int64)
    mesh.CreateFaceVertexIndicesAttr([int(i) for i in faces.reshape(-1)])
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    if normals is not None:
        mesh.CreateNormalsAttr([Gf.Vec3f(float(n[0]), float(n[1]), float(n[2])) for n in np.asarray(normals)])
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    return mesh


def _write_collision_geom(modules, stage, link_path, geom, gi):
    Gf, Sdf, Usd, UsdGeom, UsdPhysics = modules
    path = f"{link_path}/Collision_{gi}"
    gtype = geom.type
    data = np.asarray(geom._data, dtype=float)
    scale = None

    if gtype == gs.GEOM_TYPE.BOX:
        prim = UsdGeom.Cube.Define(stage, path)
        prim.CreateSizeAttr(1.0)
        scale = data[:3]  # Cube of side 1 scaled to full extents -> parser recovers geom_data = extents
    elif gtype == gs.GEOM_TYPE.SPHERE:
        prim = UsdGeom.Sphere.Define(stage, path)
        prim.CreateRadiusAttr(float(data[0]))
    elif gtype == gs.GEOM_TYPE.CAPSULE:
        prim = UsdGeom.Capsule.Define(stage, path)
        prim.CreateRadiusAttr(float(data[0]))
        prim.CreateHeightAttr(float(data[1]))
        prim.CreateAxisAttr(UsdGeom.Tokens.z)
    elif gtype == gs.GEOM_TYPE.PLANE:
        prim = UsdGeom.Plane.Define(stage, path)
        prim.CreateAxisAttr(UsdGeom.Tokens.z)
        prim.CreateWidthAttr(100.0)
        prim.CreateLengthAttr(100.0)
    else:  # MESH / CYLINDER / ELLIPSOID / TERRAIN -> triangle mesh
        prim_obj = UsdGeom.Mesh.Define(stage, path)
        _author_mesh(UsdGeom, Gf, prim_obj.GetPrim(), geom.init_verts, geom.init_faces, geom.init_normals)
        prim = prim_obj

    p = prim.GetPrim()
    _set_local_xform(UsdGeom, Gf, p, geom.init_pos, geom.init_quat, scale=scale)
    UsdPhysics.CollisionAPI.Apply(p)
    return p


def _write_visual_geom(modules, stage, link_path, vgeom, vi):
    Gf, Sdf, Usd, UsdGeom, UsdPhysics = modules
    path = f"{link_path}/Visual_{vi}"
    mesh = UsdGeom.Mesh.Define(stage, path)
    _author_mesh(UsdGeom, Gf, mesh.GetPrim(), vgeom._init_vverts, vgeom._init_vfaces, vgeom._init_vnormals)
    _set_display_color(UsdGeom, Gf, mesh, getattr(vgeom, "_color", None))
    _set_local_xform(UsdGeom, Gf, mesh.GetPrim(), vgeom._init_pos, vgeom._init_quat)
    # Visual-only: keep out of collision so the importer classifies it as visual geometry.
    UsdGeom.Imageable(mesh.GetPrim()).CreatePurposeAttr(UsdGeom.Tokens.default_)
    return mesh.GetPrim()


def _write_joint(modules, stage, entity_path, joint, parent_link_path, child_link_path):
    """Author a UsdPhysics joint connecting parent_link_path (body0) -> child_link_path (body1)."""
    Gf, Sdf, Usd, UsdGeom, UsdPhysics = modules
    jtype = joint.type
    if jtype == gs.JOINT_TYPE.FREE:
        return None  # free root: represented by a non-kinematic RigidBody and no joint

    jpath = f"{entity_path}/{_sanitize(joint.name)}"
    pos = np.asarray(joint.pos, dtype=float)

    if jtype == gs.JOINT_TYPE.REVOLUTE:
        joint_prim = UsdPhysics.RevoluteJoint.Define(stage, jpath)
        axis_vec = np.asarray(joint.dofs_motion_ang[0], dtype=float)
    elif jtype == gs.JOINT_TYPE.PRISMATIC:
        joint_prim = UsdPhysics.PrismaticJoint.Define(stage, jpath)
        axis_vec = np.asarray(joint.dofs_motion_vel[0], dtype=float)
    elif jtype == gs.JOINT_TYPE.SPHERICAL:
        joint_prim = UsdPhysics.SphericalJoint.Define(stage, jpath)
        axis_vec = None
    else:  # FIXED
        joint_prim = UsdPhysics.FixedJoint.Define(stage, jpath)
        axis_vec = None

    if parent_link_path is not None:
        joint_prim.CreateBody0Rel().SetTargets([Sdf.Path(parent_link_path)])
    joint_prim.CreateBody1Rel().SetTargets([Sdf.Path(child_link_path)])
    # Joint frame on the child: position in the child link frame, identity rotation (axis carried explicitly).
    joint_prim.CreateLocalPos1Attr(Gf.Vec3f(float(pos[0]), float(pos[1]), float(pos[2])))
    joint_prim.CreateLocalRot1Attr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))

    if axis_vec is not None and np.linalg.norm(axis_vec) > gs.EPS:
        axis_vec = axis_vec / np.linalg.norm(axis_vec)
        # Map the (already normalized) Genesis motion axis to the nearest canonical USD axis and bake the
        # residual rotation into localRot1, so the importer reconstructs the exact axis direction.
        canonical = np.eye(3)
        ax_idx = int(np.argmax(np.abs(axis_vec)))
        ref = canonical[ax_idx] * np.sign(axis_vec[ax_idx] or 1.0)
        joint_prim.CreateAxisAttr(("X", "Y", "Z")[ax_idx])
        rot_quat = gu.R_to_quat(_rotation_between(ref, axis_vec))
        joint_prim.CreateLocalRot1Attr(Gf.Quatf(float(rot_quat[0]), Gf.Vec3f(*(float(v) for v in rot_quat[1:]))))

        limit = np.asarray(joint.dofs_limit[0], dtype=float)
        lo, hi = float(limit[0]), float(limit[1])
        if np.isfinite(lo) and np.isfinite(hi):
            if jtype == gs.JOINT_TYPE.REVOLUTE:
                lo, hi = np.rad2deg(lo), np.rad2deg(hi)
            joint_prim.CreateLowerLimitAttr(lo)
            joint_prim.CreateUpperLimitAttr(hi)

        # Passive joint stiffness/damping (read by the importer from `physxLimit:<axis>:{stiffness,damping}`),
        # distinct from the active DriveAPI below.
        kind = "angular" if jtype == gs.JOINT_TYPE.REVOLUTE else "linear"
        _author_passive_stiffness_damping(
            joint_prim.GetPrim(), kind, float(joint.dofs_stiffness[0]), float(joint.dofs_damping[0])
        )

        # Active drive (stiffness/damping/effort) and joint dynamics (armature/friction).
        kp = float(joint.dofs_act_gain[0])
        kv = float(-joint.dofs_act_bias[0][2]) if joint.dofs_act_bias[0][2] else 0.0
        fr = np.asarray(joint.dofs_force_range[0], dtype=float)
        if kp or kv or np.isfinite(fr[1]):
            drive = UsdPhysics.DriveAPI.Apply(joint_prim.GetPrim(), kind)
            drive.CreateStiffnessAttr(kp)
            drive.CreateDampingAttr(kv)
            if np.isfinite(fr[1]):
                drive.CreateMaxForceAttr(float(fr[1]))
        _author_joint_dynamics(joint_prim.GetPrim(), joint)

    return joint_prim.GetPrim()


def _author_passive_stiffness_damping(prim, kind, stiffness, damping):
    """Author passive joint stiffness/damping under `physxLimit:<angular|linear>:{stiffness,damping}`."""
    if stiffness:
        prim.CreateAttribute(f"physxLimit:{kind}:stiffness", _float_sdf_type()).Set(stiffness)
    if damping:
        prim.CreateAttribute(f"physxLimit:{kind}:damping", _float_sdf_type()).Set(damping)


def _author_joint_dynamics(prim, joint):
    """Author armature / friction under the attribute names the importer looks for first."""
    armature = float(np.asarray(joint.dofs_armature, dtype=float).reshape(-1)[0]) if joint.n_dofs else 0.0
    friction = float(np.asarray(joint.dofs_frictionloss, dtype=float).reshape(-1)[0]) if joint.n_dofs else 0.0
    if armature:
        attr = prim.CreateAttribute("physxJoint:armature", _float_sdf_type())
        attr.Set(armature)
    if friction:
        attr = prim.CreateAttribute("physxJoint:jointFriction", _float_sdf_type())
        attr.Set(friction)


def _float_sdf_type():
    from pxr import Sdf

    return Sdf.ValueTypeNames.Float


def _rotation_between(a, b):
    """Minimal rotation matrix taking unit vector a to unit vector b."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else gu.quat_to_R(_orthogonal_flip_quat(a))
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def _orthogonal_flip_quat(axis):
    """180-degree rotation about any axis orthogonal to `axis` (for the antiparallel case)."""
    ortho = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    ortho = ortho - np.dot(ortho, axis) * np.asarray(axis, dtype=float)
    ortho /= np.linalg.norm(ortho)
    return np.array([0.0, *ortho])


def _export_entity(modules, stage, entity, parent_path):
    Gf, Sdf, Usd, UsdGeom, UsdPhysics = modules
    epath = f"{parent_path}/{_sanitize(entity.name)}"
    UsdGeom.Xform.Define(stage, epath)

    links = list(entity.links)
    lstart = entity.link_start
    world_T = {}
    link_prim_path = {}

    for li, link in enumerate(links):
        prel = int(link.parent_idx) - lstart if link.parent_idx is not None and link.parent_idx >= 0 else -1
        lpos, lquat = np.asarray(link.pos, dtype=float), np.asarray(link.quat, dtype=float)
        if prel < 0:
            wpos, wquat = lpos, lquat
        else:
            ppos, pquat = world_T[prel]
            wpos, wquat = gu.transform_pos_quat_by_trans_quat(lpos, lquat, ppos, pquat)
        world_T[li] = (wpos, wquat)

        lpath = f"{epath}/{_sanitize(link.name)}_{li}"
        link_prim_path[li] = lpath
        prim = UsdGeom.Xform.Define(stage, lpath).GetPrim()
        _set_local_xform(UsdGeom, Gf, prim, wpos, wquat)

        # A root link (no parent) is fixed-to-world unless it carries a FREE joint: a fixed base has either
        # no joints or only FIXED joints. Mark such roots kinematic so the importer treats them as fixed.
        joints = list(link.joints)
        is_fixed_root = prel < 0 and not any(j.type == gs.JOINT_TYPE.FREE for j in joints)
        rb = UsdPhysics.RigidBodyAPI.Apply(prim)
        if is_fixed_root:
            rb.CreateKinematicEnabledAttr(True)
        _write_mass(UsdPhysics, Gf, prim, link)

        for gi, geom in enumerate(link.geoms):
            _write_collision_geom(modules, stage, lpath, geom, gi)
        for vi, vgeom in enumerate(link.vgeoms):
            _write_visual_geom(modules, stage, lpath, vgeom, vi)

    # Joints (after all link prims exist so body relationships resolve).
    for li, link in enumerate(links):
        prel = int(link.parent_idx) - lstart if link.parent_idx is not None and link.parent_idx >= 0 else -1
        parent_path_for_joint = link_prim_path[prel] if prel >= 0 else None
        for joint in link.joints:
            if joint.type == gs.JOINT_TYPE.FIXED and prel < 0:
                continue  # fixed base handled via kinematic flag
            _write_joint(modules, stage, epath, joint, parent_path_for_joint, link_prim_path[li])

    return epath


def export_scene_to_usd(scene, path, *, overwrite=True):
    """Export a built scene's rigid entities to a USD stage at ``path``.

    Parameters
    ----------
    scene : genesis.Scene
        A built scene (``scene.build()`` already called).
    path : str | os.PathLike
        Output USD file (``.usd``, ``.usda`` or ``.usdc``).
    overwrite : bool
        Overwrite an existing file at ``path`` (default True).

    Returns
    -------
    str
        The written USD path.
    """
    from genesis.engine.entities.rigid_entity.rigid_entity import RigidEntity

    modules = _require_pxr()
    Gf, Sdf, Usd, UsdGeom, UsdPhysics = modules

    if not scene.is_built:
        gs.raise_exception("export_scene_to_usd requires a built scene (call `scene.build()` first).")

    path = os.fspath(path)
    if os.path.exists(path):
        if not overwrite:
            gs.raise_exception(f"USD file already exists: {path} (pass overwrite=True to replace).")
        os.remove(path)

    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    n = 0
    for entity in scene.entities:
        if not isinstance(entity, RigidEntity):
            gs.logger.warning(
                f"export_scene_to_usd: skipping non-rigid entity '{getattr(entity, 'name', entity)}' "
                f"({type(entity).__name__})."
            )
            continue
        if getattr(entity, "_enable_heterogeneous", False):
            gs.logger.warning(f"export_scene_to_usd: skipping heterogeneous entity '{entity.name}'.")
            continue
        _export_entity(modules, stage, entity, "/World")
        n += 1

    stage.GetRootLayer().Save()
    gs.logger.info(f"Exported {n} rigid entit(ies) to USD ~~<{path}>~~.")
    return path
