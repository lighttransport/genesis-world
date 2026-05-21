"""Compiled-scene fast-path: save a fully-built rigid scene to a portable bundle and reload it without
re-parsing assets, re-running collision post-processing (convex decomposition / decimation / merging) or
recomputing SDFs.

The expensive, currently-uncached work in ``scene.build()`` is asset parsing + collision post-processing +
SDF computation. This module captures the *result* of that work straight from a built scene and serializes
it into a portable, device-independent bundle:

    <path>/                     # a directory (conventionally suffixed ".gscene")
      manifest.json             # structure: entities/links/joints/geoms, scalars, material params, flags
      arrays.npz                # all numeric arrays (verts/faces/normals/uvs, poses, inertia, joint dofs, SDF)

On load, entities are rebuilt through the normal ``RigidEntity`` machinery but with a :class:`genesis.morphs._Compiled`
carrier morph that short-circuits ``_parse_scene`` (returns the stored info dicts), skips ``_postprocess_geoms_info``
and ``_align_link`` (the geometry is already final and aligned), and the per-geom SDF arrays are written into the
``.gsd`` cache so ``RigidGeom._preprocess`` loads them instead of recomputing.

Scope (v1): rigid entities only (``RigidEntity``). Non-rigid entities and equality constraints are not captured;
:func:`save_compiled_scene` warns and skips them.
"""

import json
import os

import numpy as np

import genesis as gs
import genesis.utils.mesh as mu
from genesis.utils.misc import get_gsd_cache_dir

BUNDLE_FORMAT_VERSION = 1
_MANIFEST_NAME = "manifest.json"
_ARRAYS_NAME = "arrays.npz"

# Material constructor fields restored verbatim (everything that affects dynamics / SDF grid shape / coupling).
_MATERIAL_FIELDS = (
    "rho",
    "friction",
    "needs_coup",
    "coup_friction",
    "coup_softness",
    "coup_restitution",
    "sdf_cell_size",
    "sdf_min_res",
    "sdf_max_res",
    "gravity_compensation",
    "coup_type",
    "coup_links",
)

# Geom metadata flags that affect downstream behavior (notably `Mesh.is_convex` reads "convexified").
_GEOM_META_KEYS = ("convexified", "decomposed", "merged", "name", "mesh_path")


def _akey(*parts):
    """Build a flat npz array key from index/field parts, e.g. ('e0', 'l1', 'g0', 'verts')."""
    return ".".join(str(p) for p in parts)


# --------------------------------------------------------------------------------------------------
# Save
# --------------------------------------------------------------------------------------------------


def _capture_joint(joint):
    """Capture a RigidJoint into (meta, arrays) where arrays are stored separately in the npz."""
    meta = {
        "name": joint.name,
        "type": int(joint.type),
        "n_qs": int(joint.n_qs),
        "n_dofs": int(joint.n_dofs),
    }
    arrays = {
        "pos": np.asarray(joint.pos, dtype=np.float64),
        "quat": np.asarray(joint.quat, dtype=np.float64),
        "init_qpos": np.asarray(joint.init_qpos, dtype=np.float64),
        "sol_params": np.asarray(joint.sol_params, dtype=np.float64),
        "dofs_motion_ang": np.asarray(joint.dofs_motion_ang, dtype=np.float64),
        "dofs_motion_vel": np.asarray(joint.dofs_motion_vel, dtype=np.float64),
        "dofs_limit": np.asarray(joint.dofs_limit, dtype=np.float64),
        "dofs_invweight": np.asarray(joint.dofs_invweight, dtype=np.float64),
        "dofs_frictionloss": np.asarray(joint.dofs_frictionloss, dtype=np.float64),
        "dofs_stiffness": np.asarray(joint.dofs_stiffness, dtype=np.float64),
        "dofs_damping": np.asarray(joint.dofs_damping, dtype=np.float64),
        "dofs_armature": np.asarray(joint.dofs_armature, dtype=np.float64),
        "dofs_act_gain": np.asarray(joint.dofs_act_gain, dtype=np.float64),
        "dofs_act_bias": np.asarray(joint.dofs_act_bias, dtype=np.float64),
        "dofs_force_range": np.asarray(joint.dofs_force_range, dtype=np.float64),
    }
    return meta, arrays


def _capture_geom(geom):
    """Capture a collision RigidGeom into (meta, arrays), including its SDF grid."""
    meta = {
        "type": int(geom.type),
        "friction": None if geom.friction is None else float(geom.friction),
        "contype": int(geom.contype),
        "conaffinity": int(geom.conaffinity),
        "metadata": {k: geom.metadata[k] for k in _GEOM_META_KEYS if k in geom.metadata},
        "has_uvs": geom.uvs is not None,
    }
    # `convexified` drives `Mesh.is_convex`; persist the actual value even if the source omitted the flag.
    meta["metadata"]["convexified"] = bool(geom.is_convex)
    arrays = {
        "verts": np.ascontiguousarray(geom.init_verts, dtype=np.float64),
        "faces": np.ascontiguousarray(geom.init_faces, dtype=np.int32),
        "normals": np.ascontiguousarray(geom.init_normals, dtype=np.float64),
        "init_pos": np.asarray(geom.init_pos, dtype=np.float64),
        "init_quat": np.asarray(geom.init_quat, dtype=np.float64),
        "data": np.asarray(geom._data, dtype=np.float64),
        "sol_params": np.asarray(geom.sol_params, dtype=np.float64),
        # SDF grid (restored into the `.gsd` cache on load). Accessing these triggers `_preprocess` if needed.
        "sdf_val": np.ascontiguousarray(geom.sdf_val),
        "sdf_grad": np.ascontiguousarray(geom.sdf_grad),
        "sdf_max": np.asarray(geom.sdf_max),
        "sdf_closest_vert": np.ascontiguousarray(geom.sdf_closest_vert),
        "T_mesh_to_sdf": np.ascontiguousarray(geom.T_mesh_to_sdf),
    }
    if geom.uvs is not None:
        arrays["uvs"] = np.ascontiguousarray(geom.uvs, dtype=np.float64)
    return meta, arrays


def _capture_vgeom(vgeom):
    """Capture a visual RigidVisGeom into (meta, arrays)."""
    meta = {
        "name": vgeom.metadata.get("name", ""),
        "has_uvs": vgeom.uvs is not None,
        "has_normals": vgeom._init_vnormals is not None,
    }
    arrays = {
        "vverts": np.ascontiguousarray(vgeom._init_vverts, dtype=np.float64),
        "vfaces": np.ascontiguousarray(vgeom._init_vfaces, dtype=np.int32),
        "init_pos": np.asarray(vgeom._init_pos, dtype=np.float64),
        "init_quat": np.asarray(vgeom._init_quat, dtype=np.float64),
        "color": np.asarray(vgeom._color, dtype=np.float64),
    }
    if vgeom._init_vnormals is not None:
        arrays["vnormals"] = np.ascontiguousarray(vgeom._init_vnormals, dtype=np.float64)
    if vgeom.uvs is not None:
        arrays["uvs"] = np.ascontiguousarray(vgeom.uvs, dtype=np.float64)
    return meta, arrays


def _capture_material(material):
    out = {}
    for field in _MATERIAL_FIELDS:
        if hasattr(material, field):
            value = getattr(material, field)
            out[field] = list(value) if isinstance(value, (tuple, list)) else value
    return out


def _capture_entity(entity, ei, arrays):
    """Capture a built RigidEntity into a manifest dict, accumulating its arrays into `arrays`."""
    link_start = entity.link_start
    links_meta = []
    for li, link in enumerate(entity.links):
        # Joints
        joints_meta = []
        for ji, joint in enumerate(link.joints):
            jmeta, jarr = _capture_joint(joint)
            for name, arr in jarr.items():
                arrays[_akey(f"e{ei}", f"l{li}", f"j{ji}", name)] = arr
            joints_meta.append(jmeta)

        # Collision geoms
        geoms_meta = []
        for gi, geom in enumerate(link.geoms):
            gmeta, garr = _capture_geom(geom)
            for name, arr in garr.items():
                arrays[_akey(f"e{ei}", f"l{li}", f"g{gi}", name)] = arr
            geoms_meta.append(gmeta)

        # Visual geoms
        vgeoms_meta = []
        for vi, vgeom in enumerate(link.vgeoms):
            vmeta, varr = _capture_vgeom(vgeom)
            for name, arr in varr.items():
                arrays[_akey(f"e{ei}", f"l{li}", f"v{vi}", name)] = arr
            vgeoms_meta.append(vmeta)

        # Link frame + inertial (post-alignment)
        parent_idx = int(link.parent_idx) - link_start if link.parent_idx >= 0 else -1
        root_idx = None if link.root_idx is None else (int(link.root_idx) - link_start if link.root_idx >= 0 else -1)
        arrays[_akey(f"e{ei}", f"l{li}", "pos")] = np.asarray(link.pos, dtype=np.float64)
        arrays[_akey(f"e{ei}", f"l{li}", "quat")] = np.asarray(link.quat, dtype=np.float64)

        lmeta = {
            "name": link.name,
            "parent_idx": parent_idx,
            "root_idx": root_idx,
            "is_robot": bool(link._is_robot),
            "joints": joints_meta,
            "geoms": geoms_meta,
            "vgeoms": vgeoms_meta,
            "inertial_mass": None if link.inertial_mass is None else float(link.inertial_mass),
        }
        for name, value in (
            ("inertial_pos", link.inertial_pos),
            ("inertial_quat", link.inertial_quat),
            ("inertial_i", link.inertial_i),
            ("invweight", link.invweight),
        ):
            if value is not None:
                arrays[_akey(f"e{ei}", f"l{li}", name)] = np.asarray(value, dtype=np.float64)
                lmeta[f"has_{name}"] = True
            else:
                lmeta[f"has_{name}"] = False
        links_meta.append(lmeta)

    return {
        "name": entity.name,
        "requires_jac_and_IK": bool(entity._requires_jac_and_IK),
        "is_local_collision_mask": bool(getattr(entity, "_is_local_collision_mask", False)),
        "material": _capture_material(entity.material),
        "links": links_meta,
    }


def save_compiled_scene(scene, path):
    """Serialize a built scene's rigid entities to a portable compiled-scene bundle at ``path``.

    Parameters
    ----------
    scene : genesis.Scene
        A scene that has already been built (``scene.build()`` called).
    path : str | os.PathLike
        Destination directory for the bundle (conventionally suffixed ``.gscene``). Created if missing.

    Notes
    -----
    Only ``RigidEntity`` instances are captured. Other entity types and equality constraints are skipped
    with a warning.
    """
    from genesis.engine.entities.rigid_entity.rigid_entity import RigidEntity

    if not scene.is_built:
        gs.raise_exception("save_compiled_scene requires a built scene (call `scene.build()` first).")

    path = os.fspath(path)
    os.makedirs(path, exist_ok=True)

    arrays = {}
    entities_meta = []
    for entity in scene.entities:
        if not isinstance(entity, RigidEntity):
            gs.logger.warning(
                f"save_compiled_scene: skipping non-rigid entity '{getattr(entity, 'name', entity)}' "
                f"({type(entity).__name__}); only RigidEntity is supported."
            )
            continue
        if getattr(entity, "_enable_heterogeneous", False):
            gs.logger.warning(f"save_compiled_scene: skipping heterogeneous entity '{entity.name}' (not supported).")
            continue
        if len(getattr(entity, "equalities", ())) > 0:
            gs.logger.warning(
                f"save_compiled_scene: entity '{entity.name}' has equality constraints which are NOT captured "
                "in v1; the reloaded entity will omit them."
            )
        ei = len(entities_meta)
        entities_meta.append(_capture_entity(entity, ei, arrays))

    manifest = {
        "format_version": BUNDLE_FORMAT_VERSION,
        "genesis_version": gs.__version__,
        "n_envs": int(scene.n_envs),
        "entities": entities_meta,
    }

    with open(os.path.join(path, _MANIFEST_NAME), "w") as f:
        json.dump(manifest, f, indent=2)
    np.savez(os.path.join(path, _ARRAYS_NAME), **arrays)

    gs.logger.info(f"Saved compiled scene ({len(entities_meta)} rigid entit(ies)) to ~~<{path}>~~.")
    return path


# --------------------------------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------------------------------


def _make_surface(color):
    surface = gs.surfaces.Default()
    try:
        surface.set_color(tuple(float(c) for c in color))
    except Exception:
        pass
    return surface


def _reconstruct_geom(meta, ei, li, gi, data, sdf_params):
    """Rebuild a collision g_info dict (with final gs.Mesh) and the SDF dict for the `.gsd` cache."""

    def get(name):
        return data[_akey(f"e{ei}", f"l{li}", f"g{gi}", name)]

    # Keep the native (float64) vertex dtype: `RigidGeom._init_verts` is trimesh-native float64 regardless of
    # `gs.precision`, and the `.gsd` cache key hashes the raw vertex bytes. Downcasting here would change the
    # hash (SDF cache miss -> recompute) and perturb the geometry by ~1e-7.
    verts = np.ascontiguousarray(get("verts"))
    faces = np.ascontiguousarray(get("faces"))
    normals = np.ascontiguousarray(get("normals"))
    uvs = np.ascontiguousarray(get("uvs")) if meta["has_uvs"] else None

    metadata = dict(meta["metadata"])
    mesh = gs.Mesh.from_attrs(
        verts=verts, faces=faces, normals=normals, surface=gs.surfaces.Default(), uvs=uvs, metadata=metadata
    )

    g_info = {
        "mesh": mesh,
        "pos": np.asarray(get("init_pos"), dtype=gs.np_float),
        "quat": np.asarray(get("init_quat"), dtype=gs.np_float),
        "type": gs.GEOM_TYPE(meta["type"]),
        "data": np.asarray(get("data"), dtype=gs.np_float),
        "contype": meta["contype"],
        "conaffinity": meta["conaffinity"],
        "friction": meta["friction"],
        "sol_params": np.asarray(get("sol_params"), dtype=gs.np_float),
    }

    # SDF dict keyed by the *final* mesh verts/faces + material sdf params. Compute the key from the rebuilt
    # mesh's own verts/faces (exactly what RigidGeom._preprocess() will hash) so the pre-written cache hits.
    gsd_dict = {
        "sdf_val": np.ascontiguousarray(get("sdf_val")),
        "sdf_grad": np.ascontiguousarray(get("sdf_grad")),
        "sdf_max": get("sdf_max").item() if get("sdf_max").ndim == 0 else np.max(get("sdf_val")),
        "sdf_closest_vert": np.ascontiguousarray(get("sdf_closest_vert")),
        "T_mesh_to_sdf": np.ascontiguousarray(get("T_mesh_to_sdf")),
    }
    gsd_path = mu.get_gsd_path(
        mesh.verts, mesh.faces, sdf_params["sdf_cell_size"], sdf_params["sdf_min_res"], sdf_params["sdf_max_res"]
    )
    return g_info, gsd_path, gsd_dict


def _reconstruct_vgeom(meta, ei, li, vi, data):
    def get(name):
        return data[_akey(f"e{ei}", f"l{li}", f"v{vi}", name)]

    verts = np.asarray(get("vverts"), dtype=gs.np_float)
    faces = np.asarray(get("vfaces"), dtype=gs.np_int)
    normals = np.asarray(get("vnormals"), dtype=gs.np_float) if meta["has_normals"] else None
    uvs = np.asarray(get("uvs"), dtype=gs.np_float) if meta["has_uvs"] else None

    vmesh = gs.Mesh.from_attrs(
        verts=verts,
        faces=faces,
        normals=normals,
        surface=_make_surface(get("color")),
        uvs=uvs,
        metadata={"name": meta["name"]},
    )
    return {
        "vmesh": vmesh,
        "pos": np.asarray(get("init_pos"), dtype=gs.np_float),
        "quat": np.asarray(get("init_quat"), dtype=gs.np_float),
        "contype": 0,
        "conaffinity": 0,
    }


def _reconstruct_joint(meta, ei, li, ji, data):
    def get(name):
        return data[_akey(f"e{ei}", f"l{li}", f"j{ji}", name)]

    n_dofs = meta["n_dofs"]
    return {
        "name": meta["name"],
        "type": gs.JOINT_TYPE(meta["type"]),
        "n_qs": meta["n_qs"],
        "n_dofs": n_dofs,
        "pos": np.asarray(get("pos"), dtype=gs.np_float),
        "quat": np.asarray(get("quat"), dtype=gs.np_float),
        "init_qpos": np.asarray(get("init_qpos"), dtype=gs.np_float),
        "sol_params": np.asarray(get("sol_params"), dtype=gs.np_float),
        "dofs_motion_ang": np.asarray(get("dofs_motion_ang"), dtype=gs.np_float),
        "dofs_motion_vel": np.asarray(get("dofs_motion_vel"), dtype=gs.np_float),
        "dofs_limit": np.asarray(get("dofs_limit"), dtype=gs.np_float),
        "dofs_invweight": np.asarray(get("dofs_invweight"), dtype=gs.np_float),
        "dofs_frictionloss": np.asarray(get("dofs_frictionloss"), dtype=gs.np_float),
        "dofs_stiffness": np.asarray(get("dofs_stiffness"), dtype=gs.np_float),
        "dofs_damping": np.asarray(get("dofs_damping"), dtype=gs.np_float),
        "dofs_armature": np.asarray(get("dofs_armature"), dtype=gs.np_float),
        "dofs_act_gain": np.asarray(get("dofs_act_gain"), dtype=gs.np_float),
        "dofs_act_bias": np.asarray(get("dofs_act_bias"), dtype=gs.np_float),
        "dofs_force_range": np.asarray(get("dofs_force_range"), dtype=gs.np_float),
    }


def _reconstruct_entity(emeta, ei, data):
    """Rebuild (l_infos, links_j_infos, links_g_infos, eqs_info) and the list of `.gsd` cache entries."""
    sdf_params = {
        "sdf_cell_size": emeta["material"].get("sdf_cell_size", 0.005),
        "sdf_min_res": emeta["material"].get("sdf_min_res", 32),
        "sdf_max_res": emeta["material"].get("sdf_max_res", 128),
    }

    l_infos, links_j_infos, links_g_infos = [], [], []
    gsd_entries = []
    for li, lmeta in enumerate(emeta["links"]):

        def larr(name):
            return data[_akey(f"e{ei}", f"l{li}", name)]

        l_info = {
            "name": lmeta["name"],
            "parent_idx": lmeta["parent_idx"],
            "is_robot": np.array(lmeta["is_robot"], dtype=np.bool_),
            "pos": np.asarray(larr("pos"), dtype=gs.np_float),
            "quat": np.asarray(larr("quat"), dtype=gs.np_float),
            "inertial_mass": lmeta["inertial_mass"],
        }
        if lmeta["root_idx"] is not None:
            l_info["root_idx"] = lmeta["root_idx"]
        for name in ("inertial_pos", "inertial_quat", "inertial_i", "invweight"):
            l_info[name] = np.asarray(larr(name), dtype=gs.np_float) if lmeta[f"has_{name}"] else None

        j_infos = [_reconstruct_joint(jm, ei, li, ji, data) for ji, jm in enumerate(lmeta["joints"])]

        g_infos = []
        for gi, gm in enumerate(lmeta["geoms"]):
            g_info, gsd_path, gsd_dict = _reconstruct_geom(gm, ei, li, gi, data, sdf_params)
            g_infos.append(g_info)
            gsd_entries.append((gsd_path, gsd_dict))
        for vi, vm in enumerate(lmeta["vgeoms"]):
            g_infos.append(_reconstruct_vgeom(vm, ei, li, vi, data))

        l_infos.append(l_info)
        links_j_infos.append(j_infos)
        links_g_infos.append(g_infos)

    return l_infos, links_j_infos, links_g_infos, [], gsd_entries


def _prewrite_gsd_cache(gsd_entries):
    """Pre-populate the `.gsd` SDF cache so RigidGeom._preprocess() loads instead of recomputing."""
    import pickle as pkl

    cache_dir = get_gsd_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    for gsd_path, gsd_dict in gsd_entries:
        if os.path.exists(gsd_path):
            continue
        with open(gsd_path, "wb") as f:
            pkl.dump(gsd_dict, f)


def load_compiled_scene(path, scene=None, material=None, surface=None, **scene_kwargs):
    """Load a compiled-scene bundle, adding its entities to a scene without re-parsing assets.

    Parameters
    ----------
    path : str | os.PathLike
        Path to a bundle directory produced by :func:`save_compiled_scene`.
    scene : genesis.Scene | None
        Scene to add entities to. If None, a new (unbuilt) :class:`genesis.Scene` is created using
        ``scene_kwargs``. The returned scene is NOT built; call ``scene.build()`` afterwards.
    material : genesis.materials.Rigid | None
        Override material for all loaded entities. If None, the per-entity material captured in the bundle
        is restored.
    surface : genesis.surfaces.Surface | None
        Override surface for all loaded entities. If None, a default surface is used.
    **scene_kwargs
        Forwarded to :class:`genesis.Scene` when ``scene`` is None.

    Returns
    -------
    tuple[genesis.Scene, list[genesis.engine.entities.RigidEntity]]
        The scene and the list of loaded entities.
    """
    path = os.fspath(path)
    with open(os.path.join(path, _MANIFEST_NAME)) as f:
        manifest = json.load(f)

    if manifest.get("format_version") != BUNDLE_FORMAT_VERSION:
        gs.raise_exception(
            f"Unsupported compiled-scene format version {manifest.get('format_version')} "
            f"(expected {BUNDLE_FORMAT_VERSION}). Re-export the bundle with this Genesis version."
        )
    if manifest.get("genesis_version") != gs.__version__:
        gs.logger.warning(
            f"Compiled scene was created with Genesis {manifest.get('genesis_version')} but the current version is "
            f"{gs.__version__}. Loading anyway; rebuild the bundle if you hit inconsistencies."
        )

    data = np.load(os.path.join(path, _ARRAYS_NAME))

    if scene is None:
        scene = gs.Scene(**scene_kwargs)

    entities = []
    for ei, emeta in enumerate(manifest["entities"]):
        l_infos, links_j_infos, links_g_infos, eqs_info, gsd_entries = _reconstruct_entity(emeta, ei, data)

        # Pre-populate the SDF cache so the build loads it instead of recomputing.
        _prewrite_gsd_cache(gsd_entries)

        compiled_data = {
            "l_infos": l_infos,
            "links_j_infos": links_j_infos,
            "links_g_infos": links_g_infos,
            "eqs_info": eqs_info,
            "is_local_collision_mask": emeta.get("is_local_collision_mask", False),
        }
        morph = gs.morphs._Compiled(
            requires_jac_and_IK=emeta.get("requires_jac_and_IK", False),
            compiled_data=compiled_data,
        )

        entity_material = material
        if entity_material is None:
            entity_material = gs.materials.Rigid(**emeta["material"])

        entity = scene.add_entity(morph, material=entity_material, surface=surface, name=emeta.get("name"))
        entities.append(entity)

    return scene, entities
