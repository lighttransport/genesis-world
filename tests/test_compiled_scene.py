"""Tests for the compiled-scene fast-path (genesis.save_compiled_scene / load_compiled_scene).

Verifies that reloading a scene from a compiled bundle reproduces a fresh build bit-for-bit (collision
geometry + SDF) and yields identical simulation behavior, for rigid objects (primitive + mesh, including
convex decomposition) and a URDF articulation.
"""

import numpy as np
import pytest

import genesis as gs

from .utils import assert_allclose, assert_equal


def _build_reference_scene(show_viewer):
    scene = gs.Scene(show_viewer=show_viewer)
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(gs.morphs.Box(size=(0.1, 0.12, 0.14), pos=(0.0, 0.0, 0.5)))
    scene.add_entity(gs.morphs.Mesh(file="meshes/bunny.obj", scale=0.3, pos=(0.3, 0.0, 0.4)))
    scene.add_entity(gs.morphs.URDF(file="urdf/kuka_iiwa/model.urdf", fixed=True, pos=(0.6, 0.0, 0.0)))
    scene.build()
    return scene


def _collect_geom_state(scene):
    """Map (entity_name, link_idx, geom_idx) -> dict of final collision-geom arrays."""
    state = {}
    for entity in scene.entities:
        for li, link in enumerate(entity.links):
            for gi, geom in enumerate(link.geoms):
                state[(entity.name, li, gi)] = {
                    "verts": np.array(geom.init_verts),
                    "faces": np.array(geom.init_faces),
                    "sdf_val": np.array(geom.sdf_val),
                    "sdf_grad": np.array(geom.sdf_grad),
                    "sdf_closest_vert": np.array(geom.sdf_closest_vert),
                    "type": int(geom.type),
                }
    return state


def _rollout(scene, kuka, n_steps=30):
    kuka.set_dofs_position(np.array([0.3, -0.4, 0.2, 0.5, -0.1, 0.2, 0.0]))
    traj = []
    for _ in range(n_steps):
        scene.step()
        traj.append(kuka.get_dofs_position().cpu().numpy().copy())
    return np.asarray(traj)


@pytest.mark.required
def test_compiled_scene_roundtrip(show_viewer, tmp_path, tol):
    """A reloaded compiled scene reproduces collision geometry/SDF bit-exactly and simulates identically."""
    # Fresh build: capture geometry and a control rollout.
    scene = _build_reference_scene(show_viewer)
    ref_geoms = _collect_geom_state(scene)
    kuka = scene.entities[-1]
    assert kuka.n_dofs == 7
    ref_traj = _rollout(scene, kuka)

    bundle = tmp_path / "scene.gscene"
    scene.save_compiled(bundle)

    # Reload into a new scene without re-parsing / re-decomposing / recomputing SDF.
    scene2, entities = gs.load_compiled_scene(bundle, show_viewer=show_viewer)
    assert len(entities) == len(scene.entities)
    scene2.build()

    new_geoms = _collect_geom_state(scene2)
    assert set(new_geoms) == set(ref_geoms)
    for key, ref in ref_geoms.items():
        new = new_geoms[key]
        assert new["type"] == ref["type"], key
        # Geometry and SDF are loaded verbatim from the bundle -> must be bit-exact.
        assert_equal(new["verts"], ref["verts"], err_msg=f"verts mismatch for {key}")
        assert_equal(new["faces"], ref["faces"], err_msg=f"faces mismatch for {key}")
        assert_equal(new["sdf_val"], ref["sdf_val"], err_msg=f"sdf_val mismatch for {key}")
        assert_equal(new["sdf_grad"], ref["sdf_grad"], err_msg=f"sdf_grad mismatch for {key}")
        assert_equal(new["sdf_closest_vert"], ref["sdf_closest_vert"], err_msg=f"sdf_closest_vert for {key}")

    # Simulation parity: identical control from identical initial state -> identical trajectory.
    kuka2 = scene2.entities[-1]
    new_traj = _rollout(scene2, kuka2)
    assert_allclose(new_traj, ref_traj, tol=tol)


@pytest.mark.required
def test_compiled_scene_requires_built(show_viewer, tmp_path):
    """save_compiled raises on an unbuilt scene."""
    scene = gs.Scene(show_viewer=show_viewer)
    scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0.0, 0.0, 0.5)))
    with pytest.raises(gs.GenesisException):
        scene.save_compiled(tmp_path / "unbuilt.gscene")
