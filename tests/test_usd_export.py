"""Round-trip tests for the USD scene exporter (scene.export_usd / gs.utils.usd.export_scene_to_usd).

Exports a built scene to USD, re-imports it via add_stage, and checks that geometry/structure and simulation
behavior are preserved. Requires the optional `usd-core` dependency.
"""

import numpy as np
import pytest

import genesis as gs

from .utils import assert_allclose

pytest.importorskip("pxr", reason="USD export requires the 'usd-core' package")


def _max_geoms_entity(scene):
    """Return the entity with the most collision geoms (e.g. a mesh object, not the 1-geom ground plane)."""
    return max(scene.entities, key=lambda e: sum(len(link.geoms) for link in e.links))


def _articulated_entity(scene):
    return next(e for e in scene.entities if e.n_dofs > 0)


@pytest.mark.required
def test_usd_export_primitives_roundtrip(show_viewer, tmp_path, tol):
    """Primitive objects (box/sphere on a plane) re-import with identical geom types/data and drop identically."""
    scene = gs.Scene(show_viewer=show_viewer)
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(gs.morphs.Box(size=(0.1, 0.2, 0.3), pos=(0.0, 0.0, 1.0)))
    scene.add_entity(gs.morphs.Sphere(radius=0.15, pos=(0.5, 0.0, 1.0)))
    scene.build()

    def geom_signature(s):
        return sorted(
            (g.type.name, tuple(np.round(np.asarray(g._data)[:3], 5)))
            for e in s.entities
            for link in e.links
            for g in link.geoms
        )

    ref_sig = geom_signature(scene)
    for _ in range(40):
        scene.step()
    ref_pos = np.stack([np.asarray(e.get_pos().cpu()) for e in scene.entities[1:]])

    usd_path = tmp_path / "objects.usda"
    scene.export_usd(usd_path)

    scene2 = gs.Scene(show_viewer=show_viewer)
    scene2.add_stage(gs.morphs.USD(file=str(usd_path)))
    scene2.build()

    assert geom_signature(scene2) == ref_sig
    for _ in range(40):
        scene2.step()
    new_pos = np.stack([np.asarray(e.get_pos().cpu()) for e in scene2.entities[1:]])
    assert_allclose(new_pos, ref_pos, tol=tol)


@pytest.mark.required
def test_usd_export_articulation_roundtrip(show_viewer, tmp_path):
    """A fixed-base KUKA URDF re-imports with the same DoFs/joint types and an (almost) identical rollout."""
    scene = gs.Scene(show_viewer=show_viewer)
    scene.add_entity(gs.morphs.Plane())
    kuka = scene.add_entity(gs.morphs.URDF(file="urdf/kuka_iiwa/model.urdf", fixed=True, pos=(0.0, 0.0, 0.0)))
    scene.build()

    def rollout(s, robot, q_init, n=40):
        robot.set_dofs_position(q_init)
        traj = []
        for _ in range(n):
            s.step()
            traj.append(robot.get_dofs_position().cpu().numpy().copy())
        return np.stack(traj)

    q0 = np.array([0.3, -0.4, 0.2, 0.5, -0.1, 0.2, 0.0])
    ref_types = [j.type.name for link in kuka.links for j in link.joints]
    ref_traj = rollout(scene, kuka, q0)

    usd_path = tmp_path / "kuka.usda"
    scene.export_usd(usd_path)

    scene2 = gs.Scene(show_viewer=show_viewer)
    scene2.add_stage(gs.morphs.USD(file=str(usd_path)))
    scene2.build()
    kuka2 = _articulated_entity(scene2)

    assert kuka2.n_dofs == kuka.n_dofs
    assert [j.type.name for link in kuka2.links for j in link.joints] == ref_types
    new_traj = rollout(scene2, kuka2, q0)
    # Joint frames, axes, limits and passive dynamics (damping/armature) are exported, so the free-fall
    # rollout matches to within solver float precision.
    assert_allclose(new_traj, ref_traj, atol=1e-5)


@pytest.mark.required
def test_usd_export_requires_built(show_viewer, tmp_path):
    """export_usd raises on an unbuilt scene."""
    scene = gs.Scene(show_viewer=show_viewer)
    scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0.0, 0.0, 0.5)))
    with pytest.raises(gs.GenesisException):
        scene.export_usd(tmp_path / "unbuilt.usda")
