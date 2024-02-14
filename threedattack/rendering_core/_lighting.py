from typing import TypeVar, cast

from panda3d.core import AmbientLight, DirectionalLight, NodePath, Spotlight

from ._errors import Panda3dAssumptionViolation
from ._scene_config import SceneConfigDict
from ._scene_util import find_node


def configure_simplepbr_lighting_internal(
    render: NodePath, model_root: NodePath, scene_config: SceneConfigDict
) -> None:
    """
    Enable simplepbr shadows and add ambient lighting on the specified node hierarchy.

    This function is INTERNAL to the "rendering_core" module and should not be invoked outside of it.

    Parameters
    ----------
    render
        The ``showbase.render`` node.
    model_root
        The root of the node hierarchy to process.
    scene_config
        The configuration of the lighting of the scene.

    Notes
    -----
    This function does not add shadows to the nondirectional point lights, since simplepbr does not have shadow support for them yet.

    Raises
    ------
    Panda3dAssumptionViolation
        If the ``showbase.render`` already has a child, called "Ambient".
    """
    lights = model_root.find_all_matches("**/+Light")
    for light in lights:
        light_node = light.node()
        shadow_area_root_name = scene_config["shadow_area_root"]
        shadow_area_root = find_node(render, "**/" + shadow_area_root_name)
        assert shadow_area_root is not None
        if isinstance(light_node, DirectionalLight) or isinstance(
            light_node, Spotlight
        ):
            light = cast("NodePath[DirectionalLight | Spotlight]", light)
            _enable_shadows(
                render=render,
                shadow_area_root=shadow_area_root,
                light=light,
                shadow_map_resoution=scene_config["shadow_map_resolution"],
            )
            _add_ambient_component(
                render=render,
                light=light,
                ambient_hpr_diff=scene_config["dir_ambient_hpr_diff"],
                ambient_weight=scene_config["dir_ambient_hpr_weight"],
            )
        render.setLight(light)
    _add_nondirectional_ambient_light(
        render,
        nondir_ambient_rgb=(
            scene_config["nondir_ambient_light_r"],
            scene_config["nondir_ambient_light_g"],
            scene_config["nondir_ambient_light_b"],
        ),
    )


T = TypeVar(
    "T",
    bound=DirectionalLight | Spotlight,
)


def _enable_shadows(
    render: NodePath,
    shadow_area_root: NodePath,
    light: "NodePath[T]",
    shadow_map_resoution: int,
) -> None:
    """
    Emable shadows on the specified light.

    Parameters
    ----------
    render
        The ``showbase.render`` object.
    shadow_area_root
        The root object of the object for which the shadow map should be calculated. Generally, this does not have to be necessarily the root of all objects, since you do not necessarily want to shadow all objects.
    light
        The light on which the shadows should be enabled.
    shadow_map_resoution
        The desired resolution of the shadow map.

    Returns
    -------

    Raises
    ------
    ValueError
        If the shadow map resolution is negative.
    """
    if shadow_map_resoution < 0:
        raise ValueError(
            f"The shadow map resolution {shadow_map_resoution} is negative."
        )

    light_node = light.node()
    light_node.setShadowCaster(True, shadow_map_resoution, shadow_map_resoution)
    render.set_light(light)

    tight_bounds = shadow_area_root.get_tight_bounds(light)
    assert tight_bounds is not None
    bmin, bmax = tight_bounds
    light_lens = light.node().get_lens()

    light_lens.set_film_offset((bmin.xz + bmax.xz) * 0.5)
    light_lens.set_film_size(bmax.xz - bmin.xz)
    light_lens.set_near_far(bmin.y, bmax.y)


def _add_ambient_component(
    render: NodePath,
    light: "NodePath[T]",
    ambient_weight: float,
    ambient_hpr_diff: float,
) -> None:
    """
    Add non-shadow casting directional components to the light.

    This is a directional alternative to the non-directional ambient lights, provided by Panda3d. This directional ambient-like light does not maek the objects so flat.

    There are four directional components. Their total intensity is equal to ``original_intensity*ambient_weight``. Their directions are calculated by adding and substracting ``ambient_hpr_diff`` from the H and P component of the rotation of the original light.

    This function reduces the intensity of the original light to ``original_intensity*(1-ambient_weight)``.

    Parameters
    ----------
    render
        The ``showbase.render`` object.
    light
        The original light.
    ambient_weight
        The total weight of the ambient-like directional components.
    ambient_hpr_diff
        The absolute difference of the H and P components of the ambient lights.

    Raises
    ------
    ValueError
        If ``0<=ambient_weight<=1`` is false.
    """
    light_node = light.node()
    light_color = (
        light_node.getColor()
    )  # light.get_color() would report that the light has no color

    ambient_hprs: list[tuple[float, float, float]] = [
        (ambient_hpr_diff, ambient_hpr_diff, 0),
        (ambient_hpr_diff, -ambient_hpr_diff, 0),
        (-ambient_hpr_diff, -ambient_hpr_diff, 0),
        (-ambient_hpr_diff, ambient_hpr_diff, 0),
    ]

    for ambient_hpr in ambient_hprs:
        ambient = light.attach_new_node(DirectionalLight("ambient"))
        ambient.setHpr(ambient_hpr)
        ambient.node().set_color(
            (
                light_color.x * ambient_weight / 4,
                light_color.y * ambient_weight / 4,
                light_color.z * ambient_weight / 4,
                1,
            )
        )
        render.set_light(ambient)
    light_node.set_color(
        (
            light_color.x * (1 - ambient_weight),
            light_color.y * (1 - ambient_weight),
            light_color.z * (1 - ambient_weight),
            1,
        )
    )


def _add_nondirectional_ambient_light(
    render: NodePath, nondir_ambient_rgb: tuple[float, float, float]
) -> None:
    """
    Add a non-directional ambient light (with name "Ambient") to the scene.

    Parameters
    ----------
    render
        The ``showbase.render`` object.
    nondir_ambient_rgb
        The color of the new light.

    Raises
    ------
    Panda3dAssumptionViolation
        If the ``showbase.render`` already has a child, called "Ambient".
    """
    AMBIENT_NAME = "Ambient"

    obj_called_ambient = find_node(render, AMBIENT_NAME)
    if obj_called_ambient is not None:
        raise Panda3dAssumptionViolation(
            'The render object already has a child, called "Ambient".'
        )
    assert obj_called_ambient is None

    ambient_light = render.attach_new_node(AmbientLight(AMBIENT_NAME))
    ambient_light_node = ambient_light.node()
    ambient_light_node.set_color(
        (nondir_ambient_rgb[0], nondir_ambient_rgb[1], nondir_ambient_rgb[2], 1)
    )
    render.set_light(ambient_light)
