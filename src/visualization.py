from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
import torch
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import tile_images, draw_collision


def observations_to_image(observation: Dict, info: Dict) -> np.ndarray:
    r"""Generate image of single frame from observation and info
    returned from a single environment step().

    Args:
        observation: observation returned from an environment step().
        info: info returned from an environment step().

    Returns:
        generated image of a single frame.
    """
    render_obs_images: List[np.ndarray] = []
    for sensor_name in observation:
        if len(observation[sensor_name].shape) > 1:
            if sensor_name == 'precise_corr':
                obs_k = observation[sensor_name]
                c1, c2, c3 = torch.split(obs_k, [9, 9, 9], dim=0)   # c1是16，c2是8,c3是4

                r = 1
                dx = torch.linspace(-r, r, 2 * r + 1)
                dy = torch.linspace(-r, r, 2 * r + 1)
                delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).view(9, -1).numpy()

                _, max_indices1 = torch.max(c1, dim=0)
                # 创建一个512x512的白色背景图片
                img_size, _, _ = observation['imagegoal_sensor_v2'].shape
                image = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
                # 分成16x16个网格
                grid_size = img_size // 16
                # 定义箭头的长度
                arrow_length = img_size // 18
                # 在每个网格中心绘制箭头
                for i, x in enumerate(range(0, img_size, grid_size)):
                    for j, y in enumerate(range(0, img_size, grid_size)):
                        center_x = x + grid_size // 2
                        center_y = y + grid_size // 2
                        index = max_indices1[j,i]
                        end_x = center_x + arrow_length * int(delta[index][1])
                        end_y = center_y + arrow_length * int(delta[index][0])
                        cv2.arrowedLine(image, (center_x, center_y), (end_x, end_y), (0, 0, 0), thickness=1, tipLength=0.2)
                render_obs_images.append(image)


            else:
                obs_k = observation[sensor_name]
                if not isinstance(obs_k, np.ndarray):
                    obs_k = obs_k.cpu().numpy()
                if obs_k.dtype != np.uint8:
                    obs_k = obs_k * 255.0
                    obs_k = obs_k.astype(np.uint8)
                if obs_k.shape[2] == 1:
                    obs_k = np.concatenate([obs_k for _ in range(3)], axis=2)
                render_obs_images.append(obs_k)

    assert (
        len(render_obs_images) > 0
    ), "Expected at least one visual sensor enabled."

    shapes_are_equal = len(set(x.shape for x in render_obs_images)) == 1
    if not shapes_are_equal:
        render_frame = tile_images(render_obs_images)
    else:
        render_frame = np.concatenate(render_obs_images, axis=1)

    # draw collision
    if "collisions.is_collision" in info and info["collisions.is_collision"]:
        render_frame = draw_collision(render_frame)

    if "top_down_map.map" in info:
        info_top_down_map = {
            'map': info['top_down_map.map'],
            'fog_of_war_mask': info['top_down_map.fog_of_war_mask'],
            'agent_map_coord': info['top_down_map.agent_map_coord'],
            'agent_angle': info['top_down_map.agent_angle'],
        }
        top_down_map = maps.colorize_draw_agent_and_fit_to_height(
            info_top_down_map, render_frame.shape[0]
        )
        render_frame = np.concatenate((render_frame, top_down_map), axis=1)
    return render_frame