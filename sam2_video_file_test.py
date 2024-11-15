import torch
import cv2
import numpy as np

class Environment():
    def __init__(self, env_name):
        self.env = cv2.VideoCapture(env_name)
        if not self.env.isOpened():
            print("Error opening video stream or file")

    def reset(self):
        obs, _reward, _done, _truncated, _info = self.step(0)
        return obs, {}

    def step(self, action):
        # ret is True if the frame is available
        ret, frame = self.env.read()
        done = not ret and frame is None

        state_observation = {
            "tcp_pose": np.zeros((7,)),
            "tcp_vel": np.zeros((6,)),
            "gripper_pose": 0,
            "tcp_force": np.zeros((3,)),
            "tcp_torque": np.zeros((3,)),
        }

        # obs, reward, done, truncated, info
        obs = dict(images=frame, state=state_observation)
        return obs, 0., done, False, {}

    def release(self):
        self.env.release()


class Agent():
    def sample_actions(self, observations):
        return 0


def gym_make(env_name):
    # swap with "FrankaPegInsert-Vision-v0"
    env_name = "../test/2024-10-30_21-55-00_failed.mp4"
    return Environment(env_name)


def sam2_predictor():
    from sam2.build_sam import build_sam2_camera_predictor
    checkpoint = "../checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    predictor = build_sam2_camera_predictor(model_cfg, checkpoint)
    return predictor


import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

COLORS = ['red', 'green', 'blue', 'purple', 'yellow', 'orange']


def color_by_index(index):
    color_idx = index % len(COLORS)
    color = to_rgb(COLORS[color_idx])
    return np.array(color) * 255.


def draw_segmentation(image, masks, object_ids, points=None):
    source = image
    if torch.is_tensor(image):
        source = np.transpose(image.clone().detach().numpy(), (1, 2, 0))
        canvas = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    canvas = np.ascontiguousarray(source)
    if canvas.max() <= 1:
        canvas = canvas * 255.0

    if object_ids is None:
        object_ids = torch.arange(1, masks.shape[0] + 1)

    for mask, object_id in zip(masks, object_ids):
        color = color_by_index(object_id)
        _mask = mask.clone().detach().cpu().numpy().squeeze()
        _mask = _mask > 0
        update = 0.6
        canvas[_mask, :] = (1.0 - update) * canvas[_mask, :] + update * color
        canvas = np.clip(canvas, 0, 255).astype(np.uint8)

    if points is not None:
        points = points.astype(np.int16).squeeze()
        for point, object_id in zip(points, object_ids):
            color = color_by_index(object_id)
            cv2.circle(canvas, point, radius=10, color=color, thickness=-1)
            cv2.circle(canvas, point, radius=11, color=(255, 255, 255), thickness=2)

    return canvas


def show_image(image):
    plt.figure(1, figsize=(20, 20))
    plt.axis('off')
    plt.imshow(image)
    plt.waitforbuttonpress()
    plt.close(1)


def save_observations(observations, name=""):
    if not len(observations):
        return
    if not name:
        from datetime import datetime
        name = f'../test/{datetime.now():%Y-%m-%d_%H-%M-%S}.mp4'
    try:
        encoder = cv2.VideoWriter_fourcc(*"mp4v")
        size = observations[0].shape[:2][::-1]  # width, height
        writer = cv2.VideoWriter(name, encoder, 10, size)

        for frame in observations:
            writer.write(frame)
        writer.release()
    except Exception as e:
        print(f"Failed to save video: {e}")


import time


def main():
    predictor = sam2_predictor()

    # emulate gym environment
    env = gym_make("FrankaPegInsert-Vision-v0")

    # emulate SERL SAC agent
    agent = Agent()

    # reset environment and acquire frames from camera
    obs, _ = env.reset()
    frame = obs["images"]  # assuming images are stacked vertically

    # add new prompts and instantly get the output on the same frame
    predictor.load_first_frame(frame)

    points = np.array([
        np.array([[265, 120]], dtype=np.float32),  # peg
        np.array([[275, 320]], dtype=np.float32)  # hole
    ])
    label = np.array([1], dtype=np.int32)  # value of 1 marks foreground point

    # track first object
    frame_idx, object_ids, masks = predictor.add_new_points(
        frame_idx=0, obj_id=1, points=points[0], labels=label
    )

    # track second object - this call returns all masks and all object ids
    frame_idx, object_ids, masks = predictor.add_new_points(
        frame_idx=0, obj_id=2, points=points[1], labels=label
    )

    # overlay masks on the frame captured from the cameras
    next_obs = draw_segmentation(frame, masks, object_ids, points)
    observations = [next_obs]
    show_image(next_obs)

    # measure delay
    delay_tracking = []
    delay_overlay = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        while True:
            print(f"step {len(observations)}")
            # take action and observe next state
            action = agent.sample_actions(observations=next_obs)
            obs, _reward, done, _truncated, _info = env.step(action=action)
            if done:
                break

            # run inference using masks as input
            time_track = time.time()
            frame = obs["images"]
            object_ids, masks = predictor.track(frame)
            delay_tracking.append(time.time() - time_track)

            # overlay mask segments
            time_overlay = time.time()
            next_obs = draw_segmentation(frame, masks, object_ids)
            delay_overlay.append(time.time() - time_overlay)
            show_image(next_obs)

            # collect observations
            observations.append(next_obs)

    env.release()  # release camera - only needed when loading from a video file
    save_observations(observations, name="../test/2024-10-30_21-55-00_failed-SEG.mp4")

    delay_tracking = np.array(delay_tracking) * 1000
    delay_overlay = np.array(delay_overlay) * 1000
    print(f"tracking delay (avg) {delay_tracking.mean():.2f}ms\n"
          f"overlay delay (avg) {delay_overlay.mean():.2f}ms")

if __name__ == "__main__":
    import os; print(os.getcwd())
    main()