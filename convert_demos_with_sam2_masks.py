import torch
import cv2
import numpy as np
import pickle as pkl

class Environment():
    def __init__(self, demo):
        # self.env = cv2.VideoCapture(env_name)
        # if not self.env.isOpened():
        #     print("Error opening video stream or file")
        self.demo = demo
        self.transitions = demo
        self.steps = 0
        print('debug')

    def reset(self):
        # obs, _reward, _done, _truncated, _info = self.step(0)
        obs_1 = self.demo[self.steps]['observations']['wrist_1']
        obs_2 = self.demo[self.steps]['observations']['wrist_2']
        obs = np.squeeze(np.concatenate((obs_1, obs_2), axis=1))
        self.steps += 1
        return obs

    def step(self):
        # # ret is True if the frame is available
        # ret, frame = self.env.read()
        # done = not ret and frame is None
        #
        # state_observation = {
        #     "tcp_pose": np.zeros((7,)),
        #     "tcp_vel": np.zeros((6,)),
        #     "gripper_pose": 0,
        #     "tcp_force": np.zeros((3,)),
        #     "tcp_torque": np.zeros((3,)),
        # }
        #
        # # obs, reward, done, truncated, info
        # obs = dict(images=frame, state=state_observation)
        obs_1 = self.demo[self.steps]['observations']['wrist_1']
        obs_2 = self.demo[self.steps]['observations']['wrist_2']
        obs = np.squeeze(np.concatenate((obs_1, obs_2), axis=1))

        done = self.demo[self.steps]['dones']
        self.steps += 1

        return obs, 0., done, False, {}

    def release(self):
        self.env.release()

    def insert_obs(self, obs):

        self.transitions[self.steps -1]['observations']['wrist_1'][0] = obs[:128,:,:]
        self.transitions[self.steps -1]['observations']['wrist_2'][0] = obs[128:,:,:]



class Agent():
    def sample_actions(self, observations):
        return 0


def gym_make(file_name):
    # swap with "FrankaPegInsert-Vision-v0"
    # env_name = "../test/2024-10-30_21-55-00_failed.mp4"
    demo_path = f'/home/ev/serl/examples/async_peg_insert_drq/{file_name}.pkl'
    with open(demo_path, "rb") as f:
        demo = pkl.load(f)
    return Environment(demo)


def sam2_predictor():
    from sam2.build_sam import build_sam2_camera_predictor
    checkpoint = "./checkpoints/sam2_hiera_large.pt"
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

    mask_all = np.zeros(masks.size()[2:]) > 0
    for mask, object_id in zip(masks, object_ids):
        color = color_by_index(object_id)
        _mask = mask.clone().detach().cpu().numpy().squeeze()
        _mask = _mask > 0
        mask_all += _mask

    new_canvas = 255*np.ones_like(source)
    new_canvas[mask_all,:] = canvas[mask_all,:]
    new_canvas = np.clip(new_canvas, 0, 255).astype(np.uint8)

    if points is not None:
        points = points.astype(np.int16).squeeze()
        for point, object_id in zip(points, object_ids):
            color = color_by_index(object_id)
            cv2.circle(new_canvas, point, radius=3, color=color, thickness=-1)
            cv2.circle(new_canvas, point, radius=4, color=(255, 255, 255), thickness=2)

    return new_canvas


def draw_segmentation_color(image, masks, object_ids, points=None):
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
            cv2.circle(canvas, point, radius=3, color=color, thickness=-1)
            cv2.circle(canvas, point, radius=4, color=(255, 255, 255), thickness=2)

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
            # writer.write(frame)
            writer.write(frame[:, :, [2, 1, 0]])
        writer.release()
    except Exception as e:
        print(f"Failed to save video: {e}")


import time


def main():
    predictor = sam2_predictor()

    # emulate gym environment
    file_name = ('peg_insert_20_demos_20_trials_pose_id_10_2024-11-25_12-23-39')
    env = gym_make(file_name)

    # emulate SERL SAC agent
    # agent = Agent()

    # reset environment and acquire frames from camera
    obs = env.reset()
    # frame = obs["images"]  # assuming images are stacked vertically
    frame = obs

    # add new prompts and instantly get the output on the same frame
    predictor.load_first_frame(frame)
    show_image(frame)

    points = np.array([
        np.array([[68, 65]], dtype=np.float32),  # peg 1
        np.array([[65, 80]], dtype=np.float32),  # hole 1
        np.array([[70, 190]], dtype=np.float32),  # peg 2
        np.array([[67, 210]], dtype=np.float32)  # hole 2
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

    # track third object - this call returns all masks and all object ids
    frame_idx, object_ids, masks = predictor.add_new_points(
        frame_idx=0, obj_id=3, points=points[2], labels=label
    )

    # track fourth object - this call returns all masks and all object ids
    frame_idx, object_ids, masks = predictor.add_new_points(
        frame_idx=0, obj_id=4, points=points[3], labels=label
    )

    # overlay masks on the frame captured from the cameras
    next_obs = draw_segmentation(frame, masks, object_ids, points)
    # next_obs = draw_segmentation_color(frame, masks, object_ids, points)
    show_image(next_obs)
    next_obs = draw_segmentation(frame, masks, object_ids)
    # next_obs = draw_segmentation_color(frame, masks, object_ids)
    observations = [next_obs]
    show_image(next_obs)
    env.insert_obs(next_obs)

    # measure delay
    delay_tracking = []
    delay_overlay = []
    transition_no = 0
    while env.steps < len(env.demo):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            while True:
                print(f"step {len(observations)}")
                # take action and observe next state
                # action = agent.sample_actions(observations=next_obs)
                obs, _reward, done, _truncated, _info = env.step()
                if done:
                    transition_no += 1
                    break

                # run inference using masks as input
                time_track = time.time()
                frame = obs
                object_ids, masks = predictor.track(frame)
                delay_tracking.append(time.time() - time_track)

                # overlay mask segments
                time_overlay = time.time()
                next_obs = draw_segmentation(frame, masks, object_ids)
                # next_obs = draw_segmentation_color(frame, masks, object_ids)
                delay_overlay.append(time.time() - time_overlay)
                # show_image(next_obs)

                # collect observations
                observations.append(next_obs)
                env.insert_obs(next_obs)

        # env.release()  # release camera - only needed when loading from a video file


        delay_tracking_ = np.array(delay_tracking) * 1000
        delay_overlay_ = np.array(delay_overlay) * 1000
        print(f"tracking delay (avg) {delay_tracking_.mean():.2f}ms\n"
              f"overlay delay (avg) {delay_overlay_.mean():.2f}ms")


    save_observations(observations, name=f"./transitions/{file_name}_masked.mp4")
    file_path = f'/home/ev/serl/examples/async_peg_insert_drq/{file_name}_masked.pkl'
    with open(file_path, "wb") as f:
        pkl.dump(env.transitions, f)

if __name__ == "__main__":
    import os; print(os.getcwd())
    main()