import torch
import cv2
import numpy as np
import pickle as pkl
from scipy import signal

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
        obs_1 = self.demo[self.steps]['observations']['wrist_1']
        obs_2 = self.demo[self.steps]['observations']['wrist_2']
        obs = np.squeeze(np.concatenate((obs_1, obs_2), axis=1))
        next_obs_1 = self.demo[self.steps]['next_observations']['wrist_1']
        next_obs_2 = self.demo[self.steps]['next_observations']['wrist_2']
        next_obs = np.squeeze(np.concatenate((next_obs_1, next_obs_2), axis=1))
        self.steps += 1
        return obs, next_obs

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
        next_obs_1 = self.demo[self.steps]['next_observations']['wrist_1']
        next_obs_2 = self.demo[self.steps]['next_observations']['wrist_2']
        next_obs = np.squeeze(np.concatenate((next_obs_1, next_obs_2), axis=1))
        done = self.demo[self.steps]['dones']
        self.steps += 1

        return obs, next_obs, 0., done, False, {}

    def release(self):
        self.env.release()

    def insert_obs(self, obs, next_obs):

        self.transitions[self.steps -1]['observations']['wrist_1'][0] = obs[:128,:,:]
        self.transitions[self.steps -1]['observations']['wrist_2'][0] = obs[128:,:,:]
        self.transitions[self.steps -1]['next_observations']['wrist_1'][0] = next_obs[:128,:,:]
        self.transitions[self.steps -1]['next_observations']['wrist_2'][0] = next_obs[128:,:,:]


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


def draw_segmentation(image, masks, object_ids, points=None, kernel_size=9):
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
        if object_id % 2 == 0 :
            kernel = np.ones((kernel_size,kernel_size))
            _mask = signal.convolve2d(_mask, kernel, boundary='symm', mode='same')
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


def draw_segmentation_color(image, masks, object_ids, points=None, kernel_size=9):
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
        if object_id % 2 == 0 :
            kernel = np.ones((kernel_size,kernel_size))
            _mask = signal.convolve2d(_mask, kernel, boundary='symm', mode='same')
            _mask = _mask > 0
        update = 0.6
        canvas[_mask, :] = (1.0 - update) * canvas[_mask, :] + update * color
        canvas = np.clip(canvas, 0, 255).astype(np.uint8)

    if points is not None:
        # points = points.astype(np.int16).squeeze()
        for point, object_id in zip(points, object_ids):
            point = point[0].astype(np.int16)
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
    predictor_obs = sam2_predictor()
    predictor_following_obs = sam2_predictor()

    kernel_size = 9
    # emulate gym environment
    pose_id = 1
    file_name = ('peg_insert_1_demos_1_trials_pose_id_6_2024-12-16_10-46-33')
    env = gym_make(file_name)

    # emulate SERL SAC agent
    # agent = Agent()

    # reset environment and acquire frames from camera
    obs, following_obs = env.reset()
    # frame = obs["images"]  # assuming images are stacked vertically
    frame = obs
    following_frame = following_obs

    # add new prompts and instantly get the output on the same frame
    predictor_obs.load_first_frame(frame)
    predictor_following_obs.load_first_frame(following_frame)
    show_image(frame)
    show_image(following_frame)

    points = [
        np.array([[68, 65]], dtype=np.float32),  # peg 1
        np.array([[69, 84],[111, 107]], dtype=np.float32),  # hole 1
        np.array([[70, 190]], dtype=np.float32), # peg 2
        np.array([[62, 209]], dtype=np.float32), # hole 2
    ]

    # points = config_mask.POINTS_ARRAY[pose_id]
    label = [
        np.array([1], dtype=np.int32),
        np.array([1,0], dtype=np.int32),
        np.array([1], dtype=np.int32),
        np.array([1], dtype=np.int32),
    ]

    # points_2nd_hole = np.array([
    #     np.array([[54, 219],[90, 216]], dtype=np.float32),  # hole 2 and negative point for hole 2
    # ])  # hole 2 and negative point for hole 2
    # labels_2nd_hole = np.array([1, 0], dtype=np.int32)

    # observation
    # track first object
    frame_idx, object_ids, masks = predictor_obs.add_new_points(
        frame_idx=0, obj_id=1, points=points[0], labels=label[0]
    )

    # track second object - this call returns all masks and all object ids
    frame_idx, object_ids, masks = predictor_obs.add_new_points(
        frame_idx=0, obj_id=2, points=points[1], labels=label[1]
    )

    # track third object - this call returns all masks and all object ids
    frame_idx, object_ids, masks = predictor_obs.add_new_points(
        frame_idx=0, obj_id=3, points=points[2], labels=label[2]
    )

    # track fourth object - this call returns all masks and all object ids
    frame_idx, object_ids, masks = predictor_obs.add_new_points(
        # frame_idx=0, obj_id=4, points=points[3], labels=label_and_negative
        frame_idx=0, obj_id=4, points=points[3], labels=label[3]
    )

    # next observation
    # track first object
    following_frame_idx, following_object_ids, following_masks = predictor_following_obs.add_new_points(
        frame_idx=0, obj_id=1, points=points[0], labels=label[0]
    )

    # track second object - this call returns all masks and all object ids
    following_frame_idx, following_object_ids, following_masks = predictor_following_obs.add_new_points(
        frame_idx=0, obj_id=2, points=points[1], labels=label[1]
    )

    # track third object - this call returns all masks and all object ids
    following_frame_idx, following_object_ids, following_masks = predictor_following_obs.add_new_points(
        frame_idx=0, obj_id=3, points=points[2], labels=label[2]
    )

    # track fourth object - this call returns all masks and all object ids
    following_frame_idx, following_object_ids, following_masks = predictor_following_obs.add_new_points(
        # frame_idx=0, obj_id=4, points=points[3], labels=label_and_negative
        frame_idx=0, obj_id=4, points=points[3], labels=label[3]
    )

    # overlay masks on the frame captured from the cameras
    frame_ = np.copy(frame)
    next_obs_ = draw_segmentation_color(frame_, masks, object_ids, points, kernel_size=kernel_size)
    show_image(next_obs_)
    next_obs = draw_segmentation(frame, masks, object_ids, kernel_size=kernel_size)
    observations = [next_obs]
    show_image(next_obs)

    following_frame_ = np.copy(following_frame)
    following_next_obs_ = draw_segmentation_color(following_frame_, following_masks, following_object_ids, points, kernel_size=kernel_size)
    show_image(following_next_obs_)
    following_next_obs = draw_segmentation(following_frame, following_masks, following_object_ids, kernel_size=kernel_size)
    following_observations = [following_next_obs]
    show_image(following_next_obs)

    env.insert_obs(next_obs, following_next_obs)

    # fix mask discontinuity
    print(f'hole 1 sum: {(masks[1] > 0).sum()}')
    print(f'hole 2 sum: {(masks[3] > 0).sum()}')
    hole1_mask_sum = (masks[1] > 0).sum()
    hole2_mask_sum = (masks[3] > 0).sum()
    first_hole1_sum = hole1_mask_sum
    first_hole2_sum = hole2_mask_sum

    print(f'following hole 1 sum: {(following_masks[1] > 0).sum()}')
    print(f'following hole 2 sum: {(following_masks[3] > 0).sum()}')
    following_hole1_mask_sum = (following_masks[1] > 0).sum()
    following_hole2_mask_sum = (following_masks[3] > 0).sum()
    first_following_hole1_sum = following_hole1_mask_sum
    first_following_hole2_sum = following_hole2_mask_sum

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
                obs, following_obs, _reward, done, _truncated, _info = env.step()
                if done:
                    transition_no += 1
                    frame = obs
                    object_ids, masks = predictor_obs.track(frame)
                    hole1_mask_sum = (masks[1] > 0).sum()
                    hole2_mask_sum = (masks[3] > 0).sum()
                    next_obs = draw_segmentation(frame, masks, object_ids, kernel_size=kernel_size)
                    observations.append(next_obs)

                    following_frame = following_obs
                    following_object_ids, following_masks = predictor_following_obs.track(frame)
                    following_hole1_mask_sum = (following_masks[1] > 0).sum()
                    following_hole2_mask_sum = (following_masks[3] > 0).sum()
                    following_next_obs = draw_segmentation(following_frame, following_masks, following_object_ids, kernel_size=kernel_size)
                    following_observations.append(following_next_obs)

                    env.insert_obs(next_obs, following_next_obs)
                    break

                # run inference using masks as input
                time_track = time.time()
                frame = obs
                object_ids, masks = predictor_obs.track(frame)
                next_hole1_mask_sum = (masks[1] > 0).sum()
                next_hole2_mask_sum = (masks[3] > 0).sum()
                print(f'hole 1 sum: {next_hole1_mask_sum}')
                print(f'hole 2 sum: {next_hole2_mask_sum}')
                # we assume continues transitions in episode
                if next_hole1_mask_sum - hole1_mask_sum > 0.9*first_hole1_sum and 0.1*first_hole1_sum > hole1_mask_sum:
                    masks[1] += -masks[1].max() - 1
                    next_hole1_mask_sum = (masks[1] > 0).sum()
                if next_hole2_mask_sum - hole2_mask_sum > 0.9*first_hole2_sum and 0.1*first_hole2_sum > hole2_mask_sum:
                    masks[3] += -masks[3].max() - 1
                    next_hole2_mask_sum = (masks[3] > 0).sum()
                hole1_mask_sum = next_hole1_mask_sum
                hole2_mask_sum = next_hole2_mask_sum

                following_frame = following_obs
                following_object_ids, following_masks = predictor_following_obs.track(following_frame)
                following_next_hole1_mask_sum = (following_masks[1] > 0).sum()
                following_next_hole2_mask_sum = (following_masks[3] > 0).sum()
                print(f'hole 1 sum: {following_next_hole1_mask_sum}')
                print(f'hole 2 sum: {following_next_hole2_mask_sum}')
                # we assume continues transitions in episode
                if following_next_hole1_mask_sum - following_hole1_mask_sum > 0.9*first_following_hole1_sum and 0.1*first_following_hole1_sum > following_hole1_mask_sum:
                    following_masks[1] += -following_masks[1].max() - 1
                    following_next_hole1_mask_sum = (following_masks[1] > 0).sum()
                if following_next_hole2_mask_sum - following_hole2_mask_sum > 0.9*first_following_hole2_sum and 0.8*first_following_hole2_sum > following_hole2_mask_sum:
                    following_masks[3] += -following_masks[3].max() - 1
                    following_next_hole2_mask_sum = (following_masks[3] > 0).sum()
                following_hole1_mask_sum = following_next_hole1_mask_sum
                following_hole2_mask_sum = following_next_hole2_mask_sum
                delay_tracking.append(time.time() - time_track)

                # overlay mask segments
                time_overlay = time.time()
                next_obs = draw_segmentation(frame, masks, object_ids, kernel_size=kernel_size)
                following_next_obs = draw_segmentation(following_frame, following_masks, following_object_ids, kernel_size=kernel_size)
                delay_overlay.append(time.time() - time_overlay)
                # show_image(next_obs)

                # collect observations
                observations.append(next_obs)
                following_observations.append(following_next_obs)
                env.insert_obs(next_obs, following_next_obs)

        # env.release()  # release camera - only needed when loading from a video file


        delay_tracking_ = np.array(delay_tracking) * 1000
        delay_overlay_ = np.array(delay_overlay) * 1000
        print(f"tracking delay (avg) {delay_tracking_.mean():.2f}ms\n"
              f"overlay delay (avg) {delay_overlay_.mean():.2f}ms")


    save_observations(observations, name=f"./transitions/{file_name}_masked_k{kernel_size}_obs_test.mp4")
    save_observations(following_observations, name=f"./transitions/{file_name}_masked_k{kernel_size}_next_obs_test.mp4")
    file_path = f'/home/ev/serl/examples/async_peg_insert_drq/{file_name}_masked_k{kernel_size}.pkl'
    with open(file_path, "wb") as f:
        pkl.dump(env.transitions, f)

if __name__ == "__main__":
    import os; print(os.getcwd())
    main()