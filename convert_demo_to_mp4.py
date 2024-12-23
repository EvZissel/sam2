import torch
import cv2
import numpy as np
import pickle as pkl


import matplotlib.pyplot as plt


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


def main():

    # upload damos
    for file in os.listdir("/home/ev/serl/examples/async_peg_insert_drq"):
        file_start = 'peg_insert_20_demos_20_trials_pose_id_2'
        if file.startswith(file_start) and not (file.endswith('_masked_k9.pkl') or file.endswith('_masked_color.pkl')):

            demo_path = f'/home/ev/serl/examples/async_peg_insert_drq/{file}'
            with open(demo_path, "rb") as f:
                demo = pkl.load(f)

            steps = 0
            obs_1 = demo[steps]['observations']['wrist_1']
            obs_2 = demo[steps]['observations']['wrist_2']
            obs = np.squeeze(np.concatenate((obs_1, obs_2), axis=1))

            observations = [obs]
            show_image(obs)

            transition_no = 0

            while steps < len(demo):
                while True:
                    print(f"step {len(observations)}")
                    obs_1 = demo[steps]['next_observations']['wrist_1']
                    obs_2 = demo[steps]['next_observations']['wrist_2']
                    obs = np.squeeze(np.concatenate((obs_1, obs_2), axis=1))

                    done = demo[steps]['dones']
                    steps += 1
                    if done:
                        transition_no += 1
                        break

                    observations.append(obs)


            save_observations(observations, name=f"./transitions/{file[:-4]}_next_observations.mp4")

if __name__ == "__main__":
    import os; print(os.getcwd())
    main()