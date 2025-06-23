import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import subprocess as sp
import cv2


def get_fps(filename):
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate",
        "-of",
        "csv=p=0",
        filename,
    ]
    with sp.Popen(command, stdout=sp.PIPE, bufsize=-1) as pipe:
        for line in pipe.stdout:
            a, b = line.decode().strip().split("/")
            return int(a) / int(b)


def video_frame_generator(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Failed to open video: {video_path}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield frame
    cap.release()


def downsample_tensor(X, factor):
    length = X.shape[0] // factor * factor
    return np.mean(X[:length].reshape(-1, factor, *X.shape[1:]), axis=1)


def render_animation(
    keypoints,
    keypoints_metadata,
    poses,
    skeleton,
    fps,
    bitrate,
    azim,
    output,
    viewport,
    limit=-1,
    downsample=1,
    size=6,
    input_video_path=None,
    input_video_skip=0,
):

    plt.ioff()
    fig = plt.figure(figsize=(size * (1 + len(poses)), size))
    canvas = FigureCanvas(fig)
    ax_in = fig.add_subplot(1, 1 + len(poses), 1)
    ax_in.set_axis_off()
    ax_in.set_title("Input")

    ax_3d = []
    lines_3d = []
    trajectories = []
    radius = 1.7
    for index, (title, data) in enumerate(poses.items()):
        ax = fig.add_subplot(1, 1 + len(poses), index + 2, projection="3d")
        ax.view_init(elev=15.0, azim=azim)
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_zlim3d([0, radius])
        ax.set_ylim3d([-radius / 2, radius / 2])
        try:
            ax.set_aspect("equal")
        except NotImplementedError:
            ax.set_aspect("auto")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.dist = 7.5
        ax.set_title(title)
        ax_3d.append(ax)
        lines_3d.append([])
        trajectories.append(data[:, 0, [0, 1]])
    poses = list(poses.values())

    if input_video_path is None:
        all_frames = np.zeros(
            (keypoints.shape[0], viewport[1], viewport[0], 3), dtype="uint8"
        )
    else:
        all_frames = [f for f in video_frame_generator(input_video_path)]
        effective_length = min(keypoints.shape[0], len(all_frames))
        all_frames = all_frames[:effective_length]
        keypoints = keypoints[input_video_skip:]
        for idx in range(len(poses)):
            poses[idx] = poses[idx][input_video_skip:]
        if fps is None:
            fps = get_fps(input_video_path)

    if downsample > 1:
        keypoints = downsample_tensor(keypoints, downsample)
        all_frames = downsample_tensor(np.array(all_frames), downsample).astype("uint8")
        for idx in range(len(poses)):
            poses[idx] = downsample_tensor(poses[idx], downsample)
            trajectories[idx] = downsample_tensor(trajectories[idx], downsample)
        fps /= downsample

    initialized = False
    image = None
    lines = []
    points = None

    if limit < 1:
        limit = len(all_frames)
    else:
        limit = min(limit, len(all_frames))

    parents = skeleton.parents()

    def update_video(i):
        nonlocal initialized, image, lines, points

        for n, ax in enumerate(ax_3d):
            ax.set_xlim3d(
                [
                    -radius / 2 + trajectories[n][i, 0],
                    radius / 2 + trajectories[n][i, 0],
                ]
            )
            ax.set_ylim3d(
                [
                    -radius / 2 + trajectories[n][i, 1],
                    radius / 2 + trajectories[n][i, 1],
                ]
            )

        joints_right_2d = keypoints_metadata["keypoints_symmetry"][1]
        colors_2d = np.full(keypoints.shape[1], "black")
        colors_2d[joints_right_2d] = "red"
        if not initialized:
            image = ax_in.imshow(all_frames[i], aspect="equal")
            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue
                if (
                    len(parents) == keypoints.shape[1]
                    and keypoints_metadata["layout_name"] != "coco"
                ):
                    lines.append(
                        ax_in.plot(
                            [keypoints[i, j, 0], keypoints[i, j_parent, 0]],
                            [keypoints[i, j, 1], keypoints[i, j_parent, 1]],
                            color="pink",
                        )
                    )
                col = "red" if j in skeleton.joints_right() else "black"
                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]
                    lines_3d[n].append(
                        ax.plot(
                            [pos[j, 0], pos[j_parent, 0]],
                            [pos[j, 1], pos[j_parent, 1]],
                            [pos[j, 2], pos[j_parent, 2]],
                            zdir="z",
                            c=col,
                        )
                    )
            points = ax_in.scatter(
                *keypoints[i].T, 10, color=colors_2d, edgecolors="white", zorder=10
            )
            initialized = True
        else:
            image.set_data(all_frames[i])
            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue
                if (
                    len(parents) == keypoints.shape[1]
                    and keypoints_metadata["layout_name"] != "coco"
                ):
                    lines[j - 1][0].set_data(
                        [keypoints[i, j, 0], keypoints[i, j_parent, 0]],
                        [keypoints[i, j, 1], keypoints[i, j_parent, 1]],
                    )
                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]
                    lines_3d[n][j - 1][0].set_xdata(
                        np.array([pos[j, 0], pos[j_parent, 0]])
                    )
                    lines_3d[n][j - 1][0].set_ydata(
                        np.array([pos[j, 1], pos[j_parent, 1]])
                    )
                    lines_3d[n][j - 1][0].set_3d_properties(
                        np.array([pos[j, 2], pos[j_parent, 2]]), zdir="z"
                    )
            points.set_offsets(keypoints[i])
        print(f"{i+1}/{limit}", end="\r")

    width = int(fig.get_figwidth() * fig.dpi)
    height = int(fig.get_figheight() * fig.dpi)
    output = str(output)

    if not output.endswith(".mp4"):
        raise ValueError("Only .mp4 output supported with OpenCV writer")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output, fourcc, fps, (width, height))

    for i in range(limit):
        update_video(i)
        canvas.draw()
        buf = canvas.buffer_rgba()
        frame = np.asarray(buf).reshape((height, width, 4))[..., :3]  # drop alpha
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame)

    video_writer.release()
    plt.close()
