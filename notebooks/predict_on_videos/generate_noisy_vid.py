import cv2
import torch
from pix2pix import augmentations

torch.random.manual_seed(42)

if __name__ == "__main__":
    vid_path = "videos/input_video_gt.mp4"

    cap = cv2.VideoCapture(vid_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("videos/input_video_noisy.mp4", fourcc, fps, (width, height))

    print("generating noisy video....")
    while cap.isOpened():
        ret, frame = cap.read()

        if ret is False:
            break

        frame = torch.Tensor(frame.astype("float32"))
        frame = (frame / (255 / 2)) - 1
        frame = augmentations.gaussian_noise(frame, std_div_denum=2)
        frame = (frame.numpy() + 1) / 2
        frame = (frame * 255).astype("uint8")

        out.write(frame)

    cap.release()
    out.release()

    print("saved video....")
