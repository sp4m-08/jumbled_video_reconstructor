import cv2, os

def assemble_video(frames_folder, frame_order, output_path, fps=30):
    frames = sorted(os.listdir(frames_folder))
    first_frame = cv2.imread(os.path.join(frames_folder, frames[0]))
    height, width, _ = first_frame.shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for idx in frame_order:
        frame = cv2.imread(os.path.join(frames_folder, frames[idx]))
        out.write(frame)

    out.release()
    print(f"[INFO] Reconstructed video saved to {output_path}")
