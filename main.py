import time
from src.extract_frames import extract_frames
from src.compute_similarity import compute_ssim_fast
from src.reconstruct_sequence import greedy_reconstruction
from src.reconstruct_sequence import tsp_reconstruction
from src.assemble_video import assemble_video
from src.utils import clear_folder

def main():
    video_path = "data/input/jumbled_video.mp4"
    frames_folder = "data/output/frames"
    output_video = "data/output/reconstructed_video.mp4"

    clear_folder(frames_folder)

    start = time.time()
    extract_frames(video_path, frames_folder)
    similarity_matrix, frame_names = compute_ssim_fast(frames_folder)
    order = tsp_reconstruction(similarity_matrix)
    assemble_video(frames_folder, order, output_video)
    end = time.time()

    total_time = end - start
    print(f"[INFO] Total execution time: {total_time:.2f} seconds")
    with open("logs/execution_time.txt", "w") as f:
        f.write(f"Total execution time: {total_time:.2f} seconds\n")

if __name__ == "__main__":
    main()
