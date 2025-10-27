import time
from src.extract_frames import extract_frames
from src.compute_similarity import compute_full_ssim
from src.reconstruct_sequence import tsp_reconstruction
from src.assemble_video import assemble_video
from src.utils import clear_folder
from src.quality_check import analyze_reconstruction

def main():
    video_path = "data/input/jumbled_video.mp4"
    frames_folder = "data/output/frames"
    output_video = "data/output/reconstructed_video.mp4"

    clear_folder(frames_folder)

    start = time.time()
    print("[INFO] Extracting frames...")
    extract_frames(video_path, frames_folder)

    print("[INFO] Computing full NxN SSIM matrix...")
    similarity_matrix, frame_names = compute_full_ssim(frames_folder)

    print("[INFO] Reconstructing frame order using TSP...")
    order = tsp_reconstruction(similarity_matrix)

    print("[INFO] Assembling reconstructed video...")
    assemble_video(frames_folder, order, output_video)

    end = time.time()
    total_time = end - start
    print(f"[INFO] Total execution time: {total_time:.2f} seconds")

    with open("logs/execution_time.txt", "w") as f:
        f.write(f"Total execution time: {total_time:.2f} seconds\n")

    # Quality check
    analyze_reconstruction(frames_folder, order)

if __name__ == "__main__":
    main()
