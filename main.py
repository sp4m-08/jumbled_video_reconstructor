import time
from src.extract_frames import extract_frames
from src.compute_similarity import compute_full_ssim
from src.reconstruct_sequence import tsp_reconstruction
from src.assemble_video import assemble_video
from src.utils import clear_folder
from src.quality_check import analyze_reconstruction
#from refine_order import local_ssim_refinement
from refine_order import refine_order
def main():
    video_path = "data/input/jumbled_video.mp4"
    frames_folder = "data/output/frames"
    output_video = "data/output/reconstructed_video.mp4"
    refined_output_video = "data/output/reconstructed_video_refined.mp4" 

    clear_folder(frames_folder)

    start = time.time()
    print("[INFO] Extracting frames...")
    extract_frames(video_path, frames_folder)

    print("[INFO] Computing full NxN SSIM matrix...")
    similarity_matrix, frame_names = compute_full_ssim(frames_folder)

    print("[INFO] Reconstructing frame order using TSP...")
    order = tsp_reconstruction(similarity_matrix)
    
    # print("[INFO] Refining order using local SSIM optimization...")
    # refined_order = local_ssim_refinement(
    #     frame_order=order, 
    #     frames_folder=frames_folder,
    #     window_size=6,
    #     loops=3
    # )
    
    print("[INFO] Refining ordering using post-processing heuristic...")
    refined_order = refine_order(order, frames_folder)


    print("[INFO] Assembling refined reconstructed video...")
    assemble_video(frames_folder, refined_order, refined_output_video)


    end = time.time()
    total_time = end - start
    print(f"[INFO] Total execution time: {total_time:.2f} seconds")

    with open("logs/execution_time.txt", "w") as f:
        f.write(f"Total execution time: {total_time:.2f} seconds\n")

    # Quality check
    analyze_reconstruction(frames_folder, refined_order)
    

if __name__ == "__main__":
    main()
