import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from tqdm import tqdm

def analyze_reconstruction(frames_folder, frame_order):
    frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith(".jpg") or f.endswith(".png")])
    ordered_files = [frame_files[i] for i in frame_order]

    consecutive_ssim = []
    pixel_diffs = []

    print("[INFO] Analyzing reconstructed frames...")
    for i in tqdm(range(len(ordered_files)-1)):
        img1 = cv2.imread(os.path.join(frames_folder, ordered_files[i]), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(os.path.join(frames_folder, ordered_files[i+1]), cv2.IMREAD_GRAYSCALE)

        score = ssim(img1, img2)
        consecutive_ssim.append(score)

        diff = np.mean(np.abs(img1.astype(float) - img2.astype(float)))
        pixel_diffs.append(diff)

    avg_ssim = np.mean(consecutive_ssim)
    avg_diff = np.mean(pixel_diffs)

    print(f"[INFO] Average consecutive-frame SSIM: {avg_ssim:.4f}")
    print(f"[INFO] Average consecutive-frame pixel difference: {avg_diff:.2f}")

    # Plot SSIM
    plt.figure(figsize=(12,5))
    plt.plot(consecutive_ssim, label="Consecutive-frame SSIM")
    plt.xlabel("Frame Index")
    plt.ylabel("SSIM")
    plt.title("Consecutive-frame SSIM in Reconstructed Video")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot pixel differences
    plt.figure(figsize=(12,5))
    plt.plot(pixel_diffs, label="Consecutive-frame Pixel Difference", color="orange")
    plt.xlabel("Frame Index")
    plt.ylabel("Pixel Difference")
    plt.title("Consecutive-frame Pixel Differences in Reconstructed Video")
    plt.grid(True)
    plt.legend()
    plt.show()

    return avg_ssim, avg_diff
