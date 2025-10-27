import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

def compute_full_ssim(frames_folder, resize_to=(320, 180)):
    frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith((".jpg", ".png"))])
    frames = []
    for f in frame_files:
        img = cv2.imread(os.path.join(frames_folder, f), cv2.IMREAD_GRAYSCALE)
        if resize_to:
            img = cv2.resize(img, resize_to)
        frames.append(img)

    N = len(frames)
    similarity_matrix = np.zeros((N, N))

    print(f"[INFO] Computing full NxN SSIM matrix for {N} frames...")
    for i in tqdm(range(N)):
        for j in range(i, N):
            score = ssim(frames[i], frames[j])
            similarity_matrix[i, j] = score
            similarity_matrix[j, i] = score  # symmetric

    return similarity_matrix, frame_files
