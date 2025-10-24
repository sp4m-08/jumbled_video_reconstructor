import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

def compute_ssim_fast(frames_folder, resize_to=(320, 180)):
    frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith((".jpg", ".png"))])
    frames = []
    for f in frame_files:
        img = cv2.imread(os.path.join(frames_folder, f), cv2.IMREAD_GRAYSCALE)
        if resize_to:
            img = cv2.resize(img, resize_to)
        frames.append(img)

    N = len(frames)
    similarity_matrix = np.zeros((N, N))  # NxN matrix

    print(f"[INFO] Computing SSIM for {N} frames...")
    for i in tqdm(range(N-1), desc="Computing SSIM"):
        score = ssim(frames[i], frames[i+1])
        similarity_matrix[i, i+1] = score
        similarity_matrix[i+1, i] = score  # make symmetric

    return similarity_matrix, frame_files

# Usage
frames_folder = "data/output/frames"
ssim_scores = compute_ssim_fast(frames_folder)
print(f"Computed SSIM for {len(ssim_scores)} consecutive frame pairs.")




# import cv2,os
# import numpy as np
# from tqdm import tqdm
# from skimage.metrics import structural_similarity as ssim
# from multiprocessing import Pool, cpu_count

# def compute_pair(args):
#     i,j,frames_folder,frames = args
#     img1 = cv2.imread(os.path.join(frames_folder,frames[i]))
#     img2 = cv2.imread(os.path.join(frames_folder,frames[j]))
#     img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
#     img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
#     score = ssim(img1,img2)
#     return (i,j,score)

# def compute_similarity_ssim(frames_folder):
#     frames = sorted(os.listdir(frames_folder))
#     n = len(frames)
#     similarity = np.zeros((n,n))
    
#     tasks = [(i,j,frames_folder,frames)for i in range(n) for j in range(i+1,n)]
#     print(f"[INFO] Computing SSIM for {n} frames using {cpu_count()} cores...")
#     with Pool(cpu_count()) as p:
#         for i,j,score in tqdm(p.imap_unordered(compute_pair,tasks),total=len(tasks)):
#             similarity[i][j] = similarity[j][i] = score
#     return similarity, frames
