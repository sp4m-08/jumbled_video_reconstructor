import cv2
import os
from skimage.metrics import structural_similarity as ssim

def compute_ssim(frameA, frameB):
    """Compute SSIM safely, return -1 if frame read failed."""
    if frameA is None or frameB is None:
        return -1
    return ssim(frameA, frameB, channel_axis=2)


def refine_order(order, frames_folder, prefix="frame_", ext=".jpg", window=4):
    """
    Takes a frame order and locally refines by checking if reversing a small
    subsequence improves SSIM score.
    """
    print("[REFINE] Loading frames...")

    # Load frames using the actual file naming format
    frames = []
    for idx in order:
        filename = f"{prefix}{idx:04d}{ext}"
        path = os.path.join(frames_folder, filename)

        frame = cv2.imread(path)
        if frame is None:
            print(f"[WARN] Frame not found: {path}")
        frames.append(frame)

    print("[REFINE] Starting refinement pass...")

    improved = True
    while improved:
        improved = False
        for i in range(len(frames) - window):
            original = frames[i:i + window]
            swapped = list(reversed(original))

            # Compute SSIM score for current sequence
            original_score = sum(compute_ssim(original[j], original[j + 1]) for j in range(window - 1))
            swapped_score = sum(compute_ssim(swapped[j], swapped[j + 1]) for j in range(window - 1))

            # If swapped is better → replace
            if swapped_score > original_score:
                frames[i:i + window] = swapped
                order[i:i + window] = order[i:i + window][::-1]
                improved = True
                print(f"[REFINE] Swap improved sequence at index {i}")

    print("[REFINE] Refinement complete ✅")
    return order
