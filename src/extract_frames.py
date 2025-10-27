import cv2,os
def extract_frames(video_path,output_folder):
    os.makedirs(output_folder,exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0
    
    while True:
        ret,frame = cap.read()
        if not ret:
            break
        cv2.imwrite(f"{output_folder}/frame_{count:04d}.jpg",frame)
        count = count+1

    cap.release()
    print(f"[INFO] Extracted {count} frames to {output_folder}")
    
