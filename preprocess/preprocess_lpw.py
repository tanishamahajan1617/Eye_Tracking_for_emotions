import os
import cv2
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

DATASET_PATH = "data/lpw"
OUTPUT_IMG_DIR = "data/lpw_extracted/images"
OUTPUT_DIR = "data/lpw_extracted"

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

data = []

def extract(video_path, label_path):
    cap = cv2.VideoCapture(video_path)

    with open(label_path, "r") as f:
        labels = [line.strip().split() for line in f.readlines()]

    frame_id = 0
    saved_id = 0

    video_name = os.path.basename(video_path).replace(".avi", "")

    while True:
        ret, frame = cap.read()
        if not ret or frame_id >= len(labels):
            break

        if frame_id % 3 == 0:
            try:
                x, y = map(float, labels[frame_id])
                h, w, _ = frame.shape

                x = x / w
                y = y / h

                img_name = f"{video_name}_{saved_id}.jpg"
                img_path = os.path.join(OUTPUT_IMG_DIR, img_name)

                cv2.imwrite(img_path, frame)

                data.append([img_name, x, y, video_name])
                saved_id += 1
            except:
                pass

        frame_id += 1

    cap.release()

def process():
    for subject in os.listdir(DATASET_PATH):
        subject_path = os.path.join(DATASET_PATH, subject)

        if not os.path.isdir(subject_path):
            continue

        for file in os.listdir(subject_path):
            if file.endswith(".avi"):
                video_path = os.path.join(subject_path, file)
                label_path = video_path.replace(".avi", ".txt")

                if os.path.exists(label_path):
                    print("Processing:", file)
                    extract(video_path, label_path)

    df = pd.DataFrame(data, columns=["image", "x", "y", "video"])

    # ===== SPLIT =====
    gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_idx, temp_idx = next(gss.split(df, groups=df["video"]))

    train_df = df.iloc[train_idx]
    temp_df = df.iloc[temp_idx]

    gss2 = GroupShuffleSplit(test_size=0.5, n_splits=1, random_state=42)
    val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df["video"]))

    val_df = temp_df.iloc[val_idx]
    test_df = temp_df.iloc[test_idx]

    train_df.to_csv(f"{OUTPUT_DIR}/train.csv", index=False)
    val_df.to_csv(f"{OUTPUT_DIR}/val.csv", index=False)
    test_df.to_csv(f"{OUTPUT_DIR}/test.csv", index=False)

    print("DONE! Train, Val, Test CSVs created.")

if __name__ == "__main__":
    process()