import json
import os


def angle_based_split(total=360, train_per_sector=40, val_per_sector=10):
    sectors = 6
    sector_size = total // sectors
    train_idx = []
    val_idx = []
    used = set()

    for i in range(sectors):
        start = i * sector_size
        step_train = sector_size / train_per_sector
        step_val = sector_size / val_per_sector
        train_candidates = [
            int(start + j * step_train) for j in range(train_per_sector)
        ]
        val_candidates = [
            int(start + train_per_sector + j * step_val) for j in range(val_per_sector)
        ]
        # Sadece 0-359 aralığındaki indeksleri ekle
        train_idx.extend([idx for idx in train_candidates if 0 <= idx < total])
        val_idx.extend([idx for idx in val_candidates if 0 <= idx < total])
        used.update(train_idx + val_idx)

    all_idx = set(range(total))
    test_idx = sorted(list(all_idx - used))
    return sorted(train_idx), sorted(val_idx), test_idx


def generate_split_json(data_dir="dataset", label_json="data.json", output_dir="."):
    with open(label_json, "r") as f:
        label_map = json.load(f)

    train_list, val_list, test_list = [], [], []

    for video_id, model_name in label_map.items():
        video_path = os.path.join(data_dir, video_id)
        if not os.path.isdir(video_path):
            print(f"Skipping missing directory: {video_path}")
            continue

        # 360 görüntü varsayımı
        frame_paths = [os.path.join(video_path, f"{i:04d}.jpg") for i in range(360)]

        train_idx, val_idx, test_idx = angle_based_split()

        for idx in train_idx:
            train_list.append(
                {"image_path": frame_paths[idx], "model_name": model_name}
            )
        for idx in val_idx:
            val_list.append({"image_path": frame_paths[idx], "model_name": model_name})
        for idx in test_idx:
            test_list.append({"image_path": frame_paths[idx], "model_name": model_name})

    # JSON çıktısı
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "train.json"), "w") as f:
        json.dump(train_list, f, indent=2)
    with open(os.path.join(output_dir, "val.json"), "w") as f:
        json.dump(val_list, f, indent=2)
    with open(os.path.join(output_dir, "test.json"), "w") as f:
        json.dump(test_list, f, indent=2)

    print(f"✅ train.json: {len(train_list)} örnek")
    print(f"✅ val.json: {len(val_list)} örnek")
    print(f"✅ test.json: {len(test_list)} örnek")


if __name__ == "__main__":
    generate_split_json(data_dir="dataset", label_json="data.json", output_dir=".")
