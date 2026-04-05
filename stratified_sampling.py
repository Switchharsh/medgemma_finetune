import pandas as pd
import random
from pathlib import Path


def create_stratified_ctrate_split(
    labels_csv_path,
    reports_csv_path,
    output_txt_path,
    target_size=1000,
):
    """Create stratified dataset split using labels from one CSV and matching VolumeName in reports CSV."""
    random.seed(42)

    labels_df = pd.read_csv(labels_csv_path)
    print(f"Loaded labels CSV: {len(labels_df)} rows")

    reports_df = pd.read_csv(reports_csv_path)
    print(f"Loaded reports CSV: {len(reports_df)} rows")

    available_volumes = set(labels_df['VolumeName']) & set(reports_df['VolumeName'])
    print(f"Volumes in both files: {len(available_volumes)}")

    labels_df = labels_df[labels_df['VolumeName'].isin(available_volumes)].copy()
    print(f"Using {len(labels_df)} volumes with both labels and reports")

    label_columns = [col for col in labels_df.columns if col != 'VolumeName']

    normal_mask = labels_df[label_columns].sum(axis=1) == 0
    normal_volumes = labels_df[normal_mask]['VolumeName'].tolist()
    print(f"Normal volumes: {len(normal_volumes)}")

    abnormal_buckets = {}
    for col in label_columns:
        abnormal_volumes = labels_df[labels_df[col] == 1]['VolumeName'].tolist()
        abnormal_buckets[col] = abnormal_volumes

    target_normal = int(target_size * 0.30)
    target_per_abnormality = int((target_size - target_normal) / len(label_columns))

    selected_volumes = set()

    sampled_normal = random.sample(normal_volumes, min(target_normal, len(normal_volumes)))
    selected_volumes.update(sampled_normal)
    print(f"Sampled {len(sampled_normal)} normal volumes")

    for col in label_columns:
        available = [vol for vol in abnormal_buckets[col] if vol not in selected_volumes]
        sampled = random.sample(available, min(target_per_abnormality, len(available)))
        selected_volumes.update(sampled)

    if len(selected_volumes) < target_size:
        all_volumes = set(labels_df['VolumeName'].tolist())
        remaining = list(all_volumes - selected_volumes)
        needed = target_size - len(selected_volumes)
        if remaining:
            selected_volumes.update(random.sample(remaining, min(needed, len(remaining))))

    output_path = Path(output_txt_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for vol_name in list(selected_volumes)[:target_size]:
            f.write(f"{vol_name}\n")

    print(f"\nSaved {len(list(selected_volumes)[:target_size])} volumes to: {output_path}")


if __name__ == "__main__":
    create_stratified_ctrate_split(
        labels_csv_path='/vol/idea_longterm/datasets/CT-RATE/dataset/multi_abnormality_labels/train_predicted_labels.csv',
        reports_csv_path='/vol/idea_longterm/datasets/CT-RATE/dataset/radiology_text_reports/train_reports.csv',
        output_txt_path='/vol/ideadata/ac54awik/Medgemma_finetune/data/ctrate_train_1k.txt',
        target_size=1000,
    )
