import pandas as pd
import random


def create_stratified_ctrate_split(labels_csv_path, output_txt_path, target_size=1000):
    """Create stratified dataset split with ~30% normal, ~70% abnormal samples."""
    random.seed(42)

    df = pd.read_csv(labels_csv_path)

    label_columns = [
        'Medical material', 'Arterial wall calcification', 'Cardiomegaly',
        'Pericardial effusion', 'Coronary artery wall calcification',
        'Hiatal hernia', 'Lymphadenopathy', 'Emphysema', 'Atelectasis',
        'Lung nodule', 'Lung opacity', 'Pulmonary fibrotic sequela',
        'Pleural effusion', 'Mosaic attenuation pattern', 'Peribronchial thickening',
        'Consolidation', 'Bronchiectasis', 'Interlobular septal thickening'
    ]

    # Normal bucket (all 18 labels are 0)
    normal_mask = df[label_columns].sum(axis=1) == 0
    normal_ids = df[normal_mask]['AccessionNo'].tolist()

    # Buckets for each abnormality
    abnormal_buckets = {}
    for col in label_columns:
        abnormal_buckets[col] = df[df[col] == 1]['AccessionNo'].tolist()

    # Sample: ~30% Normal, ~70% Abnormal
    target_normal = int(target_size * 0.30)
    target_per_abnormality = int((target_size - target_normal) / len(label_columns))

    selected_ids = set()

    # Sample normals
    sampled_normal = random.sample(normal_ids, min(target_normal, len(normal_ids)))
    selected_ids.update(sampled_normal)

    # Sample abnormalities
    for col in label_columns:
        available_ids = [acc for acc in abnormal_buckets[col] if acc not in selected_ids]
        sampled_abnormal = random.sample(available_ids, min(target_per_abnormality, len(available_ids)))
        selected_ids.update(sampled_abnormal)

    # Fill randomly if under target_size
    if len(selected_ids) < target_size:
        all_ids = set(df['AccessionNo'].tolist())
        remaining_ids = list(all_ids - selected_ids)
        needed = target_size - len(selected_ids)
        selected_ids.update(random.sample(remaining_ids, min(needed, len(remaining_ids))))

    # Save to file
    with open(output_txt_path, 'w') as f:
        for acc_id in list(selected_ids)[:target_size]:
            f.write(f"{acc_id}\n")

    print(f"Created stratified dataset with {len(selected_ids)} unique accessions.")
    print(f"Normal scans included: {len(sampled_normal)}")


if __name__ == "__main__":
    create_stratified_ctrate_split(
        labels_csv_path='path_to/train.csv',
        output_txt_path='ctrate_train_1k.txt'
    )
