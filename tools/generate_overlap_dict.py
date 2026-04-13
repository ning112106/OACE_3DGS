import os
import json
import argparse
import collections
from itertools import combinations

'''
colmap model_converter \
    --input_path /data/qjn/gaussian-splatting_loss/datasets/on-the-go/arcdetriomphe/sparse/0 \
    --output_path /data/qjn/gaussian-splatting_loss/datasets/on-the-go/arcdetriomphe/sparse/0_txt \
    --output_type TXT

colmap model_converter \
    --input_path /data/qjn/gaussian-splatting_loss/datasets/trevi-fountain/sparse/0 \
    --output_path /data/qjn/gaussian-splatting_loss/datasets/trevi-fountain/sparse/0/points3D.ply \
    --output_type PLY

python generate_overlap_dict_full.py \
    --colmap_path /data/qjn/gaussian-splatting_loss/datasets/trevi-fountain/sparse/0_txt \
    --output /data/qjn/gaussian-splatting_loss/datasets/trevi-fountain/sparse/overlap_dict.json \
    --min_common 50 \
    --min_overlap_ratio 0.2
'''
def read_images_txt(path):

    image_id_to_name = {}

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):

        line = lines[i].strip()

        if line.startswith("#") or len(line) == 0:
            i += 1
            continue

        parts = line.split()

        if len(parts) < 9:
            i += 1
            continue

        image_id = int(parts[0])
        image_name = parts[-1]

        image_id_to_name[image_id] = image_name

        i += 2

    return image_id_to_name


def read_points3D_txt(path):

    point3D_observations = []
    image_point_count = collections.defaultdict(int)

    with open(path, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if line.startswith("#") or len(line) == 0:
                continue

            parts = line.split()

            if len(parts) < 8:
                continue

            track_data = parts[8:]

            image_ids = set()

            for i in range(0, len(track_data), 2):

                image_id = int(track_data[i])

                image_ids.add(image_id)

                image_point_count[image_id] += 1

            point3D_observations.append(image_ids)

    return point3D_observations, image_point_count


def build_overlap_dict(
        images_txt,
        points3D_txt,
        min_common=50,
        min_overlap_ratio=0.1):

    image_id_to_name = read_images_txt(images_txt)

    point3D_obs, image_point_count = read_points3D_txt(points3D_txt)

    overlap = collections.defaultdict(int)

    # 统计 pair common points
    for view_ids in point3D_obs:

        for id1, id2 in combinations(view_ids, 2):

            name1 = image_id_to_name.get(id1)
            name2 = image_id_to_name.get(id2)

            if name1 and name2:

                key = tuple(sorted((name1, name2)))

                overlap[key] += 1

    neighbor_dict = collections.defaultdict(list)

    for (img1, img2), common in overlap.items():

        id1 = None
        id2 = None

        # 反查 image id
        for k, v in image_id_to_name.items():
            if v == img1:
                id1 = k
            if v == img2:
                id2 = k

        if id1 is None or id2 is None:
            continue

        n1 = image_point_count[id1]
        n2 = image_point_count[id2]

        min_points = min(n1, n2)

        if min_points == 0:
            continue

        overlap_ratio = common / min_points

        #双条件筛选
        if (
            common >= min_common and
            overlap_ratio >= min_overlap_ratio
        ):

            neighbor_dict[img1].append(
                [img2, common, overlap_ratio]
            )

            neighbor_dict[img2].append(
                [img1, common, overlap_ratio]
            )

    return neighbor_dict


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate overlap_dict.json with ratio filtering."
    )

    parser.add_argument(
        "--colmap_path",
        type=str,
        required=True
    )

    parser.add_argument(
        "--min_common",
        type=int,
        default=50
    )

    parser.add_argument(
        "--min_overlap_ratio",
        type=float,
        default=0.2
    )

    parser.add_argument(
        "--output",
        type=str,
        default="overlap_dict.json"
    )

    args = parser.parse_args()

    images_txt = os.path.join(
        args.colmap_path,
        "images.txt"
    )

    points3D_txt = os.path.join(
        args.colmap_path,
        "points3D.txt"
    )

    print("Reading COLMAP TXT files...")

    overlap_dict = build_overlap_dict(
        images_txt,
        points3D_txt,
        min_common=args.min_common,
        min_overlap_ratio=args.min_overlap_ratio
    )

    with open(args.output, "w", encoding="utf-8") as f:

        json.dump(
            overlap_dict,
            f,
            indent=2
        )

    print(
        f"Saved overlap_dict.json "
        f"({len(overlap_dict)} images)"
    )