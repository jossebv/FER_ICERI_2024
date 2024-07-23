import pandas as pd
import os
from tqdm import tqdm

DATA_DIR = "/mnt/RESOURCES/josemanuelbravo/FER_2024/data/FER-2013"
IMAGES_DIR = os.path.join(DATA_DIR, "raw")


def main():
    for subset in tqdm(os.listdir(IMAGES_DIR), desc="Iterating subsets"):
        annotations = pd.DataFrame([], columns=["path", "emotion"])
        for emotion in tqdm(
            os.listdir(os.path.join(IMAGES_DIR, subset)),
            desc=f"Iterating emotions of subset {subset}",
        ):
            if not os.path.isdir(os.path.join(IMAGES_DIR, subset, emotion)):
                continue

            for image in os.listdir(os.path.join(IMAGES_DIR, subset, emotion)):
                image_annotation = pd.Series(
                    {
                        "path": os.path.join(IMAGES_DIR, subset, emotion, image),
                        "emotion": emotion,
                    }
                )
                annotations = pd.concat(
                    (annotations, image_annotation.to_frame().T), ignore_index=True
                )

        annotations.to_csv(
            os.path.join(DATA_DIR, f"{subset}_annotations.csv"), index=False
        )


if __name__ == "__main__":
    main()
