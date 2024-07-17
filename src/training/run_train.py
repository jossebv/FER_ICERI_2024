import sys
import keras
import wandb

sys.path.append(".")

import src.models.CNN as CNN
from src.data_loader.ravdess_dataloader import prepare_dataloader

ANNOT_PATH = "data/RAVDESS/annotations_frames.csv"
NUM_CLASSES = 8


def main():
    trainloader = prepare_dataloader(ANNOT_PATH, split="train", batch_size=100)
    testloader = prepare_dataloader(ANNOT_PATH, split="test", batch_size=100)

    wandb.init(project="FER_DEMOSEI_2024", entity="josebravopacheco-team", group="CNN")

    model = CNN.create_cnn(num_classes=NUM_CLASSES)

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    model.fit(
        x=trainloader,
        epochs=60,
        validation_data=testloader,
        callbacks=[wandb.keras.WandbCallback()],
    )

    wandb.finish()

    model.save("models/cnn.keras")


if __name__ == "__main__":
    main()
