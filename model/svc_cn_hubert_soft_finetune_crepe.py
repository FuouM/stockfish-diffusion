from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from audio_stuff import AudioFolderDataset

_base_ = [
    "./_base_/archs/diff_svc_v2.py",
    "./_base_/trainers/base.py",
    "./_base_/schedulers/warmup_cosine_finetune.py",
    "./_base_/datasets/audio_folder.py",
]

speaker_mapping = {
    "Placeholder": 0,
}

dataset = dict(
    train=dict(
        _delete_=True,  # Delete the default train dataset
        type="ConcatDataset",
        datasets=[
            dict(
                type="AudioFolderDataset",
                path="dataset/train",
                speaker_id=speaker_mapping["Placeholder"],
            ),
        ],
        # Are there any other ways to do this?
        collate_fn=AudioFolderDataset.collate_fn,
    ),
    valid=dict(
        _delete_=True,  # Delete the default valid dataset
        type="ConcatDataset",
        datasets=[
            dict(
                type="AudioFolderDataset",
                path="dataset/valid",
                speaker_id=speaker_mapping["Placeholder"],
            ),
        ],
        collate_fn=AudioFolderDataset.collate_fn,
    ),
)

model = dict(
    speaker_encoder=dict(
        input_size=len(speaker_mapping),
    ),
    text_encoder=dict(
        type="NaiveProjectionEncoder",
        input_size=256,
        output_size=256,
    ),
)

preprocessing = dict(
    text_features_extractor=dict(
        type="ChineseHubertSoft",
        pretrained=True,
        gate_size=25,
    ),
    pitch_extractor=dict(
        type="CrepePitchExtractor",
    ),
)

# The following trainer val and save checkpoints every 1000 steps
trainer = dict(
    val_check_interval=1000,
    callbacks=[
        ModelCheckpoint(
            filename="{epoch}-{step}-{valid_loss:.2f}",
            every_n_train_steps=5000,
            save_top_k=-1,
        ),
        LearningRateMonitor(logging_interval="step"),
    ],
)
