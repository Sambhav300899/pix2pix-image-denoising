from pix2pix import dataloader
from pix2pix import models
from pix2pix import augmentations
import pathlib
import torch
from pix2pix import helper
from pix2pix import trainer
from pix2pix.config import dataset, aug_config, model, settings, training
import datetime as dt
import os
import mlflow
import logging

torch.random.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("starting training...")
    img_list = [str(x) for x in pathlib.Path(dataset.DATASET_PATH).rglob("*.jpg")]

    total_num_files = len(img_list)
    tvt_split = dataset.TRAIN_VAL_TEST_SPLIT

    train_list = img_list[: int(tvt_split[0] * total_num_files)]
    val_list = img_list[
        int(tvt_split[0] * total_num_files) : int((tvt_split[0] + tvt_split[1]) * total_num_files)
    ]
    test_list = img_list[int((tvt_split[0] + tvt_split[1]) * total_num_files) :]
    sample_plot_files_list = test_list[:10]

    logger.info(f"Number of train samples: {len(train_list)}")
    logger.info(f"Number of val samples: {len(val_list)}")
    logger.info(f"Number of test samples: {len(test_list)}")

    logger.info("loading dataloaders...")
    train_gen = dataloader.lfw_dataset(
        img_paths=train_list,
        transforms=augmentations.augs(**aug_config.TRAIN_AUGS),
    )
    val_gen = dataloader.lfw_dataset(
        img_paths=val_list,
        transforms=augmentations.augs(**aug_config.VAL_AUGS),
    )
    test_gen = dataloader.lfw_dataset(
        img_paths=test_list,
        transforms=augmentations.augs(**aug_config.TEST_AUGS),
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_gen, batch_size=dataset.BATCH_SIZE, shuffle=dataset.SHUFFLE_TRAIN
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_gen, batch_size=dataset.BATCH_SIZE, shuffle=dataset.SHUFFLE_VAL
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_gen, batch_size=dataset.BATCH_SIZE, shuffle=dataset.SHUFFLE_TEST
    )

    logger.info("loading models...")
    gen = models.unet(norm_layer=torch.nn.InstanceNorm2d, **model.GEN_ARGS)
    helper.init_weights(gen)
    gen = torch.nn.DataParallel(gen)
    disc = models.patchGAN(norm_layer=torch.nn.InstanceNorm2d, **model.DISC_ARGS)

    logger.info("loading optimizers...")
    optim_d = torch.optim.Adam(disc.parameters(), lr=training.DISC_LR, betas=[0.5, 0.99])
    optim_g = torch.optim.Adam(gen.parameters(), lr=training.GEN_LR, betas=[0.5, 0.99])

    model_name = f'pix2pix_{model.GEN_ARGS["num_init_filters"]}{"_dropout_" if model.GEN_ARGS["use_dropout"] else "_"}{dt.datetime.now().strftime("%d%m%Y_%H%M%S")}'

    model_dir = os.path.join(settings.SAVE_DIRECTORY, model_name)
    config_folder_path = os.path.join(model_dir, "config_dicts")
    os.makedirs(os.path.join(config_folder_path), exist_ok=True)
    logger.info(f"model artifacts will be saved to {model_dir}")

    helper.save_json(model.json(), os.path.join(config_folder_path, "model.json"))
    helper.save_json(settings.json(), os.path.join(config_folder_path, "settings.json"))
    helper.save_json(training.json(), os.path.join(config_folder_path, "training.json"))
    helper.save_json(dataset.json(), os.path.join(config_folder_path, "dataset.json"))
    helper.save_json(aug_config.json(), os.path.join(config_folder_path, "aug_config.json"))

    with mlflow.start_run(run_name=model_name) as run:
        logger.info("begin logging of mlflow artifacts...")
        mlflow.log_param("num_train_samples", len(train_dataloader) * dataset.BATCH_SIZE)
        mlflow.log_param("num_val_samples", len(val_dataloader) * dataset.BATCH_SIZE)
        mlflow.log_param("num_test_samples", len(test_dataloader) * dataset.BATCH_SIZE)
        mlflow.log_param("batch_size", dataset.BATCH_SIZE)
        mlflow.log_param("total_gen_params", sum(p.numel() for p in gen.parameters()))
        mlflow.log_param("total_disc_params", sum(p.numel() for p in disc.parameters()))

        mlflow.log_dict(model.dict(), "model.json")
        mlflow.log_dict(settings.dict(), "settings.json")
        mlflow.log_dict(training.dict(), "training.json")
        mlflow.log_dict(dataset.dict(), "dataset.json")
        mlflow.log_dict(aug_config.dict(), "aug_config.json")

        logger.info("initialialising trainer...")
        p2p = trainer.pix2pix_trainer(
            gen=gen,
            disc=disc,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            optim_d=optim_d,
            optim_g=optim_g,
            model_save_dir=model_dir,
            device=settings.DEVICE,
            sample_img_list=sample_plot_files_list,
            model_name=model_name,
            save_checkpoints=training.SAVE_CHECKPOINTS,
            log_mlflow_metrics=True,
        )
        logger.info(f"training model for {training.EPOCHS} epochs...")
        p2p.train(num_epochs=training.EPOCHS)

        # do this so that the test augmentations are the same each time and
        # therefore we have a consistent test set
        logger.info("generating test result for trained model...")
        torch.random.manual_seed(42)
        test_results = p2p.test(test_dataloader)

        for test_metric, metric_val in test_results.items():
            mlflow.log_metric(test_metric, metric_val)

        mlflow.end_run()
