import dataloader
import models
import augmentations
import pathlib
import torch
import helper
import trainer
from config import dataset, aug_config, model, settings, training
import datetime as dt
import os
import mlflow

torch.random.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    img_list = [str(x) for x in pathlib.Path(dataset.DATASET_PATH).rglob("*.jpg")]

    total_num_files = len(img_list)
    tvt_split = dataset.TRAIN_VAL_TEST_SPLIT

    train_list = img_list[: int(tvt_split[0] * total_num_files)][:10]
    val_list = img_list[
        int(tvt_split[0] * total_num_files) : int((tvt_split[0] + tvt_split[1]) * total_num_files)
    ][:10]
    test_list = img_list[int((tvt_split[0] + tvt_split[1]) * total_num_files) :][:10]
    sample_plot_files_list = train_list[:10]

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

    gen = models.unet(norm_layer=torch.nn.InstanceNorm2d, **model.GEN_ARGS)
    disc = models.patchGAN(norm_layer=torch.nn.InstanceNorm2d, **model.DISC_ARGS)

    optim_d = torch.optim.Adam(disc.parameters(), lr=training.DISC_LR, betas=[0.5, 0.99])
    optim_g = torch.optim.Adam(gen.parameters(), lr=training.GEN_LR, betas=[0.5, 0.99])

    model_name = f'pix2pix_{model.GEN_ARGS["num_init_filters"]}_{dt.datetime.now().strftime("%d%m%Y_%H%M%S")}'

    model_dir = os.path.join(settings.SAVE_DIRECTORY, model_name)
    config_folder_path = os.path.join(model_dir, "config_dicts")
    os.makedirs(os.path.join(config_folder_path), exist_ok=True)

    helper.save_json(model.json(), os.path.join(config_folder_path, "model.json"))
    helper.save_json(settings.json(), os.path.join(config_folder_path, "settings.json"))
    helper.save_json(training.json(), os.path.join(config_folder_path, "training.json"))
    helper.save_json(dataset.json(), os.path.join(config_folder_path, "dataset.json"))
    helper.save_json(aug_config.json(), os.path.join(config_folder_path, "aug_config.json"))

    with mlflow.start_run(run_name=model_name) as run:
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
        p2p.train(num_epochs=training.EPOCHS)

        # do this so that the test augmentations are the same each time and
        # therefore we have a consistent test set
        torch.random.manual_seed(42)
        test_results = p2p.test(test_dataloader)

        for test_metric, metric_val in test_results.items():
            mlflow.log_metric(test_metric, metric_val)

        # mlflow.log_metrics()
    # for epoch in range(num_epochs):
    #     print(f"{epoch} / {num_epochs}")
    #     running_disc_loss = 0
    #     running_gen_loss = 0
    #     running_mae_loss = 0

    #     for ip, tgt in tqdm.tqdm(train_dataloader, total=len(train_list) // bs):
    #         optim_d.zero_grad()

    #         ip = ip.to(device)
    #         tgt = tgt.to(device)

    #         real_target = torch.ones(ip.shape[0], 1, 30, 30).to(device)
    #         fake_target = torch.zeros(ip.shape[0], 1, 30, 30).to(device)

    #         generated_img = gen(ip)

    #         # print (np.max(ip.detach().numpy()), np.max(tgt.detach().numpy()), np.max(generated_img.detach().numpy()))
    #         # print (np.min(ip.detach().numpy()), np.min(tgt.detach().numpy()), np.min(generated_img.detach().numpy()))

    #         disc_inp_fake = torch.cat((ip, generated_img), 1)
    #         D_fake = disc(disc_inp_fake.detach())
    #         D_fake_loss = losses.disc_loss(D_fake, fake_target)

    #         disc_inp_real = torch.cat((ip, tgt), 1)
    #         D_real = disc(disc_inp_real)
    #         D_real_loss = losses.disc_loss(D_fake, real_target)

    #         D_total_loss = (D_fake_loss + D_real_loss) / 2
    #         disc_loss_list.append(D_total_loss)
    #         D_total_loss.backward()
    #         optim_d.step()

    #         optim_g.zero_grad()
    #         fake_gen = torch.cat((ip, generated_img), 1)
    #         G = disc(fake_gen)
    #         G_loss = losses.gen_loss(generated_img, tgt, G, real_target)

    #         gen_loss_list.append(G_loss)
    #         # msssim = pytorch_msssim.ms_ssim((tgt + 1) / 2, (generated_img + 1) / 2, data_range=1)
    #         # G_loss = G_loss + msssim
    #         G_loss.backward()

    #         optim_g.step()

    #         mean_absolute_error = torch.nn.L1Loss()

    #         running_disc_loss += D_total_loss.item()
    #         running_gen_loss += G_loss.item()
    #         running_mae_loss += mean_absolute_error(tgt.detach(), generated_img.detach())

    #         # msssim = pytorch_msssim.ms_ssim((tgt + 1) / 2, (generated_img + 1) / 2, data_range = 1)

    #     epoch_gen_loss = running_gen_loss / len(train_dataloader)
    #     epoch_disc_loss = running_disc_loss / len(train_dataloader)
    #     running_mae_loss = running_mae_loss / len(train_dataloader)

    #     print(
    #         f"gen_epoch_loss: {epoch_gen_loss} disc_epoch_loss: {epoch_disc_loss} mae_epoch_loss {running_mae_loss}"
    #     )

    #     torch.save(gen, "gen.pth")
