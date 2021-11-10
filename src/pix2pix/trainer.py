import losses
import tqdm
import torch
import os
import helper
import matplotlib.pyplot as plt
import dataloader
import augmentations
import pytorch_msssim
import mlflow
import logging
from typing import List

logger = logging.getLogger(__name__)

plt.ioff()


class pix2pix_trainer:
    """
    trainer class, this handles plotting, saving metrics, plotting sample imgs etc.

    gen (py:obj:`torch.nn.Module`): generator model
    disc (py:obj:`torch.nn.Module`): discriminator model
    train_dataloader(py:obj:`torch.utils.data.DataLoader`): train dataloader
    val_dataloader(py:obj:`torch.utils.data.DataLoader`): val dataloader
    optim_d (py:obj:`torch.optim.Adam`): optimizer for discriminator
    optim_g (py:obj:`torch.optim.Adam`): optimizer for generator
    model_name (str): model name
    log_mlflow_metrics (bool): flag to log mlfow metrics
    model_save_dir (str): directory to save the model
    device (str): device for computation
    history (dict): history data if loading old checkpoint
    save_checkpoints (bool): flag to save checkpoints
    sample_img_list(List[str]): list of images to be used for sample plots
    img_size (List[int]): img_size
    """

    def __init__(
        self,
        gen: torch.nn.Module,
        disc: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        optim_d: torch.optim.Adam,
        optim_g: torch.optim.Adam,
        model_name: str,
        log_mlflow_metrics: bool,
        model_save_dir: str = "./generated",
        device: str = "cpu",
        history: dict = None,
        save_checkpoints: bool = False,
        sample_img_list: List[str] = None,
        img_size: List[int] = (256, 256),
    ) -> None:
        self.gen = gen.to(device)
        self.disc = disc.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.model_save_dir = model_save_dir
        self.device = device
        self.optim_g = optim_g
        self.optim_d = optim_d
        self.save_checkpoints = save_checkpoints
        self.sample_img_list = sample_img_list
        self.img_size = img_size
        self.model_name = model_name
        self.log_mlflow_metrics = log_mlflow_metrics

        logger.debug(f"got {len(self.sample_img_list)} imgs to plot per epoch")

        if sample_img_list:
            self.noise_vector = torch.zeros(([len(sample_img_list), 3, self.img_size[0], self.img_size[1]]))

            for i in range(len(sample_img_list)):
                std = torch.rand(1)
                mean = 0

                self.noise_vector[i, :, :, :] = (
                    torch.randn([3, self.img_size[0], self.img_size[1]]) * std + mean
                )

        if history:
            self.history = history
        else:
            self.history = self._reset_history()

        self._create_dirs()

    def _create_dirs(self) -> None:
        """
        method to create directories to save data
        """
        logger.debug("creating directories to store data")
        self.checkpoint_dir = os.path.join(self.model_save_dir, "checkpoints")
        self.plot_dir = os.path.join(self.model_save_dir, "plots")
        self.sample_plots_dir = os.path.join(self.model_save_dir, "sample_img_plots")

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(self.sample_plots_dir, exist_ok=True)

    def _reset_history(self) -> dict:
        """
        method to initialise metrics dictionary

        Returns:
        (dict): metric dict
        """
        logger.debug("reset history dict")

        return {
            "gen_loss": [],
            "disc_loss": [],
            "mae": [],
            "ssim": [],
            # "psnr": [],
            "val_gen_loss": [],
            "val_mae": [],
            "val_ssim": [],
            # "val_psnr": [],
        }

    def _log_mlflow_metrics_for_epoch(self, epoch: int) -> None:
        """
        method to log mlflow metrics

        Args:
            epoch (int): epoch number
        """
        for metric_name, metric_value in self.history.items():
            mlflow.log_metric(metric_name, metric_value[-1], step=epoch)

    def train(self, num_epochs: int) -> None:
        """
        train the model for specified number of epochs, log metrics, save checkpoints and save all training + model assets

        Args:
            num_epochs (int): number of epochs to train the model
        """
        for epoch in range(1, num_epochs + 1):
            print(f"Epoch {epoch}/{num_epochs}")
            gen_loss, disc_loss, mae_loss, ssim = self._train_step()
            val_gen_loss, val_mae, val_ssim = self._test_step(self.val_dataloader)

            self.history["gen_loss"].append(gen_loss)
            self.history["disc_loss"].append(disc_loss)
            self.history["mae"].append(mae_loss)
            self.history["ssim"].append(ssim)
            # self.history["psnr"].append(psnr)

            self.history["val_gen_loss"].append(val_gen_loss)
            self.history["val_mae"].append(val_mae)
            self.history["val_ssim"].append(val_ssim)
            # self.history["val_psnr"].append(val_psnr)

            print(f"val_gen_loss: {val_gen_loss:.03f} - val_mae: {val_mae:.03f}")
            print()

            if self.save_checkpoints:
                logger.debug(f"saving checkpoint for epoch {epoch}")
                self._save_checkpoint(epoch)

            if self.sample_img_list:
                logger.debug(f"saving sample img plots for epoch {epoch}")
                self._plot_sample_imgs(epoch)

            if self.log_mlflow_metrics:
                logger.debug(f"logging mlflow metrics for epoch {epoch}")
                self._log_mlflow_metrics_for_epoch(epoch)

        self._plot()
        gen_save_path = os.path.join(self.model_save_dir, "gen_" + self.model_name + ".pth")
        disc_save_path = os.path.join(self.model_save_dir, "disc_" + self.model_name + ".pth")

        logger.info(f"saving final generator model to {gen_save_path}")
        torch.save(self.gen, gen_save_path)

        logger.info(f"saving final discriminator model to {disc_save_path}")
        torch.save(self.disc, disc_save_path)

        logger.info(f"saving model history to {self.model_save_dir}")
        helper.save_json(self.history, os.path.join(self.model_save_dir, "history.json"))

    def test(self, dataloader: torch.utils.data.DataLoader) -> dict:
        """
        test the model for specified dataloader

        Args:
            dataloader (py:obj:`torch.utils.data.DataLoader`): dataloader to make predictions on
        Returns:
            (dict): return dictionary of metrics
        """
        test_gen_loss, test_mae, test_ssim = self._test_step(dataloader=dataloader)

        return {
            "test_gen_loss": test_gen_loss,
            "test_mae": test_mae,
            "test_ssim": test_ssim,
        }

    def _train_step(self) -> List[float]:
        """
        single train step for the model

        Returns:
            (List[float]): return epoch losses and metrics
        """
        self.gen.train()
        self.disc.train()

        gen_running_loss = 0
        disc_running_loss = 0
        mae_running_loss = 0
        ssim_running = 0
        # psnr_running = 0

        progbar = tqdm.tqdm(
            enumerate(self.train_dataloader, 1),
            total=len(self.train_dataloader),
        )

        for i, (ip, tgt) in progbar:
            ip = ip.to(self.device)
            tgt = tgt.to(self.device)

            # desired targets for the discriminator
            self.optim_d.zero_grad()
            real_target = torch.ones(ip.shape[0], 1, 30, 30).to(self.device)
            fake_target = torch.zeros(ip.shape[0], 1, 30, 30).to(self.device)

            # get preds from generator
            generated_img = self.gen(ip)

            # get discriminator predictions for generated y
            disc_inp_fake = torch.cat((ip, generated_img), 1)
            disc_fake_y = self.disc(disc_inp_fake.detach())
            disc_fake_loss = losses.disc_loss(disc_fake_y, fake_target)

            # get discriminator predictions for true y
            disc_inp_real = torch.cat((ip, tgt), 1)
            disc_real_y = self.disc(disc_inp_real)
            disc_real_loss = losses.disc_loss(disc_real_y, real_target)

            # calculate total discriminator loss, do backprop on it and optimise disc
            disc_total_loss = (disc_fake_loss + disc_real_loss) / 2
            disc_total_loss.backward()
            self.optim_d.step()

            # optimize generator
            self.optim_g.zero_grad()
            disc_inp_fake = torch.cat((ip, generated_img), 1)
            disc_fake_y = self.disc(disc_inp_fake)
            gen_loss = losses.gen_loss(generated_img, tgt, disc_fake_y, real_target)
            gen_loss.backward()
            self.optim_g.step()

            mean_absolute_error = torch.nn.L1Loss()

            disc_running_loss += disc_total_loss.item()
            gen_running_loss += gen_loss.item()
            mae_running_loss += mean_absolute_error(tgt.detach(), generated_img.detach()).to("cpu").item()
            ssim_running += (
                pytorch_msssim.ms_ssim(
                    (tgt.detach() + 1) / 2,
                    (generated_img.detach() + 1) / 2,
                    data_range=1,
                )
                .to("cpu")
                .item()
            )

            # ssim_running += metrics.ssim_for_batch(
            #     (tgt.detach().cpu() + 1) / 2, (generated_img.detach().cpu() + 1) / 2
            # ).item()
            # psnr_running += metrics.psnr_for_batch(
            #     (tgt.detach().cpu() + 1) / 2, (generated_img.detach().cpu() + 1) / 2
            # ).item()

            progbar_descp = f"gen_loss: {gen_running_loss / i:.03f} - disc_loss: {disc_running_loss / i:.03f}"
            progbar_descp += f" - mae: {mae_running_loss / i:.03f} - ssim: {ssim_running / i:.03f}"

            progbar.set_description(progbar_descp)

        epoch_gen_loss = gen_running_loss / len(self.train_dataloader)
        epoch_disc_loss = disc_running_loss / len(self.train_dataloader)
        epoch_mae_loss = mae_running_loss / len(self.train_dataloader)
        epoch_ssim = ssim_running / len(self.train_dataloader)
        # epoch_psnr = psnr_running / len(self.train_dataloader)

        return epoch_gen_loss, epoch_disc_loss, epoch_mae_loss, epoch_ssim

    def _test_step(self, dataloader: torch.utils.data.DataLoader) -> List[float]:
        """
        single test step for the model

        Returns:
            (List[float]): return epoch losses and metrics
        """
        self.gen.eval()
        self.disc.eval()

        gen_running_loss = 0
        mae_running_loss = 0
        ssim_running = 0
        # psnr_running = 0

        for ip, tgt in dataloader:
            ip = ip.to(self.device)
            tgt = tgt.to(self.device)

            generated_img = self.gen(ip)

            real_target = torch.ones(ip.shape[0], 1, 30, 30).to(self.device)
            disc_inp_fake = torch.cat((ip, generated_img), 1)
            disc_fake_y = self.disc(disc_inp_fake)
            gen_loss = losses.gen_loss(generated_img, tgt, disc_fake_y, real_target)

            mean_absolute_error = torch.nn.L1Loss()
            gen_running_loss += gen_loss.item()
            mae_running_loss += mean_absolute_error(tgt.detach(), generated_img.detach()).to("cpu").item()
            ssim_running += (
                pytorch_msssim.ms_ssim(
                    (tgt.detach() + 1) / 2,
                    (generated_img.detach() + 1) / 2,
                    data_range=1,
                )
                .to("cpu")
                .item()
            )

            # ssim_running += metrics.ssim_for_batch(
            #     (tgt.detach().cpu() + 1) / 2, (generated_img.detach().cpu() + 1) / 2
            # ).item()
            # psnr_running += metrics.psnr_for_batch(
            #     (tgt.detach().cpu() + 1) / 2, (generated_img.detach().cpu() + 1) / 2
            # ).item()

        test_gen_loss = gen_running_loss / len(dataloader)
        test_mae_loss = mae_running_loss / len(dataloader)
        test_ssim = ssim_running / len(dataloader)
        # test_psnr = psnr_running / len(dataloader)

        return test_gen_loss, test_mae_loss, test_ssim

    def _plot_sample_imgs(self, epoch_num: int) -> None:
        """
        plot sample images for the model after every epoch
        """
        self.gen.eval()
        self.disc.eval()

        sample_dataloader = dataloader.lfw_dataset(
            img_paths=self.sample_img_list,
            transforms=augmentations.augs(gaussian_noise=False),
        )

        n_rows = 10
        n_cols = 3

        fig = plt.figure(figsize=(10, 45))
        k = 0

        for i in range(0, n_cols * n_rows - 1, n_cols):
            ip, tgt = sample_dataloader[k]
            ip = ip.unsqueeze(0) + self.noise_vector[k, :, :, :]
            ip = ip.clip(-1, 1)
            ip = ip.to(self.device)

            pred = self.gen(ip)
            pred = pred.squeeze().detach().to("cpu")
            ip = ip.squeeze().to("cpu")

            # cast to 0,1 to display
            ip = (ip + 1) / 2
            tgt = (tgt + 1) / 2
            pred = (pred + 1) / 2

            ax = fig.add_subplot(n_rows, n_cols, i + 1)
            ax.imshow(ip.permute(1, 2, 0))
            ax.axis("off")

            if i == 0:
                ax.set_title("Input", fontsize=20)

            ax = fig.add_subplot(n_rows, n_cols, i + 2)
            ax.imshow(tgt.permute(1, 2, 0))
            ax.axis("off")

            if i == 0:
                ax.set_title("Target", fontsize=20)

            ax = fig.add_subplot(n_rows, n_cols, i + 3)
            ax.imshow(pred.permute(1, 2, 0))
            ax.axis("off")

            if i == 0:
                ax.set_title("Predicted", fontsize=20)

            k = k + 1

        fig.tight_layout()
        plt.savefig(os.path.join(self.sample_plots_dir, f"epoch_{epoch_num}_samples.png"))
        plt.close()

    def _save_checkpoint(self, epoch_num: int) -> None:
        """
        save model checkpoint

        Args:
            epoch_num (int): epoch number to save model for
        """
        gen_ckpt_save_path = os.path.join(
            self.checkpoint_dir,
            f"gen_epoch_{epoch_num}_" + self.model_name + ".pth",
        )
        disc_ckpt_save_path = os.path.join(
            self.checkpoint_dir,
            f"disc_epoch_{epoch_num}_" + self.model_name + ".pth",
        )

        torch.save(self.gen, gen_ckpt_save_path)
        torch.save(self.disc, disc_ckpt_save_path)

    def _plot(self) -> None:
        """
        plot model training metrics
        """
        logger.info(f"saving model training plots to {self.plot_dir}")

        metric_keys = self.history.keys()
        metric_keys = [key for key in metric_keys if key[:4] != "val_"]

        for metric in metric_keys:
            fig = plt.figure(figsize=(20, 20))
            fig
            plt.plot(self.history[metric], label=metric)

            if "val_" + metric in self.history.keys():
                plt.plot(self.history["val_" + metric], label="val_" + metric)

            plt.ylabel(metric)
            plt.xlabel("epoch")
            plt.legend()
            plt.savefig(os.path.join(self.plot_dir, f"{metric}.png"))
