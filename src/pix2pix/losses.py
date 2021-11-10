from torch import nn

mean_absolute_error = nn.L1Loss()
crossentropy_loss = nn.BCELoss()


def gen_loss(generated_img, target_img, disc_fake_y, real_target, lambda_param=100):
    gen_loss = crossentropy_loss(disc_fake_y, real_target)
    mse = mean_absolute_error(generated_img, target_img)
    gen_total_loss = gen_loss + (lambda_param * mse)

    return gen_total_loss


def disc_loss(output, label):
    return crossentropy_loss(output, label)
