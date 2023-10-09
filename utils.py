from torch.utils.tensorboard import SummaryWriter
import numpy as np

class Logger(object):
    def __init__(self, log_dir, **kwargs):
        """Create a summary writer logging to log_dir."""
        self.writer = SummaryWriter(log_dir, **kwargs)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        self.writer.add_scalar(tag, value, step)

    def image_summary(self, tag, images, step):
        """Log a list of images.
        Args::images: numpy of shape (Batch x C x H x W) in the range [-1.0, 1.0]
        """
        if type(images) == tuple or type(images) == list:
            images = np.array(images)
        self.writer.add_images("{}".format(tag), images, global_step=step, dataformats="NCHW")

            
    def histo_summary(self, tag, values, step, bins="auto"):
        """Log a histogram of the tensor of values."""
        self.writer.add_histogram('{}'.format(tag), values, bins=bins, global_step=step)
