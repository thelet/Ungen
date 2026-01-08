from Core.Interfaces import ISimilarityMetric
import lpips


class LPIPSMetric(ISimilarityMetric):
    def __init__(self, net='vgg', device='cuda'):
        super(LPIPSMetric, self).__init__()
        # Initialize the official LPIPS model
        self.loss_fn = lpips.LPIPS(net=net).to(device)
        
    def forward(self, img1, img2):
        # LPIPS expects inputs in range [-1, 1], but our pipeline usually uses [0, 1]
        # We normalize explicitly here to be safe.
        img1_norm = (img1 * 2) - 1
        img2_norm = (img2 * 2) - 1
        
        return self.loss_fn(img1_norm, img2_norm).mean()