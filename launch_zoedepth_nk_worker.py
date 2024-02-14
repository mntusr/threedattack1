import numpy as np
import torch  # type: ignore

from foreign_model_commons import RemoteDepthEstWorkerMainLoop


def main():
    repo = "isl-org/ZoeDepth"
    # Zoe_NK
    model_zoe_n = torch.hub.load(repo, "ZoeD_NK", pretrained=True, config_mode="infer")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    zoe = model_zoe_n.to(DEVICE)
    zoe.eval()

    def proc_fn(im: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            im_tensor = torch.tensor(im).to(DEVICE)
            # The published evaluation code also uses flip augmentation, but does not use input padding.
            depth_tensor = zoe.infer(im_tensor, with_flip_aug=True, pad_input=False)
            return depth_tensor.cpu().numpy().astype(np.float32)

    # do i have to rescale?
    # see line 213 at misc.py, but infer has a rescale op too

    main_loop = RemoteDepthEstWorkerMainLoop("zoedepth")
    main_loop.wait_for_input(proc_fn)


if __name__ == "__main__":
    main()
