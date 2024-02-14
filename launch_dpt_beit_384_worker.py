from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch  # type: ignore

from foreign_model_commons import RemoteDepthEstWorkerMainLoop


def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    model = torch.hub.load(
        "isl-org/MiDaS", "DPT_BEiT_L_384", trust_repo=True, pretrained=True
    ).to(DEVICE)
    model.eval()

    midas_transforms = torch.hub.load("isl-org/MiDaS", "transforms", trust_repo=True)
    transform = midas_transforms.dpt_transform

    def proc_fn(ims: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            im_tensor: torch.Tensor = transform_batch(transform, ims=ims).to(DEVICE)
            disp_tensor = model(im_tensor)
            disp_tensor = torch.nn.functional.interpolate(
                disp_tensor.unsqueeze(1),
                size=(ims.shape[2], ims.shape[3]),
                mode="bicubic",
                align_corners=False,
            )
            return disp_tensor.cpu().numpy().astype(np.float32)

    main_loop = RemoteDepthEstWorkerMainLoop("dptbeit384")
    main_loop.wait_for_input(proc_fn)


def transform_batch(elem_transform_fn: Any, ims: np.ndarray) -> torch.Tensor:
    elem_tensors: list[torch.Tensor] = []
    for i in range(ims.shape[0]):
        cv_like_im = (ims[i].transpose([1, 2, 0]) * 255).astype(np.uint8)
        elem_tensor = elem_transform_fn(cv_like_im)
        elem_tensors.append(elem_tensor)

    return torch.cat(elem_tensors, dim=0)


if __name__ == "__main__":
    main()
