
from post_process_volume import post_process_volume
import numpy as np
from postprocessing_utils.vsnr import vsnr


alphas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
sigmas = [(1, 30), (5, 30), (10, 30), (1, 40), (5, 40), (10, 40), (1, 50), (5, 50), (10, 50)]
thetas = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340]
thetas = [0]

for alpha in alphas:
    for sigma in sigmas:
        for theta in thetas:
            post_process_volume(
                source_path='/media/julian/Data/projects/steyer/amst_introduction/2022-02-16_PF_as43a_Acantharia/workflow01_sift_vsnr_clahe/pre_align_sift/pre_align/',
                target_path=f'/media/julian/Data/projects/steyer/amst_introduction/2022-02-16_PF_as43a_Acantharia/workflow01_sift_vsnr_clahe/sift_vsnr_a{alpha}_s{sigma}_t{theta}/',
                dtype='uint16',
                n_workers=1,
                postprocess=vsnr,
                postprocess_args=dict(
                    alpha=alpha,
                    sigma=sigma,
                    theta=theta
                ),
                roi_z=np.s_[16:17],
                verbose=True
            )
