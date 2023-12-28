# MICA - Dataset Instructions

The MICA dataset consists of about 2300 subjects, built by unifying existing small- and medium-scale datasets under a common FLAME topology. 
For each subdataset, we provide the reconstructed face geometries including the FLAME parameters.
Follow the instructions on our github repository [MICA link](https://github.com/Zielon/MICA/tree/master/datasets) to download the entire data (specifically, the datasets have to be downloaded separately).

In the case of any questions: mica@tue.mpg.de

**Note that you have to comply with the licence of the original dataset owners. The data and more information can be found under [Florence 2D/3D](https://www.micc.unifi.it/resources/datasets/florence-3d-faces/)**

The data is shared only for academic, non-commercial usage.

## Data Structure

This dataset contains registered meshes including their corresponding FLAME parameters. Actors are split between individual folders with a unique identifier based on the original dataset: 
```shell
root\
    FLAME_parameters\
        actor_id\
            *.npz
    registrations\
        actor_id\
            *.obj
```

**Note that we use the FLAME2020 model.** To retrieve FLAME2020 parameters from the *.npz files, you can simply do:
```python
import numpy as np
import torch

params = np.load('path.npz', allow_pickle=True)
pose = torch.tensor(params['pose']).float()
betas = torch.tensor(params['betas']).float()

flame_params = {
    'shape_params': betas[:300],
    'expression_params': betas[300:],
    'pose_params': torch.cat([pose[:3], pose[6:9]]),
}
```

## Licence:
This dataset is an extension of an already existing one, therefore, you have to comply with the original license. We share this dataset only for academic, non-commercial usage.

If you use this dataset in your research please cite MICA:
```bibtex
@article{MICA:ECCV2022,
  author = {Zielonka, Wojciech and Bolkart, Timo and Thies, Justus},
  title = {Towards Metrical Reconstruction of Human Faces},
  journal = {European Conference on Computer Vision},
  year = {2022}
}
```

## Useful links:
[MICA](https://zielon.github.io/mica/)
[FLAME](https://flame.is.tue.mpg.de/)

