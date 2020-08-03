![BYOL method](/method.png)

# Bootstrap Your Own Latent (BYOL) in PyTorch
PyTorch implementation of [BYOL](https://arxiv.org/abs/2006.07733), developed by Google Deepmind, with inspiration from [here](https://github.com/lucidrains/byol-pytorch).
BYOL is a self-supervised method, highly similar to current contrastive learning methods, without the need for negative samples.

Essentially, BYOL projects an embedding of two independent views of a single image to some low-dimensional space using an online model, and a target model (EMA of online model). Afterwards, a predictor (MLP) predicts the target projection from the online projection, and the loss is backpropagated only through the online model parameters. Intuitively; if the two embeddings are good (close to each other), it should be easy to predict one from the other.

## Install requirements
To install the needed requirements in a new conda environment (BYOL) use

```bash
conda env create -f environment.yml
```

## Example usage
Apply the BYOL class by specifying (1) the neural network used as backbone, (2) image dimensions for randomized cropping (must match input dimension of the backbone), and (3) the position or name of the layer in the backbone which should be used as the embedding.

One can freely specify all the parameters of the BYOL instance, but they are currently alligned with the original paper.

```python
import torch
from BYOL.byol import BYOL
from torchvision import models

# Initialize seed and hyperparameters
seed = 0
imgSize = 256

# Ensure reproducibility
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# Initialize backbone, BYOL and optimizer
resnet = models.resnet50(pretrained=True)
byol = BYOL(resnet, imageSize=imgSize, embeddingLayer='avgpool')
optimizer = torch.optim.Adam(byol.parameters(), lr=3e-4)

# GPU compatibility 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
byol = byol.to(device)

# Train embedding model according to BYOL paper
for epoch in range(15):
    images = torch.randn(10, 3, imgSize, imgSize).to(device)
    loss = byol.contrast(images)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    byol.updateTargetEncoder() # update target encoder by EMA
    print(f'Epoch {epoch+1:>2} --- Loss: {loss.item():2.5f}')
```

After training, the BYOL instance will produce improved embeddings for downstream tasks simply by calling the instance on a (batch of) images.
```python
images = torch.randn(10, 3, imgSize, imgSize).to(device)
embeddings = byol(images)
```

## Citation
Remember to cite the paper.
```bibtex
@misc{grill2020bootstrap,
    title = {Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning},
    author = {Jean-Bastien Grill and Florian Strub and Florent Altché and Corentin Tallec and Pierre H. Richemond and Elena Buchatskaya and Carl Doersch and Bernardo Avila Pires and Zhaohan Daniel Guo and Mohammad Gheshlaghi Azar and Bilal Piot and Koray Kavukcuoglu and Rémi Munos and Michal Valko},
    year = {2020},
    eprint = {2006.07733},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
```