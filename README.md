# Bootstrap Your Own Latent (BYOL) in PyTorch
PyTorch implementation of [https://arxiv.org/abs/2006.07733](BYOL) with inspiration from [https://github.com/lucidrains/byol-pytorch](here).
BYOL is a self-supervised method, highly similar to current contrastive learning methods, without the need for negative samples.

# Install requirements
To install the needed requirements in a new conda environment use

```bash
conda create --name <env> --file requirements.txt
```

# Example usage

```python
# Initialize seed and hyperparameters
seed = 0
imgSize = 25

# Ensure reproducibility
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# GPU compatibility 
cudaAvailable = torch.cuda.is_available()
device = torch.device('cuda' if cudaAvailable else 'cpu')

# Initialize backbone, BYOL and optimizer
resnet = models.resnet18(pretrained=True)
byol = BYOL(resnet, imageSize=imgSize, embeddingLayer='avgpool')
byol = byol.to(device)
optimizer = torch.optim.Adam(byol.parameters(), lr=3e-4)

# Train embedding model according to BYOL paper
for epoch in range(15):
    images = torch.randn(2, 3, imgSize, imgSize).to(device)
    loss = byol.contrast(images)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    byol.updateTargetEncoder() # update target encoder by EMA
    print(f'Epoch {epoch+1:>2} --- Loss: {loss.item():2.5f}')
```

# Citation
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