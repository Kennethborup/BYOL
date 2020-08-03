import torch
import torch.nn as nn
from kornia import augmentation, filters, color
import copy

from BYOL.utils import EMA, RandomApply, Hook

    
def partLoss(x, y):
    """
    Half of symmetric loss function according to BYOL paper.
    """
    x = torch.nn.functional.normalize(x, dim=-1, p=2)
    y = torch.nn.functional.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


class MLP(nn.Module):
    """
    Simple MLP model for projector and predictor in BYOL paper.
    
    :param inputDim: int; amount of input nodes
    :param projectionDim: int; amount of output nodes
    :param hiddenDim: int; amount of hidden nodes
    """
    def __init__(self, inputDim, projectionDim, hiddenDim=4096):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(inputDim, hiddenDim),
            nn.BatchNorm1d(hiddenDim),
            nn.ReLU(inplace=True),
            nn.Linear(hiddenDim, projectionDim)
        )

    def forward(self, x):
        return self.net(x)
    

class ModelWrapper(nn.Module):
    """
    Wrapper of backbone model to pipe output to a projector and obtain both embeddings
    and projected embeddings.
    
    :param model: PyTorch model; encoding backbone model.
    :param projectionDim: int; amount of output nodes for projector
    :param projectionHiddenDim: int; amount of nodes in hidden layer of projector
    :param embeddingLayer: int or str; layer position or layer name for backbone model, to use as output
    layer for the projector
    """
    def __init__(self, model, projectionDim, projectionHiddenDim, embeddingLayer=-2):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.embeddingLayer = embeddingLayer

        self.projector = None
        self.projectionDim = projectionDim
        self.projectionHiddenDim = projectionHiddenDim
        
        # Register hook
        if self.embeddingLayer != -1:
            self.hook = Hook()
            self.hook.setHook(self._getLayer(self.embeddingLayer))

    def _getLayer(self, embeddingLayer):
        """
        Fetch layer from model.
        
        :param embeddingLayer: int or str;  layer position or layer name for backbone model, to use as output
        layer for the projector
        """
        if type(embeddingLayer) == str:
            modules = dict([*self.model.named_modules()])
            return modules.get(embeddingLayer, None)
        elif type(embeddingLayer) == int:
            children = [*self.model.children()]
            return children[embeddingLayer]
        else:
            raise  NameError(f'Hidden layer ({self.layer}) not found!')

    def getProjector(self, dim):
        return MLP(dim, self.projectionDim, self.projectionHiddenDim)
            
    def embed(self, x):
        """
        Embed x according to backbone model and embeddingLayer.
        """
        # If projection is last layer, just get output (assuming it is flattened)
        if self.embeddingLayer == -1:
            return self.model(x)
        else: # If embedding is not last layer forward model and get hidden representation by hook      
            _ = self.model(x)
            embedding = self.hook.val()
            embedding = embedding.reshape(embedding.size(0), -1)
            return embedding
        
    def forward(self, x):
        """
        Calculate projected embedding according to BYOL paper.
        """
        # Get embedding
        embedding = self.embed(x)
        
        # Get projection
        if self.projector is None:
            projector = self.getProjector(dim=embedding.size(1))
            self.projector = projector.to(embedding)
        projection = self.projector(embedding)
        return projection

# BYOL
class BYOL(nn.Module):
    """
    BYOL class for constructing a BYOL like model structure, forward pass and contrastive learning procedure.
    
    :param model: PyTorch model; encoding backbone model.
    :param imageSize: int; width and height of crops for model.
    :param embeddingLayer: int or str; layer position or layer name for backbone model, to use as output
    layer for the projector
    :param projectionDim: int; amount of output nodes for projector
    :param projectionHiddenDim: int; amount of nodes in hidden layer of projector
    :param emaDecay: float; decay rate for EMA of target model weights
    """
    def __init__(self, model, imageSize, embeddingLayer=-2, projectionDim=256, projectionHiddenDim=4096, emaDecay=0.99):
        super(BYOL, self).__init__()
        
        # Default SimCLR augmentations
        self.augment = nn.Sequential(
            RandomApply(augmentation.ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.8),
            augmentation.RandomGrayscale(p=0.2),
            augmentation.RandomHorizontalFlip(),
            RandomApply(filters.GaussianBlur2d((3, 3), (1.5, 1.5)), p=0.1),
            augmentation.RandomResizedCrop((imageSize, imageSize)),
            color.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225]))
        )
        
        # Initialize models, predictors and EMA
        self.onlineEncoder = ModelWrapper(model, projectionDim, projectionHiddenDim, embeddingLayer)
        self.onlinePredictor = MLP(projectionDim, projectionDim, projectionHiddenDim)
        self.targetEncoder = copy.deepcopy(self.onlineEncoder)
        self.targetEMA = EMA(emaDecay)
    
    def updateTargetEncoder(self):
        """
        Update weights of self.targetEncoder by an EMA of self.onlineEncoder.
        """
        for onlineParams, targetParams in zip(self.onlineEncoder.parameters(), self.targetEncoder.parameters()):
                targetParams.data = self.targetEMA(MA=targetParams.data, value=onlineParams.data)
    
    def contrast(self, x):
        """
        Calculate loss on two views of x according to BYOL paper.
        """
        # Two independent augmentation --> two views
        view1, view2 = self.augment(x), self.augment(x)
        
        # Get online embeddings
        onlineProjection1 = self.onlineEncoder(view1)
        onlineProjection2 = self.onlineEncoder(view2)
        
        # Get target embeddings
        with torch.no_grad():
            targetProjection1 = self.targetEncoder(view1)
            targetProjection2 = self.targetEncoder(view2)

        # Get predictions
        onlinePred1 = self.onlinePredictor(onlineProjection1)
        onlinePred2 = self.onlinePredictor(onlineProjection2)
        
        # Calculate loss terms
        loss1 = partLoss(onlinePred1, targetProjection2.detach())
        loss2 = partLoss(onlinePred2, targetProjection1.detach())
        loss = loss1 + loss2
        
        return loss.mean()
    
    def forward(self, x):
        """
        Calculate embedding (not projected).
        """
        embedding = self.onlineEncoder.embed(x)
        return embedding