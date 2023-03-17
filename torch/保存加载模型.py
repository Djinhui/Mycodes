import torch
import torchvision.models as models

model = models.vgg16(pretrained=True)


# 一：Saving and Loading Model Weights

# 1.SAVE: PyTorch models store the learned parameters in an internal state dictionary, called state_dict
torch.save(model.state_dict(), 'model_weights.pth')
# 2.LOAD: To load model weights, you need to create an instance of the same model first,
#  and then load the parameters using load_state_dict() method
model = models.vgg16() # we do not specify pretrained=True, i.e. do not load default weights
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

# 二：Saving and Loading Models with Shapes
'''
When loading model weights, we needed to instantiate the model class first, 
because the class defines the structure of a network. 
We might want to save the structure of this class together with the model,
in which case we can pass model (and not model.state_dict()) to the saving function
'''
torch.save(model, 'model.pth')
model = torch.load('model.pth')