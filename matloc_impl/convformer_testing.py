import torch
from torchsummary import summary
from urllib.request import urlopen
from PIL import Image
import timm

img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

model = timm.create_model('convformer_b36.sail_in1k_384', pretrained=True).cuda()
model = model.eval()

# # get model specific transforms (normalization, resize)
# data_config = timm.data.resolve_model_data_config(model)
# transforms = timm.data.create_transform(**data_config, is_training=False)

# output = model(transforms(img).unsqueeze(0))  # unsqueeze single image into batch of 1

# top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k=5)

print(summary(model, (3,224,224)))
# print(model)
# print(dir(model))
# print(model.children())

# modified = torch.nn.Sequential(model.stem, *list(model.stages.children())[:30])
# modified = torch.nn.Sequential(*[model.flatten()[i] for i in range(10)])


# modified = torch.nn.Sequential(model.stem, model.stages[0])
# modified = modified.eval()
# print(summary(modified, (3,224,224)))

# dummy_image = torch.zeros([1,3,224,224]).cuda()

# out = modified.forward(dummy_image)

# print(out.shape)