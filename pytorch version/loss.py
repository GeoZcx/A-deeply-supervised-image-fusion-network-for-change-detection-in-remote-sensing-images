import torch.nn as nn

def cd_loss(input,target):
    bce_loss = nn.BCELoss()
    bce_loss = bce_loss(torch.log(input),torch.log(target))

    smooth = 1.
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    dic_loss = 1 - ((2. * intersection + smooth)/(iflat.sum() + tflat.sum() + smooth))

    return  dic_loss + bce_loss
