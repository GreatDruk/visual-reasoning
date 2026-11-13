from torch import nn


class VisualReasoningModel(nn.Module):
    """image -> (encoder) -> fusion(image_features, text_encoder(caption)) -> logit.

    `image_encoder` is None when features are cached: forward then receives the cached
    feature maps directly. `spatial=False` averages maps into vectors for pooled heads.
    """

    def __init__(self, text_encoder, fusion, image_encoder=None, spatial=True):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.fusion = fusion
        self.spatial = spatial

    def forward(self, image, tokens, lengths):
        if self.image_encoder is not None:
            image = self.image_encoder(image)
        if not self.spatial and image.dim() == 4:
            image = image.mean(dim=(2, 3))
        return self.fusion(image, self.text_encoder(tokens, lengths))
