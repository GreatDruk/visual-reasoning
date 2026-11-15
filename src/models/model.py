from torch import nn


class VisualReasoningModel(nn.Module):
    """image -> (encoder) -> fusion(image_features, text_encoder(caption)) -> logit.

    `image_encoder` is None when features are cached: forward then receives the cached
    feature maps directly. `spatial=False` averages maps into vectors for pooled heads.
    Pair heads receive one image per twin pair, both text encodings and an edit mask;
    they return centered logits aligned with the original caption rows.
    """

    def __init__(self, text_encoder, fusion, image_encoder=None, spatial=True):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.fusion = fusion
        self.spatial = spatial
        self.requires_pairs = getattr(fusion, "requires_pairs", False)
        self.requires_count_inputs = getattr(fusion, "requires_count_inputs", False)
        if getattr(fusion, "freeze_base", False):
            self.text_encoder.requires_grad_(False)
            if self.image_encoder is not None:
                self.image_encoder.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if getattr(self.fusion, "freeze_base", False):
            self.text_encoder.eval()
            if self.image_encoder is not None:
                self.image_encoder.eval()
        return self

    def forward(self, image, tokens, lengths, twin_index=None, difference_mask=None, count_inputs=None):
        if self.requires_pairs:
            if twin_index is None or difference_mask is None:
                raise ValueError("pair-choice fusion requires twin_index and difference_mask")
            # Encode each image only once; text remains in the original batch order.
            image = image[twin_index[:, 0].to(image.device)]
        if self.image_encoder is not None:
            image = self.image_encoder(image)
        if not self.spatial and image.dim() == 4:
            image = image.mean(dim=(2, 3))
        text = self.text_encoder(tokens, lengths)
        if self.requires_count_inputs:
            count_text = None
            if count_inputs is not None:
                ci = count_inputs
                filters = self.text_encoder(ci["filter_tokens"], ci["filter_lengths"])
                references = self.text_encoder(ci["reference_tokens"], ci["reference_lengths"])
                count_text = (ci["rows"], ci["numbers"], ci["relations"], filters, references)
            return self.fusion(image, text, twin_index, difference_mask, count_text=count_text)
        if self.requires_pairs:
            return self.fusion(image, text, twin_index, difference_mask)
        return self.fusion(image, text)
