import torch
from . import initialization as init

class a():
    def __init__(
            self
    ):
        pass
    def b(self):
        print('k')


class SegmentationModel(torch.nn.Module):

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)
        if hasattr(self, 'segmentation_head2'):  # decide whether the attribute/method exists
            if self.segmentation_head2 is not None:
                init.initialize_head(self.segmentation_head2)
        else:
            self.segmentation_head2 = None # if this is not defined, z.B in deepLab

        if hasattr(self, 'seg_head_doc1'):  # decide whether the attribute/method exists
            if self.seg_head_doc1 is not None:
                init.initialize_head(self.seg_head_doc1)
        else:
            self.seg_head_doc1 = None
        if hasattr(self, 'seg_head_doc2'):  # decide whether the attribute/method exists
            if self.seg_head_doc2 is not None:
                init.initialize_head(self.seg_head_doc2)
        else:
            self.seg_head_doc2 = None

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)
        # or if is_multidoc !!!

        if self.segmentation_head2 is not None and self.seg_head_doc1 is not None and self.seg_head_doc2 is not None and self.hcc_head_doc2 is not None:
            mask_doc1 = self.seg_head_doc1(decoder_output)
            mask_doc2 = self.seg_head_doc2(decoder_output)
            masks2 = self.segmentation_head2(decoder_output)
            hcc_doc1 = self.hcc_head_doc1(decoder_output)
            hcc_doc2 = self.hcc_head_doc2(decoder_output)
            return masks, mask_doc1, mask_doc2, masks2, hcc_doc1, hcc_doc2

        if self.segmentation_head2 is not None and self.seg_head_doc1 is not None and self.seg_head_doc2 is not None:
            mask_doc1 = self.seg_head_doc1(decoder_output)
            mask_doc2 = self.seg_head_doc2(decoder_output)
            masks2 = self.segmentation_head2(decoder_output)
            return masks, mask_doc1, mask_doc2, masks2

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])  # fed features from last layer, omg!
            return masks, labels

        if self.segmentation_head2 is not None:
            masks2 = self.segmentation_head2(decoder_output)
            # print("this way to seg2")
            return masks, masks2

        if self.classification_head is not None and self.segmentation_head2 is not None:
            labels = self.classification_head(features[-1])
            masks2 = self.segmentation_head2(decoder_output)
            return masks, masks2, labels

        return masks

    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)

        return x

