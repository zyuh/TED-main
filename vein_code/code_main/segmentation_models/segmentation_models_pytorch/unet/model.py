from typing import Optional, Union, List
from .decoder import UnetDecoder
from ..encoders import get_encoder
from ..base import SegmentationModel
from ..base import SegmentationHead, ClassificationHead


class Unet(SegmentationModel):
    """Unet_ is a fully convolution neural network for image semantic segmentation

    Args:
        encoder_name: name of classification model (without last dense layers) used as feature
            extractor to build segmentation model.
        encoder_depth (int): number of stages used in decoder, larger depth - more features are generated.
            e.g. for depth=3 encoder will generate list of features with following spatial shapes
            [(H,W), (H/2, W/2), (H/4, W/4), (H/8, W/8)], so in general the deepest feature tensor will have
            spatial resolution (H/(2^depth), W/(2^depth)]
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        decoder_channels: list of numbers of ``Conv2D`` layer filters in decoder blocks
        decoder_use_batchnorm: if ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation`` layers
            is used. If 'inplace' InplaceABN will be used, allows to decrease memory consumption.
            One of [True, False, 'inplace']
        decoder_attention_type: attention module used in decoder of the model
            One of [``None``, ``scse``]
        in_channels: number of input channels for model, default is 3.
        classes: a number of classes for output (output shape - ``(batch, classes, h, w)``).
        activation: activation function to apply after final convolution;
            One of [``sigmoid``, ``softmax``, ``logsoftmax``, ``identity``, callable, None]
        aux_params: if specified model will have additional classification auxiliary output
            build on top of encoder, supported params:
                - classes (int): number of classes
                - pooling (str): one of 'max', 'avg'. Default is 'avg'.
                - dropout (float): dropout factor in [0, 1)
                - activation (str): activation function to apply "sigmoid"/"softmax" (could be None to return logits)

    Returns:
        ``torch.nn.Module``: **Unet**

    .. _Unet:
        https://arxiv.org/pdf/1505.04597

    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: str = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
        aux_branches_params: Optional[dict] = None,  # nested dicts in a dict
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False, # !! resnet false!!
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_branches_params is not None:
            if 'classification_head' in aux_branches_params.keys():
                self.classification_head = ClassificationHead(
                    in_channels=self.encoder.out_channels[-1], **aux_branches_params['classification_head']
                )
            else:
                self.classification_head = None
            if 'segmentation_head2' in aux_branches_params.keys():
                self.segmentation_head2 = SegmentationHead(
                    in_channels=decoder_channels[-1], **aux_branches_params['segmentation_head2']
                )  # use default except out_channels
            else:
                self.segmentation_head2 = None
            if 'seg_head_doc1' in aux_branches_params.keys():
                self.seg_head_doc1 = SegmentationHead(
                    in_channels=decoder_channels[-1], **aux_branches_params['seg_head_doc1']
                )  # use default except out_channels
            else:
                self.seg_head_doc1 = None
            if 'seg_head_doc2' in aux_branches_params.keys():
                self.seg_head_doc2 = SegmentationHead(
                    in_channels=decoder_channels[-1], **aux_branches_params['seg_head_doc2']
                )  # use default except out_channels
            else:
                self.seg_head_doc2 = None
            if 'hcc_head_doc1' in aux_branches_params.keys():
                self.hcc_head_doc1 = SegmentationHead(
                    in_channels=decoder_channels[-1], **aux_branches_params['seg_head_doc1']
                )  # use default except out_channels
            else:
                self.hcc_head_doc1 = None
            if 'hcc_head_doc2' in aux_branches_params.keys():
                self.hcc_head_doc2 = SegmentationHead(
                    in_channels=decoder_channels[-1], **aux_branches_params['seg_head_doc2']
                )  # use default except out_channels
            else:
                self.hcc_head_doc2 = None
        else:
            self.classification_head = None
            self.segmentation_head2 = None
            self.seg_head_doc1 = None
            self.seg_head_doc2 = None
            self.hcc_head_doc1 = None
            self.hcc_head_doc2 = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()
