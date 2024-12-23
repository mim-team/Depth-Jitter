# import torch
# import torch.nn as nn
# from positional_encodings.torch_encodings import PositionalEncodingPermute2D, Summer
# from q2l_labeller.models.timm_backbone import TimmBackbone
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn import TransformerEncoder, TransformerEncoderLayer

# class TransformerEncoderLayerWithAttention(nn.TransformerEncoderLayer):
#     def __init__(self, *args, **kwargs):
#         super(TransformerEncoderLayerWithAttention, self).__init__(*args, **kwargs)

#     def forward(self, src, src_mask=None, src_key_padding_mask=None):
#         src2, attn_weights = self.self_attn(src, src, src, attn_mask=src_mask,
#                                             key_padding_mask=src_key_padding_mask)
#         src = src + self.dropout1(src2)
#         src = self.norm1(src)
#         src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
#         src = src + self.dropout2(src2)
#         src = self.norm2(src)
#         return src, attn_weights

# class TransformerEncoderWithAttention(nn.Module):
#     def __init__(self, encoder_layer, num_layers, norm=None):
#         super(TransformerEncoderWithAttention, self).__init__()
#         self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
#         self.num_layers = num_layers
#         self.norm = norm

#     def forward(self, src, mask=None, src_key_padding_mask=None):
#         output = src
#         attentions = []

#         for mod in self.layers:
#             output, attn_weights = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
#             attentions.append(attn_weights)

#         if self.norm is not None:
#             output = self.norm(output)

#         return output, attentions


    
# class Query2Label(nn.Module):
#     def __init__(
#         self, model, conv_out, num_classes, hidden_dim=256, nheads=8, 
#         encoder_layers=6, decoder_layers=6, use_pos_encoding=False):
#         super().__init__()

#         self.num_classes = num_classes
#         self.hidden_dim = hidden_dim
#         self.use_pos_encoding = use_pos_encoding

#         self.backbone = TimmBackbone(model)
#         self.conv = nn.Conv2d(conv_out, hidden_dim, 1)
        
#         encoder_layer = TransformerEncoderLayerWithAttention(d_model=hidden_dim, nhead=nheads)
#         self.transformer = TransformerEncoderWithAttention(encoder_layer, num_layers=encoder_layers)

#         if self.use_pos_encoding:
#             self.pos_encoder = PositionalEncodingPermute2D(hidden_dim)
#             self.encoding_adder = Summer(self.pos_encoder)

#         self.classifier = nn.Linear(hidden_dim, num_classes)
#         self.label_emb = nn.Parameter(torch.rand(1, num_classes, hidden_dim))

#     def forward(self, x, return_attention=False):
#         out = self.backbone(x)
#         h = self.conv(out)
#         B, C, H, W = h.shape

#         if self.use_pos_encoding:
#             h = self.encoding_adder(h * 0.1)

#         h = h.flatten(2).permute(2, 0, 1)  # Shape: [H*W, B, C]
#         label_emb = self.label_emb.repeat(B, 1, 1).transpose(0, 1)
#         h, attentions = self.transformer(h, None, None)

#         h = h.transpose(0, 1).contiguous()  # Shape: [B, seq_len, hidden_dim]

#         # Debug: Print the shape of attention weights
#         # for i, att in enumerate(attentions):
#         #     print(f"Layer {i} attention shape: {att.shape}")

#         h = h.mean(dim=1)  # Shape: [B, hidden_dim]

#         logits = self.classifier(h)

#         if return_attention:
#             return logits, attentions  # Return raw attentions
#         return logits



import torch
import torch.nn as nn

from positional_encodings.torch_encodings import PositionalEncodingPermute2D, Summer

from q2l_labeller.models.timm_backbone import TimmBackbone

class Query2Label(nn.Module):
    """Modified Query2Label model

    Unlike the model described in the paper (which uses a modified DETR 
    transformer), this version uses a standard, unmodified Pytorch Transformer. 
    Learnable label embeddings are passed to the decoder module as the target 
    sequence (and ultimately is passed as the Query to MHA).
    """
    def __init__(
        self, model, conv_out, num_classes, hidden_dim=256, nheads=8, 
        encoder_layers=6, decoder_layers=6, use_pos_encoding=False):
        """Initializes model

        Args:
            model (str): Timm model descriptor for backbone.
            conv_out (int): Backbone output channels.
            num_classes (int): Number of possible label classes
            hidden_dim (int, optional): Hidden channels from linear projection of
            backbone output. Defaults to 256.
            nheads (int, optional): Number of MHA heads. Defaults to 8.
            encoder_layers (int, optional): Number of encoders. Defaults to 6.
            decoder_layers (int, optional): Number of decoders. Defaults to 6.
            use_pos_encoding (bool, optional): Flag for use of position encoding. 
            Defaults to False.
        """        
        
        super().__init__()

        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.use_pos_encoding = use_pos_encoding

        self.backbone = TimmBackbone(model)
        self.conv = nn.Conv2d(conv_out, hidden_dim, 1)
        self.transformer = nn.Transformer(
            hidden_dim, nheads, encoder_layers, decoder_layers)

        if self.use_pos_encoding:
            # returns the encoding object
            self.pos_encoder = PositionalEncodingPermute2D(hidden_dim)

            # returns the summing object
            self.encoding_adder = Summer(self.pos_encoder)

        # prediction head
        self.classifier = nn.Linear(num_classes * hidden_dim, num_classes)

        # learnable label embedding
        self.label_emb = nn.Parameter(torch.rand(1, num_classes, hidden_dim))

    def forward(self, x):
        """Passes batch through network

        Args:
            x (Tensor): Batch of images

        Returns:
            Tensor: Output of classification head
        """        
        # produces output of shape [N x C x H x W]
        out = self.backbone(x)
        
        # reduce number of feature planes for the transformer
        h = self.conv(out)
        B, C, H, W = h.shape

        # add position encodings
        if self.use_pos_encoding:
            
            # input with encoding added
            h = self.encoding_adder(h*0.1)

        # convert h from [N x C x H x W] to [H*W x N x C] (N=batch size)
        # this corresponds to the [SIZE x BATCH_SIZE x EMBED_DIM] dimensions 
        # that the transformer expects
        h = h.flatten(2).permute(2, 0, 1)
        
        # image feature vector "h" is sent in after transformation above; we 
        # also convert label_emb from [1 x TARGET x (hidden)EMBED_SIZE] to 
        # [TARGET x BATCH_SIZE x (hidden)EMBED_SIZE]
        label_emb = self.label_emb.repeat(B, 1, 1)
        label_emb = label_emb.transpose(0, 1)
        h = self.transformer(h, label_emb).transpose(0, 1)
        
        # output from transformer was of dim [TARGET x BATCH_SIZE x EMBED_SIZE];
        # however, we transposed it to [BATCH_SIZE x TARGET x EMBED_SIZE] above.
        # below we reshape to [BATCH_SIZE x TARGET*EMBED_SIZE].
        #
        # next, we project transformer outputs to class labels
        h = torch.reshape(h,(B, self.num_classes * self.hidden_dim))

        return self.classifier(h)
        






