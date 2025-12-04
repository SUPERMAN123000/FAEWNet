# Copyright (c) Open-CD. All rights reserved.
from mmengine.model import BaseModule
from mmengine.dist import is_main_process

try:
    from peft import get_peft_config, get_peft_model

    PEFT_INSTALLED = True
except:
    get_peft_config = None
    get_peft_model = None
    PEFT_INSTALLED = False

from opencd.registry import MODELS


@MODELS.register_module()
class VisionTransformerTurner(BaseModule):
    def __init__(
            self,
            encoder_cfg,
            peft_cfg=None,
            init_cfg=None,
    ):
        if not PEFT_INSTALLED:
            raise ImportError(
                'peft is not installed, '
                'we suggest install peft by '
                '"pip install peft"'  # noqa
            )

        super().__init__(init_cfg=init_cfg)
        vision_encoder = MODELS.build(encoder_cfg)
        vision_encoder.init_weights()
        if peft_cfg is not None and isinstance(peft_cfg, dict):
            config = {
                "peft_type": "LORA",
                "r": 16,
                'target_modules': ["qkv"],
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "bias": "none",
                "inference_mode": False,
            }
            config.update(peft_cfg)
            peft_config = get_peft_config(config)
            self.vision_encoder = get_peft_model(vision_encoder, peft_config)
            if is_main_process():
                self.vision_encoder.print_trainable_parameters()
        else:
            self.vision_encoder = vision_encoder
            # freeze the vision encoder
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.vision_encoder(x)
######################################################################################################
# # Copyright (c) Open-CD. All rights reserved.
# from mmengine.model import BaseModule
# from mmengine.dist import is_main_process
#
# try:
#     from peft import get_peft_config, get_peft_model  # 确保已安装AdaLoRA
#
#     PEFT_INSTALLED = True
# except:
#     get_peft_config = None
#     get_peft_model = None
#     PEFT_INSTALLED = False
#
# from opencd.registry import MODELS
#
#
# @MODELS.register_module()
# class VisionTransformerTurner(BaseModule):
#     def __init__(
#             self,
#             encoder_cfg,
#             peft_cfg=None,
#             init_cfg=None,
#     ):
#         if not PEFT_INSTALLED:
#             raise ImportError(
#                 'peft is not installed, '
#                 'we suggest install peft by '
#                 '"pip install peft"'  # noqa
#             )
#
#         super().__init__(init_cfg=init_cfg)
#         vision_encoder = MODELS.build(encoder_cfg)
#         vision_encoder.init_weights()
#         if peft_cfg is not None and isinstance(peft_cfg, dict):
#             config = {
#                 "peft_type": "ADALORA",  # 替换为AdaLoRA类型
#                 "r": 16,  # 可以根据需要调整
#                 "target_modules": ["qkv"],
#                 "lora_alpha": 32,  # AdaLoRA的超参数
#                 "lora_dropout": 0.1,  # 使用适当的dropout
#                 "bias": "none",
#                 "inference_mode": False,
#             }
#             config.update(peft_cfg)
#             peft_config = get_peft_config(config)
#             self.vision_encoder = get_peft_model(vision_encoder, peft_config)
#             if is_main_process():
#                 self.vision_encoder.print_trainable_parameters()
#         else:
#             self.vision_encoder = vision_encoder
#             # freeze the vision encoder
#             for param in self.vision_encoder.parameters():
#                 param.requires_grad = False
#
#     def forward(self, x):
#         return self.vision_encoder(x)
####################################################################################################
# # Import necessary libraries
# from mmengine.model import BaseModule
# from mmengine.dist import is_main_process
#
# try:
#     from peft import LoraConfig, get_peft_model
#
#     PEFT_INSTALLED = True
# except ImportError:
#     LoraConfig = None
#     get_peft_model = None
#     PEFT_INSTALLED = False
#
# from opencd.registry import MODELS
#
#
# @MODELS.register_module()
# class VisionTransformerTurner(BaseModule):
#     def __init__(
#             self,
#             encoder_cfg,
#             peft_cfg=None,
#             init_cfg=None,
#     ):
#         if not PEFT_INSTALLED:
#             raise ImportError(
#                 'peft is not installed. '
#                 'Install peft by running "pip install peft".'  # noqa
#             )
#
#         super().__init__(init_cfg=init_cfg)
#         vision_encoder = MODELS.build(encoder_cfg)
#         vision_encoder.init_weights()
#
#         if peft_cfg is not None and isinstance(peft_cfg, dict):
#             # Update config to use DoRA instead of LoRA
#             config = {
#                 "peft_type": "LORA",
#                 "r": 16,
#                 "target_modules": ["qkv"],
#                 "lora_alpha": 32,
#                 "lora_dropout": 0.05,
#                 "bias": "none",
#                 "inference_mode": False,
#                 "use_dora": True  # Enable DoRA
#             }
#             config.update(peft_cfg)
#
#             # Create the LoraConfig with DoRA enabled
#             lora_config = LoraConfig(**config)
#             self.vision_encoder = get_peft_model(vision_encoder, lora_config)
#
#             if is_main_process():
#                 self.vision_encoder.print_trainable_parameters()
#
#         else:
#             self.vision_encoder = vision_encoder
#             # Freeze the vision encoder if no peft config is provided
#             for param in self.vision_encoder.parameters():
#                 param.requires_grad = False
#
#     def forward(self, x):
#         return self.vision_encoder(x)
##########################################################################
# # Import necessary libraries
# from mmengine.model import BaseModule
# from mmengine.dist import is_main_process
#
# try:
#     from peft import LoraConfig, get_peft_model
#
#     PEFT_INSTALLED = True
# except ImportError:
#     LoraConfig = None
#     get_peft_model = None
#     PEFT_INSTALLED = False
#
# from opencd.registry import MODELS
#
#
# @MODELS.register_module()
# class VisionTransformerTurner(BaseModule):
#     def __init__(
#             self,
#             encoder_cfg,
#             peft_cfg=None,
#             init_cfg=None,
#     ):
#         if not PEFT_INSTALLED:
#             raise ImportError(
#                 'peft is not installed. '
#                 'Install peft by running "pip install peft".'  # noqa
#             )
#
#         super().__init__(init_cfg=init_cfg)
#         vision_encoder = MODELS.build(encoder_cfg)
#         vision_encoder.init_weights()
#
#         if peft_cfg is not None and isinstance(peft_cfg, dict):
#             # Update config to use DoRA instead of LoRA
#             config = {
#                 "peft_type": "LORA",
#                 "r": 16,
#                 "target_modules": ["qkv"],
#                 "lora_alpha": 32,
#                 "lora_dropout": 0.05,
#                 "bias": "none",
#                 "inference_mode": False,
#                 "use_rslora": True # Enable RSLoRA
#             }
#             config.update(peft_cfg)
#
#             # Create the LoraConfig with DoRA enabled
#             lora_config = LoraConfig(**config)
#             self.vision_encoder = get_peft_model(vision_encoder, lora_config)
#
#             if is_main_process():
#                 self.vision_encoder.print_trainable_parameters()
#
#         else:
#             self.vision_encoder = vision_encoder
#             # Freeze the vision encoder if no peft config is provided
#             for param in self.vision_encoder.parameters():
#                 param.requires_grad = False
#
#     def forward(self, x):
#         return self.vision_encoder(x)