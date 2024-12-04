import torch


class LlavaOnevisionUtils:
    # https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava_onevision/modeling_llava_onevision.py

    def __init__(self, config, newline):
        self.config = config
        self.image_newline = newline

    def postprocess_image(self, image_features, image_sizes,
                          image_num_patches):

        import math

        from torch import nn
        from transformers.models.llava_onevision.modeling_llava_onevision import (
            get_anyres_image_grid_shape, unpad_image)
        image_features = torch.split(image_features, image_num_patches, dim=0)
        vision_aspect_ratio = self.config.vision_aspect_ratio

        # LlavaOnevisionForConditionalGeneration.pack_image_features
        new_image_features = []
        feature_lens = []
        for image_idx, image_feature in enumerate(image_features):
            if image_feature.shape[0] > 1:
                base_image_feature = image_feature[0]
                image_feature = image_feature[1:]
                height = width = self.config.vision_config.image_size // self.config.vision_config.patch_size
                if height * width != base_image_feature.shape[0]:
                    raise ValueError(
                        "The number of patches is not consistent with the image size."
                    )
                num_patch_height, num_patch_width = get_anyres_image_grid_shape(
                    image_sizes[image_idx],
                    self.config.image_grid_pinpoints,
                    self.config.vision_config.image_size,
                )
                image_feature = image_feature.view(num_patch_height,
                                                   num_patch_width, height,
                                                   width, -1)
                image_feature = image_feature.permute(4, 0, 2, 1,
                                                      3).contiguous()
                image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                image_feature = unpad_image(image_feature,
                                            image_sizes[image_idx])
                max_num_patches = int(vision_aspect_ratio.strip("anyres_max_"))
                channels, curr_height, curr_width = image_feature.shape
                ratio = math.sqrt(curr_height * curr_width /
                                  (max_num_patches * height**2))
                if ratio > 1.1:
                    image_feature = image_feature[None]
                    image_feature = nn.functional.interpolate(
                        image_feature,
                        [int(curr_height // ratio),
                         int(curr_width // ratio)],
                        mode="bilinear")[0]
                if self.image_newline is not None:
                    image_feature = torch.cat(
                        (
                            image_feature,
                            self.image_newline[:, None, None].expand(
                                *image_feature.shape[:-1], 1).to(
                                    image_feature.device, image_feature.dtype),
                        ),
                        dim=-1,
                    )
                image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                image_feature = torch.cat((base_image_feature, image_feature),
                                          dim=0)
            else:
                image_feature = image_feature[0]
                if self.image_newline is not None:
                    image_feature = torch.cat(
                        (image_feature,
                         self.image_newline[None].to(image_feature)),
                        dim=0)
            new_image_features.append(image_feature)
            feature_lens.append(image_feature.size(0))

        max_tokens = max(x.shape[0] for x in new_image_features)
        new_image_features = [
            torch.nn.functional.pad(table,
                                    (0, 0, 0, max_tokens - table.shape[0]),
                                    mode='constant')
            for table in new_image_features
        ]
        image_features = torch.stack(new_image_features, dim=0)
        return image_features

    def postprocess_video(self, image_features, batch_size, frames):

        # LlavaOnevisionForConditionalGeneration.apply_pooling
        import math

        from torch import nn
        height = width = self.config.vision_config.image_size // self.config.vision_config.patch_size
        batch_frames, seq_len, dim = image_features.shape
        image_features = image_features.view(batch_frames, height, width, -1)
        image_features = image_features.permute(0, 3, 1, 2).contiguous()

        height, width = image_features.shape[2:]
        scaled_shape = [math.ceil(height / 2), math.ceil(width / 2)]
        image_features = nn.functional.interpolate(image_features,
                                                   size=scaled_shape,
                                                   mode="bilinear")

        image_features = image_features.permute(0, 2, 3, 1)
        image_features = image_features.view(batch_frames, -1, dim)

        video_features = image_features.reshape(
            batch_size, frames * image_features.shape[1], -1)
        image_newline = self.image_newline[None, None, :].repeat(
            batch_size, 1, 1).to(video_features.device)
        video_features = torch.cat((video_features, image_newline), dim=1)
        return video_features
