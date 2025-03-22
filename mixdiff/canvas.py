from copy import deepcopy
from dataclasses import asdict, dataclass
from enum import Enum
import numpy as np
from numpy import pi, exp, sqrt
import re
import torch
from torchvision.transforms.functional import resize
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from typing import List, Optional, Tuple, Union

from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker


class MaskModes(Enum):
    """Modes in which the influence of diffuser is masked"""

    CONSTANT = "constant"
    GAUSSIAN = "gaussian"
    QUARTIC = "quartic"  # See https://en.wikipedia.org/wiki/Kernel_(statistics)


class RerollModes(Enum):
    """Modes in which the reroll regions operate"""

    RESET = "reset"  # Completely reset the random noise in the region
    EPSILON = "epsilon"  # Alter slightly the latents in the region


@dataclass
class CanvasRegion:
    """Class defining a rectangular region in the canvas"""

    row_init: int  # Region starting row in pixel space (included)
    row_end: int  # Region end row in pixel space (not included)
    col_init: int  # Region starting column in pixel space (included)
    col_end: int  # Region end column in pixel space (not included)
    region_seed: int = None  # Seed for random operations in this region
    noise_eps: float = (
        0.0  # Deviation of a zero-mean gaussian noise to be applied over the latents in this region. Useful for slightly "rerolling" latents
    )

    def __post_init__(self):
        # Initialize arguments if not specified
        if self.region_seed is None:
            self.region_seed = np.random.randint(9999999999)
        # Check coordinates are non-negative
        for coord in [self.row_init, self.row_end, self.col_init, self.col_end]:
            if coord < 0:
                raise ValueError(
                    f"A CanvasRegion must be defined with non-negative indices, found ({self.row_init}, {self.row_end}, {self.col_init}, {self.col_end})"
                )
        # Check coordinates are divisible by 8, else we end up with nasty rounding error when mapping to latent space
        for coord in [self.row_init, self.row_end, self.col_init, self.col_end]:
            if coord // 8 != coord / 8:
                raise ValueError(
                    f"A CanvasRegion must be defined with locations divisible by 8, found ({self.row_init}-{self.row_end}, {self.col_init}-{self.col_end})"
                )
        # Check noise eps is non-negative
        if self.noise_eps < 0:
            raise ValueError(
                f"A CanvasRegion must be defined noises eps non-negative, found {self.noise_eps}"
            )
        # Compute coordinates for this region in latent space
        self.latent_row_init = self.row_init // 8
        self.latent_row_end = self.row_end // 8
        self.latent_col_init = self.col_init // 8
        self.latent_col_end = self.col_end // 8

    @property
    def width(self):
        return self.col_end - self.col_init

    @property
    def height(self):
        return self.row_end - self.row_init

    def get_region_generator(self, device="cpu"):
        """Creates a torch.Generator based on the random seed of this region"""
        # Initialize region generator
        return torch.Generator(device).manual_seed(self.region_seed)

    @property
    def __dict__(self):
        return asdict(self)


@dataclass
class DiffusionRegion(CanvasRegion):
    """Abstract class defining a region where some class of diffusion process is acting"""

    pass


@dataclass
class RerollRegion(CanvasRegion):
    """Class defining a rectangular canvas region in which initial latent noise will be rerolled"""

    reroll_mode: RerollModes = RerollModes.RESET.value


@dataclass
class Text2ImageRegion(DiffusionRegion):
    """Class defining a region where a text guided diffusion process is acting"""

    prompt: str = ""  # Text prompt guiding the diffuser in this region
    guidance_scale: float = (
        7.5  # Guidance scale of the diffuser in this region. If None, randomize
    )
    mask_type: MaskModes = (
        MaskModes.GAUSSIAN.value
    )  # Kind of weight mask applied to this region
    mask_weight: float = 1.0  # Global weights multiplier of the mask
    tokenized_prompt = None  # Tokenized prompt
    encoded_prompt = None  # Encoded prompt

    def __post_init__(self):
        super().__post_init__()
        # Mask weight cannot be negative
        if self.mask_weight < 0:
            raise ValueError(
                f"A Text2ImageRegion must be defined with non-negative mask weight, found {self.mask_weight}"
            )
        # Mask type must be an actual known mask
        if self.mask_type not in [e.value for e in MaskModes]:
            raise ValueError(
                f"A Text2ImageRegion was defined with mask {self.mask_type}, which is not an accepted mask ({[e.value for e in MaskModes]})"
            )
        # Randomize arguments if given as None
        if self.guidance_scale is None:
            self.guidance_scale = np.random.randint(5, 30)
        # Clean prompt
        self.prompt = re.sub(" +", " ", self.prompt).replace("\n", " ")

    def tokenize_prompt(self, tokenizer):
        """Tokenizes the prompt for this diffusion region using a given tokenizer"""
        self.tokenized_prompt = tokenizer(
            self.prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

    def encode_prompt(self, text_encoder, device):
        """Encodes the previously tokenized prompt for this diffusion region using a given encoder"""
        assert self.tokenized_prompt is not None, ValueError(
            "Prompt in diffusion region must be tokenized before encoding"
        )
        self.encoded_prompt = text_encoder(self.tokenized_prompt.input_ids.to(device))[
            0
        ]


@dataclass
class Image2ImageRegion(DiffusionRegion):
    """Class defining a region where an image guided diffusion process is acting"""

    reference_image: torch.FloatTensor = None
    strength: float = 0.8  # Strength of the image

    def __post_init__(self):
        super().__post_init__()
        if self.reference_image is None:
            raise ValueError(
                "Must provide a reference image when creating an Image2ImageRegion"
            )
        if self.strength < 0 or self.strength > 1:
            raise ValueError(
                f"The value of strength should in [0.0, 1.0] but is {self.strength}"
            )
        # Rescale image to region shape
        self.reference_image = resize(
            self.reference_image, size=[self.height, self.width]
        )

    def encode_reference_image(self, encoder, device, generator, cpu_vae=False):
        """Encodes the reference image for this Image2Image region into the latent space"""
        # Place encoder in CPU or not following the parameter cpu_vae
        if cpu_vae:
            # Note here we use mean instead of sample, to avoid moving also generator to CPU, which is troublesome
            self.reference_latents = (
                encoder.cpu().encode(self.reference_image).latent_dist.mean.to(device)
            )
        else:
            self.reference_latents = encoder.encode(
                self.reference_image.to(device)
            ).latent_dist.sample(generator=generator)
        self.reference_latents = 0.18215 * self.reference_latents

    @property
    def __dict__(self):
        # This class requires special casting to dict because of the reference_image tensor. Otherwise it cannot be casted to JSON

        # Get all basic fields from parent class
        super_fields = {
            key: getattr(self, key)
            for key in DiffusionRegion.__dataclass_fields__.keys()
        }
        # Pack other fields
        return {
            **super_fields,
            "reference_image": self.reference_image.cpu().tolist(),
            "strength": self.strength,
        }


@dataclass
class MaskWeightsBuilder:
    """用于计算给定扩散区域的权重张量的辅助类"""

    latent_space_dim: int  # U-net潜在空间的大小
    nbatch: int = 1  # U-net中的批次大小

    def compute_mask_weights(self, region: DiffusionRegion) -> torch.tensor:
        """
        计算给定扩散区域的权重张量

        这个函数是权重掩码系统的核心，根据区域的掩码类型选择合适的权重生成方法。
        掩码权重决定了区域对最终图像的贡献程度，特别是在重叠区域。
        """
        # 不同掩码类型对应的权重构建函数映射
        MASK_BUILDERS = {
            MaskModes.CONSTANT.value: self._constant_weights,  # 常数权重（边界突变）
            MaskModes.GAUSSIAN.value: self._gaussian_weights,  # 高斯权重（平滑渐变）
            MaskModes.QUARTIC.value: self._quartic_weights,  # 四次方权重（有界平滑过渡）
        }
        # 根据区域的掩码类型调用相应的权重生成函数
        return MASK_BUILDERS[region.mask_type](region)

    def _constant_weights(self, region: DiffusionRegion) -> torch.tensor:
        """
        计算给定扩散区域的常数权重

        生成一个全部值相等的权重张量，使区域内的所有像素有相同的贡献。
        这种掩码在区域边界会产生突变，可能导致明显的边界效应。
        """
        # 计算区域在潜在空间中的宽度和高度
        latent_width = region.latent_col_end - region.latent_col_init
        latent_height = region.latent_row_end - region.latent_row_init
        # 创建全1张量并乘以区域的掩码权重系数
        return (
            torch.ones(self.nbatch, self.latent_space_dim, latent_height, latent_width)
            * region.mask_weight
        )

    def _gaussian_weights(self, region: DiffusionRegion) -> torch.tensor:
        """
        生成区域贡献的高斯权重掩码

        在区域中心权重最高，向边缘逐渐降低。
        这种平滑的过渡有助于减少区域间的边界效应，实现更自然的融合。
        """
        # 计算区域在潜在空间中的宽度和高度
        latent_width = region.latent_col_end - region.latent_col_init
        latent_height = region.latent_row_end - region.latent_row_init

        var = 0.01  # 高斯分布的方差，控制衰减速率
        # 计算水平方向的高斯分布（中心最高，边缘最低）
        midpoint = (latent_width - 1) / 2  # -1是因为索引从0到latent_width-1
        x_probs = [
            exp(
                -(x - midpoint)
                * (x - midpoint)
                / (latent_width * latent_width)
                / (2 * var)
            )
            / sqrt(2 * pi * var)
            for x in range(latent_width)
        ]
        # 计算垂直方向的高斯分布
        midpoint = (latent_height - 1) / 2
        y_probs = [
            exp(
                -(y - midpoint)
                * (y - midpoint)
                / (latent_height * latent_height)
                / (2 * var)
            )
            / sqrt(2 * pi * var)
            for y in range(latent_height)
        ]

        # 使用外积计算二维高斯分布，并乘以区域的掩码权重系数
        weights = np.outer(y_probs, x_probs) * region.mask_weight
        # 将权重复制到所有批次和通道维度
        return torch.tile(
            torch.tensor(weights), (self.nbatch, self.latent_space_dim, 1, 1)
        )

    def _quartic_weights(self, region: DiffusionRegion) -> torch.tensor:
        """
        生成区域贡献的四次方权重掩码

        四次方核函数在扩散区域上有有界支撑，并且向区域边界平滑衰减。
        与高斯掩码相比，四次方掩码在边界处降为零，确保了区域外完全没有影响。
        """
        quartic_constant = 15.0 / 16.0  # 四次方核函数的常数

        # 计算列方向的四次方核权重
        # 将区域像素映射到[-0.995, 0.995]区间，以便应用核函数
        support = (
            np.array(range(region.latent_col_init, region.latent_col_end))
            - region.latent_col_init
        ) / (region.latent_col_end - region.latent_col_init - 1) * 1.99 - (1.99 / 2.0)
        # 应用四次方核函数：K(u) = (15/16)(1-u²)²
        x_probs = quartic_constant * np.square(1 - np.square(support))

        # 计算行方向的四次方核权重
        support = (
            np.array(range(region.latent_row_init, region.latent_row_end))
            - region.latent_row_init
        ) / (region.latent_row_end - region.latent_row_init - 1) * 1.99 - (1.99 / 2.0)
        y_probs = quartic_constant * np.square(1 - np.square(support))

        # 使用外积计算二维四次方分布，并乘以区域的掩码权重系数
        weights = np.outer(y_probs, x_probs) * region.mask_weight
        # 将权重复制到所有批次和通道维度
        return torch.tile(
            torch.tensor(weights), (self.nbatch, self.latent_space_dim, 1, 1)
        )


class StableDiffusionCanvasPipeline(DiffusionPipeline):
    """Stable Diffusion pipeline that mixes several diffusers in the same canvas"""

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler],
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )

    def decode_latents(self, latents, cpu_vae=False):
        """Decodes a given array of latents into pixel space"""
        # scale and decode the image latents with vae
        if cpu_vae:
            lat = deepcopy(latents).cpu()
            vae = deepcopy(self.vae).cpu()
        else:
            lat = latents
            vae = self.vae

        lat = 1 / 0.18215 * lat
        image = vae.decode(lat).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        return self.numpy_to_pil(image)

    def get_latest_timestep_img2img(self, num_inference_steps, strength):
        """Finds the latest timesteps where an img2img strength does not impose latents anymore"""
        # get the original timestep using init_timestep
        offset = self.scheduler.config.get("steps_offset", 0)
        init_timestep = int(num_inference_steps * (1 - strength)) + offset
        init_timestep = min(init_timestep, num_inference_steps)

        t_start = min(
            max(num_inference_steps - init_timestep + offset, 0),
            num_inference_steps - 1,
        )
        latest_timestep = self.scheduler.timesteps[t_start]

        return latest_timestep

    @torch.no_grad()
    def __call__(
        self,
        canvas_height: int,
        canvas_width: int,
        regions: List[DiffusionRegion],
        num_inference_steps: Optional[int] = 50,
        seed: Optional[int] = 12345,
        reroll_regions: Optional[List[RerollRegion]] = None,
        cpu_vae: Optional[bool] = False,
        decode_steps: Optional[bool] = False,
    ):
        """
        执行多区域扩散混合的主函数

        参数:
            canvas_height: 画布高度（像素）
            canvas_width: 画布宽度（像素）
            regions: 扩散区域列表，每个区域可以是Text2Image或Image2Image类型
            num_inference_steps: 扩散步骤数
            seed: 随机种子，用于初始化潜在空间的噪声
            reroll_regions: 需要重新生成随机噪声的区域列表
            cpu_vae: 是否在CPU上执行VAE解码（对于大图像有助于节省GPU内存）
            decode_steps: 是否保存中间步骤的图像
        """
        if reroll_regions is None:
            reroll_regions = []
        batch_size = 1

        if decode_steps:
            steps_images = []

        # 准备调度器，设置时间步长
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)

        # 按类型分类扩散区域
        text2image_regions = [
            region for region in regions if isinstance(region, Text2ImageRegion)
        ]
        image2image_regions = [
            region for region in regions if isinstance(region, Image2ImageRegion)
        ]

        # 准备文本嵌入向量
        for region in text2image_regions:
            region.tokenize_prompt(self.tokenizer)  # 将提示文本转换为token
            region.encode_prompt(
                self.text_encoder, self.device
            )  # 编码提示文本为特征向量

        # 创建初始噪声潜在变量
        # 潜在空间尺寸比像素空间小8倍（VAE的下采样因子）
        latents_shape = (
            batch_size,
            self.unet.config.in_channels,
            canvas_height // 8,
            canvas_width // 8,
        )
        generator = torch.Generator(self.device).manual_seed(seed)
        init_noise = torch.randn(latents_shape, generator=generator, device=self.device)

        # 如果有reroll区域，重置这些区域的噪声（完全替换）
        for region in reroll_regions:
            if region.reroll_mode == RerollModes.RESET.value:
                region_shape = (
                    latents_shape[0],
                    latents_shape[1],
                    region.latent_row_end - region.latent_row_init,
                    region.latent_col_end - region.latent_col_init,
                )
                init_noise[
                    :,
                    :,
                    region.latent_row_init : region.latent_row_end,
                    region.latent_col_init : region.latent_col_end,
                ] = torch.randn(
                    region_shape,
                    generator=region.get_region_generator(self.device),
                    device=self.device,
                )

        # 为区域添加epsilon噪声（轻微扰动，而非完全替换）
        all_eps_rerolls = regions + [
            r for r in reroll_regions if r.reroll_mode == RerollModes.EPSILON.value
        ]
        for region in all_eps_rerolls:
            if region.noise_eps > 0:
                region_noise = init_noise[
                    :,
                    :,
                    region.latent_row_init : region.latent_row_end,
                    region.latent_col_init : region.latent_col_end,
                ]
                eps_noise = (
                    torch.randn(
                        region_noise.shape,
                        generator=region.get_region_generator(self.device),
                        device=self.device,
                    )
                    * region.noise_eps
                )
                init_noise[
                    :,
                    :,
                    region.latent_row_init : region.latent_row_end,
                    region.latent_col_init : region.latent_col_end,
                ] += eps_noise

        # 根据调度器要求缩放初始噪声
        latents = init_noise * self.scheduler.init_noise_sigma

        # 为Text2Image区域准备无条件嵌入向量（用于分类器自由引导）
        for region in text2image_regions:
            max_length = region.tokenized_prompt.input_ids.shape[-1]
            uncond_input = self.tokenizer(
                [""] * batch_size,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(self.device)
            )[0]

            # 为实现分类器自由引导，需要两次前向传播
            # 这里将无条件嵌入和文本嵌入连接成一个批次，避免做两次前向传播
            region.encoded_prompt = torch.cat(
                [uncond_embeddings, region.encoded_prompt]
            )

        # 准备Image2Image区域的潜在变量
        for region in image2image_regions:
            region.encode_reference_image(
                self.vae, device=self.device, generator=generator
            )

        # 为每个区域准备权重掩码
        # 这些掩码决定了每个区域对最终图像的贡献程度，尤其是在重叠区域
        mask_builder = MaskWeightsBuilder(
            latent_space_dim=self.unet.config.in_channels, nbatch=batch_size
        )
        mask_weights = [
            mask_builder.compute_mask_weights(region).to(self.device)
            for region in text2image_regions
        ]

        # 开始扩散时间步循环
        for i, t in tqdm(enumerate(self.scheduler.timesteps)):
            # 为每个区域执行扩散步骤
            noise_preds_regions = []

            # 处理所有text2image区域
            for region in text2image_regions:
                # 提取当前区域的潜在变量
                region_latents = latents[
                    :,
                    :,
                    region.latent_row_init : region.latent_row_end,
                    region.latent_col_init : region.latent_col_end,
                ]
                # 为分类器自由引导复制潜在变量（一份无条件，一份有条件）
                latent_model_input = torch.cat([region_latents] * 2)
                # 根据调度器规则缩放模型输入
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )
                # 预测噪声残差
                noise_pred = self.unet(
                    latent_model_input, t, encoder_hidden_states=region.encoded_prompt
                )["sample"]
                # 执行引导（classifier-free guidance）
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                # 将无条件预测与文本引导预测按照引导尺度混合
                noise_pred_region = noise_pred_uncond + region.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
                noise_preds_regions.append(noise_pred_region)

            # 合并所有区域的噪声预测 - 这是实现无缝混合的核心部分
            noise_pred = torch.zeros(
                latents.shape, device=self.device
            )  # 创建全零张量存储合并的噪声预测
            contributors = torch.zeros(
                latents.shape, device=self.device
            )  # 创建全零张量记录每个位置的贡献者数量

            # 将每个区域的贡献添加到整体潜在变量中
            for region, noise_pred_region, mask_weights_region in zip(
                text2image_regions, noise_preds_regions, mask_weights
            ):
                # 将区域的噪声预测乘以权重掩码后添加到整体噪声预测中
                # 掩码确保在区域边界处有平滑过渡，避免拼接痕迹
                noise_pred[
                    :,
                    :,
                    region.latent_row_init : region.latent_row_end,
                    region.latent_col_init : region.latent_col_end,
                ] += (
                    noise_pred_region * mask_weights_region
                )
                # 同时记录每个位置的权重总和，用于后续归一化
                contributors[
                    :,
                    :,
                    region.latent_row_init : region.latent_row_end,
                    region.latent_col_init : region.latent_col_end,
                ] += mask_weights_region

            # 对重叠区域求平均值，确保贡献正确混合
            # 这一步是实现无缝融合的关键 - 每个位置的噪声预测由权重加权平均
            noise_pred /= contributors
            # 处理可能的NaN值（如果某个位置没有任何DiffusionRegion覆盖）
            noise_pred = torch.nan_to_num(noise_pred)

            # 使用调度器计算下一个噪声样本 x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            # 处理Image2Image区域：根据强度参数覆盖调度器生成的潜在变量
            for region in image2image_regions:
                # 获取图像引导的最后一个时间步（由强度参数决定）
                influence_step = self.get_latest_timestep_img2img(
                    num_inference_steps, region.strength
                )
                # 只在影响时间步之前覆盖（由区域的强度决定）
                if t > influence_step:
                    timestep = t.repeat(batch_size)
                    # 获取区域的初始噪声
                    region_init_noise = init_noise[
                        :,
                        :,
                        region.latent_row_init : region.latent_row_end,
                        region.latent_col_init : region.latent_col_end,
                    ]
                    # 将参考图像的潜在变量与噪声结合
                    region_latents = self.scheduler.add_noise(
                        region.reference_latents, region_init_noise, timestep
                    )
                    # 覆盖当前区域的潜在变量
                    latents[
                        :,
                        :,
                        region.latent_row_init : region.latent_row_end,
                        region.latent_col_init : region.latent_col_end,
                    ] = region_latents

            # 如果需要，保存中间步骤的图像
            if decode_steps:
                steps_images.append(self.decode_latents(latents, cpu_vae))

        # 缩放并解码最终的潜在变量到像素空间
        image = self.decode_latents(latents, cpu_vae)

        # 准备输出
        output = {"sample": image}
        if decode_steps:
            output = {**output, "steps_images": steps_images}
        return output
