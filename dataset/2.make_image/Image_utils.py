# image_utils.py

import os
import cv2
import torch
import numpy as np
from PIL import Image

from diffusers import (
    StableDiffusionXLPipeline, EulerDiscreteScheduler,
    DDIMScheduler, AutoencoderKL, ControlNetModel, StableDiffusionControlNetPipeline,
    StableDiffusionPipeline
)
from huggingface_hub import hf_hub_download

from insightface.app import FaceAnalysis
from insightface.utils import face_align

from uniportrait import inversion
from uniportrait.uniportrait_attention_processor import attn_args
from uniportrait.uniportrait_pipeline import UniPortraitPipeline

# ------------------------------------------------------------
# 기본 환경 설정
# ------------------------------------------------------------
device = "cuda"
torch_dtype = torch.float16

# ------------------------------------------------------------
# 전역 파이프라인들 (None으로 초기화)
# ------------------------------------------------------------
control_pipe = None
uniportrait_pipeline = None
sdxl_pipe = None
sd_v1_pipe = None

# (새로 추가) CosmicMan-SDXL 전역 파이프라인
cosmicman_pipe = None
cosmicman_refiner = None

# ------------------------------------------------------------
# 모델 경로들
# ------------------------------------------------------------
base_model_path = "SG161222/Realistic_Vision_V5.1_noVAE"
vae_model_path = "stabilityai/sd-vae-ft-mse"
controlnet_pose_ckpt = "lllyasviel/control_v11p_sd15_openpose"
image_encoder_path = "models/IP-Adapter/models/image_encoder"
ip_ckpt = "models/IP-Adapter/models/ip-adapter_sd15.bin"
face_backbone_ckpt = "models/glint360k_curricular_face_r101_backbone.bin"
uniportrait_faceid_ckpt = "models/uniportrait-faceid_sd15.bin"
uniportrait_router_ckpt = "models/uniportrait-router_sd15.bin"

# ------------------------------------------------------------
# 얼굴 인식용 FaceAnalysis 초기화
# ------------------------------------------------------------
face_app = FaceAnalysis(providers=["CUDAExecutionProvider"], allowed_modules=["detection"])
face_app.prepare(ctx_id=0, det_size=(640, 640))

# ------------------------------------------------------------
# (1) ControlNet + UniPortrait 파이프라인 로드
# ------------------------------------------------------------
def load_control_uniportrait_pipeline():
    """
    ControlNet + UniPortrait용 파이프라인을 전역 변수에 로드.
    이미 로드되어 있다면 재로드하지 않음.
    """
    global control_pipe, uniportrait_pipeline

    if control_pipe is not None and uniportrait_pipeline is not None:
        return  # 이미 로드됨

    # ControlNet
    pose_controlnet = ControlNetModel.from_pretrained(controlnet_pose_ckpt, torch_dtype=torch_dtype)
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    vae = AutoencoderKL.from_pretrained(vae_model_path, torch_dtype=torch_dtype)

    # ControlNet 파이프라인
    control_pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model_path,
        controlnet=[pose_controlnet],  # list로 multiple controlnets 가능
        torch_dtype=torch_dtype,
        scheduler=noise_scheduler,
        vae=vae,
    )
    control_pipe.to(device)

    # UniPortrait 래퍼 파이프라인
    uniportrait_pipeline = UniPortraitPipeline(
        control_pipe,
        image_encoder_path=image_encoder_path,
        ip_ckpt=ip_ckpt,
        face_backbone_ckpt=face_backbone_ckpt,
        uniportrait_faceid_ckpt=uniportrait_faceid_ckpt,
        uniportrait_router_ckpt=uniportrait_router_ckpt,
        device=device,
        torch_dtype=torch_dtype
    )

# ------------------------------------------------------------
# (2) SDXL(일반 T2I) 파이프라인 로드
# ------------------------------------------------------------
def load_pipeline():
    """
    Stable Diffusion (SDXL) 파이프라인을 전역 변수에 로드.
    """
    global sdxl_pipe
    if sdxl_pipe is not None:
        return  # 이미 로드된 경우

    base = "stabilityai/stable-diffusion-xl-base-1.0"
    repo = "ByteDance/SDXL-Lightning"
    ckpt = "sdxl_lightning_4step_lora.safetensors"

    # 1) SDXL 파이프라인 로드
    sdxl_pipe = StableDiffusionXLPipeline.from_pretrained(
        base,
        torch_dtype=torch.float16,
        variant="fp16"
    ).to(device)

    # 2) LoRA 가중치 로드
    sdxl_pipe.load_lora_weights(hf_hub_download(repo, ckpt))
    sdxl_pipe.fuse_lora()

    # 3) 샘플러 설정
    sdxl_pipe.scheduler = EulerDiscreteScheduler.from_config(
        sdxl_pipe.scheduler.config,
        timestep_spacing="trailing"
    )

# ------------------------------------------------------------
# (3) SD v1.4 파이프라인 로드
# ------------------------------------------------------------
def load_v1_pipeline():
    """
    Stable Diffusion v1.4 파이프라인을 전역 변수에 로드.
    """
    global sd_v1_pipe
    if sd_v1_pipe is not None:
        return  # 이미 로드된 경우

    model_id = "CompVis/stable-diffusion-v1-4"
    sd_v1_pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    sd_v1_pipe.to(device)

# ------------------------------------------------------------
# (새로 추가) CosmicMan-SDXL 파이프라인 로드
# ------------------------------------------------------------
def load_cosmicman_pipeline():
    """
    CosmicMan-SDXL 파이프라인을 전역 변수에 로드.
    """
    global cosmicman_pipe, cosmicman_refiner

    if cosmicman_pipe is not None and cosmicman_refiner is not None:
        return  # 이미 로드된 경우

    from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, UNet2DConditionModel, EulerDiscreteScheduler
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    base_path = "stabilityai/stable-diffusion-xl-base-1.0"
    refiner_path = "stabilityai/stable-diffusion-xl-refiner-1.0"
    unet_path = "cosmicman/CosmicMan-SDXL"

    # Load model
    unet = UNet2DConditionModel.from_pretrained(unet_path, torch_dtype=torch.float16)
    cosmicman_pipe = StableDiffusionXLPipeline.from_pretrained(
        base_path,
        unet=unet,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )
    cosmicman_pipe.scheduler = EulerDiscreteScheduler.from_pretrained(
        base_path,
        subfolder="scheduler",
        torch_dtype=torch.float16
    )
    cosmicman_pipe = cosmicman_pipe.to(device)

    cosmicman_refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        refiner_path,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )
    cosmicman_refiner = cosmicman_refiner.to(device)

# ------------------------------------------------------------
# 얼굴 정렬(전처리) 함수
# ------------------------------------------------------------
def process_faceid_image(pil_faceid_image: Image.Image) -> Image.Image:
    """
    주어진 PIL Image에서 얼굴을 인식해 정렬된 얼굴 이미지를 반환.
    얼굴이 없으면 원본 이미지를 그대로 반환.
    """
    np_faceid_image = np.array(pil_faceid_image.convert("RGB"))
    img = cv2.cvtColor(np_faceid_image, cv2.COLOR_RGB2BGR)

    faces = face_app.get(img)
    if len(faces) == 0:
        print("Warning: No face detected in the image. Using the original image.")
        return pil_faceid_image

    # 가장 큰 얼굴 선택
    faces = sorted(
        faces,
        key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
        reverse=True
    )
    faceid_face = faces[0]

    # 얼굴 정렬
    norm_face = face_align.norm_crop(img, landmark=faceid_face.kps, image_size=224)
    pil_faceid_align_image = Image.fromarray(cv2.cvtColor(norm_face, cv2.COLOR_BGR2RGB))
    return pil_faceid_align_image

# ------------------------------------------------------------
# (4) UniPortrait + ControlNet: 1명 얼굴용
# ------------------------------------------------------------
def generate_uniportrait_image(
    prompt: str,
    negative_prompt: str,
    face_image_path: str,
    output_path: str = "output.png",
    faceid_scale: float = 0.7,
    face_structure_scale: float = 0.1,
    num_samples: int = 1,
    seed: int = None,
    resolution: tuple = (512, 512),
    steps: int = 25,
):
    """
    UniPortrait 이미지 생성 (1명 얼굴).
    """
    load_control_uniportrait_pipeline()

    # 얼굴 정렬
    pil_faceid_image = Image.open(face_image_path).convert("RGB")
    align_image = process_faceid_image(pil_faceid_image)
    cond_faceids = [{"refs": [align_image]}]

    # attn_args 설정
    attn_args.reset()
    attn_args.lora_scale = 1.0 if len(cond_faceids) == 1 else 0.0
    attn_args.multi_id_lora_scale = 0.0
    attn_args.faceid_scale = faceid_scale
    attn_args.num_faceids = len(cond_faceids)
    attn_args.ip_scale = 0.0

    h, w = resolution
    prompt_list = [prompt] * num_samples
    negative_prompt_list = [negative_prompt] * num_samples

    # ControlNet용 더미 이미지
    dummy_image = [torch.zeros([1, 3, h, w])]

    images = uniportrait_pipeline.generate(
        prompt=prompt_list,
        negative_prompt=negative_prompt_list,
        cond_faceids=cond_faceids,
        face_structure_scale=face_structure_scale,
        seed=seed,
        guidance_scale=10,
        num_inference_steps=steps,
        image=dummy_image,
        controlnet_conditioning_scale=[0.0]
    )

    if images:
        images[0].save(output_path)
        return images[0]
    return None

# ------------------------------------------------------------
# (5) UniPortrait 다중 얼굴용 (2~5명)
# ------------------------------------------------------------
def generate_uniportrait_image_with_2_faces(
    prompt: str,
    negative_prompt: str,
    face_image_path_1: str,
    face_image_path_2: str,
    output_path: str = "multi_face_result.png",
    faceid_scale: float = 0.7,
    face_structure_scale: float = 0.3,
    num_samples: int = 1,
    seed: int = None,
    resolution: tuple = (512, 512),
    steps: int = 25
):
    load_control_uniportrait_pipeline()

    align_image_1 = process_faceid_image(Image.open(face_image_path_1).convert("RGB"))
    align_image_2 = process_faceid_image(Image.open(face_image_path_2).convert("RGB"))

    cond_faceids = [
        {"refs": [align_image_1]},
        {"refs": [align_image_2]}
    ]

    attn_args.reset()
    attn_args.lora_scale = 0.0
    attn_args.multi_id_lora_scale = 1.0
    attn_args.faceid_scale = faceid_scale
    attn_args.num_faceids = len(cond_faceids)
    attn_args.ip_scale = 0.0

    h, w = resolution
    prompt_list = [prompt] * num_samples
    negative_prompt_list = [negative_prompt] * num_samples

    images = uniportrait_pipeline.generate(
        prompt=prompt_list,
        negative_prompt=negative_prompt_list,
        cond_faceids=cond_faceids,
        face_structure_scale=face_structure_scale,
        seed=seed,
        guidance_scale=7.5,
        num_inference_steps=steps,
        image=[torch.zeros([1, 3, h, w])],
        controlnet_conditioning_scale=[0.0]
    )

    images[0].save(output_path)
    return images[0]

def generate_uniportrait_image_with_3_faces(
    prompt: str,
    negative_prompt: str,
    face_image_path_1: str,
    face_image_path_2: str,
    face_image_path_3: str,
    output_path: str = "multi_face_result_3.png",
    faceid_scale: float = 0.7,
    face_structure_scale: float = 0.3,
    num_samples: int = 1,
    seed: int = None,
    resolution: tuple = (512, 512),
    steps: int = 25
):
    load_control_uniportrait_pipeline()

    align_image_1 = process_faceid_image(Image.open(face_image_path_1).convert("RGB"))
    align_image_2 = process_faceid_image(Image.open(face_image_path_2).convert("RGB"))
    align_image_3 = process_faceid_image(Image.open(face_image_path_3).convert("RGB"))

    cond_faceids = [
        {"refs": [align_image_1]},
        {"refs": [align_image_2]},
        {"refs": [align_image_3]}
    ]

    attn_args.reset()
    attn_args.lora_scale = 0.0
    attn_args.multi_id_lora_scale = 1.0
    attn_args.faceid_scale = faceid_scale
    attn_args.num_faceids = len(cond_faceids)
    attn_args.ip_scale = 0.0

    h, w = resolution
    prompt_list = [prompt] * num_samples
    negative_prompt_list = [negative_prompt] * num_samples

    images = uniportrait_pipeline.generate(
        prompt=prompt_list,
        negative_prompt=negative_prompt_list,
        cond_faceids=cond_faceids,
        face_structure_scale=face_structure_scale,
        seed=seed,
        guidance_scale=7.5,
        num_inference_steps=steps,
        image=[torch.zeros([1, 3, h, w])],
        controlnet_conditioning_scale=[0.0]
    )

    images[0].save(output_path)
    return images[0]

def generate_uniportrait_image_with_4_faces(
    prompt: str,
    negative_prompt: str,
    face_image_path_1: str,
    face_image_path_2: str,
    face_image_path_3: str,
    face_image_path_4: str,
    output_path: str = "multi_face_result_4.png",
    faceid_scale: float = 0.7,
    face_structure_scale: float = 0.3,
    num_samples: int = 1,
    seed: int = None,
    resolution: tuple = (512, 512),
    steps: int = 25
):
    load_control_uniportrait_pipeline()

    align_image_1 = process_faceid_image(Image.open(face_image_path_1).convert("RGB"))
    align_image_2 = process_faceid_image(Image.open(face_image_path_2).convert("RGB"))
    align_image_3 = process_faceid_image(Image.open(face_image_path_3).convert("RGB"))
    align_image_4 = process_faceid_image(Image.open(face_image_path_4).convert("RGB"))

    cond_faceids = [
        {"refs": [align_image_1]},
        {"refs": [align_image_2]},
        {"refs": [align_image_3]},
        {"refs": [align_image_4]}
    ]

    attn_args.reset()
    attn_args.lora_scale = 0.0
    attn_args.multi_id_lora_scale = 1.0
    attn_args.faceid_scale = faceid_scale
    attn_args.num_faceids = len(cond_faceids)
    attn_args.ip_scale = 0.0

    h, w = resolution
    prompt_list = [prompt] * num_samples
    negative_prompt_list = [negative_prompt] * num_samples

    images = uniportrait_pipeline.generate(
        prompt=prompt_list,
        negative_prompt=negative_prompt_list,
        cond_faceids=cond_faceids,
        face_structure_scale=face_structure_scale,
        seed=seed,
        guidance_scale=7.5,
        num_inference_steps=steps,
        image=[torch.zeros([1, 3, h, w])],
        controlnet_conditioning_scale=[0.0]
    )

    images[0].save(output_path)
    return images[0]

def generate_uniportrait_image_with_5_faces(
    prompt: str,
    negative_prompt: str,
    face_image_path_1: str,
    face_image_path_2: str,
    face_image_path_3: str,
    face_image_path_4: str,
    face_image_path_5: str,
    output_path: str = "multi_face_result_5.png",
    faceid_scale: float = 0.7,
    face_structure_scale: float = 0.3,
    num_samples: int = 1,
    seed: int = None,
    resolution: tuple = (512, 512),
    steps: int = 25
):
    load_control_uniportrait_pipeline()

    align_image_1 = process_faceid_image(Image.open(face_image_path_1).convert("RGB"))
    align_image_2 = process_faceid_image(Image.open(face_image_path_2).convert("RGB"))
    align_image_3 = process_faceid_image(Image.open(face_image_path_3).convert("RGB"))
    align_image_4 = process_faceid_image(Image.open(face_image_path_4).convert("RGB"))
    align_image_5 = process_faceid_image(Image.open(face_image_path_5).convert("RGB"))

    cond_faceids = [
        {"refs": [align_image_1]},
        {"refs": [align_image_2]},
        {"refs": [align_image_3]},
        {"refs": [align_image_4]},
        {"refs": [align_image_5]}
    ]

    attn_args.reset()
    attn_args.lora_scale = 0.0
    attn_args.multi_id_lora_scale = 1.0
    attn_args.faceid_scale = faceid_scale
    attn_args.num_faceids = len(cond_faceids)
    attn_args.ip_scale = 0.0

    h, w = resolution
    prompt_list = [prompt] * num_samples
    negative_prompt_list = [negative_prompt] * num_samples

    images = uniportrait_pipeline.generate(
        prompt=prompt_list,
        negative_prompt=negative_prompt_list,
        cond_faceids=cond_faceids,
        face_structure_scale=face_structure_scale,
        seed=seed,
        guidance_scale=7.5,
        num_inference_steps=steps,
        image=[torch.zeros([1, 3, h, w])],
        controlnet_conditioning_scale=[0.0]
    )

    images[0].save(output_path)
    return images[0]

# ------------------------------------------------------------
# (기존) SD v1.4로 T2I
# ------------------------------------------------------------
def generate_image2(prompt: str, negative_prompt: str, output_filename: str = "output.png"):
    """
    stable-diffusion-v1-4 모델을 사용해 이미지를 생성하는 함수.
    """
    global sd_v1_pipe
    if sd_v1_pipe is None:
        load_v1_pipeline()

    image = sd_v1_pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=50,
        guidance_scale=7.5
    ).images[0]

    image.save(output_filename)


# ------------------------------------------------------------
# (새로 추가) CosmicMan-SDXL로 인물 생성 함수
# ------------------------------------------------------------
def generate_cosmicman_image(
    prompt: str,
    negative_prompt: str,
    output_filename: str = "output.png"
):
    """
    CosmicMan-SDXL 파이프라인으로 인물(혹은 일반) 이미지를 생성하는 함수.
    """
    load_cosmicman_pipeline()

    # 1차 생성 (base)
    image = cosmicman_pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=30,
        guidance_scale=7.5,
        height=1024,
        width=1024,
        output_type="latent"
    ).images[0]

    # 2차 리파이너
    image = cosmicman_refiner(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image[None, :]
    ).images[0]

    image.save(output_filename)
    return image

def generate_landscape_image(
    prompt: str,
    negative_prompt: str,
    output_filename: str = "output.png"
):
    """
    CosmicMan-SDXL 파이프라인으로 인물(혹은 일반) 이미지를 생성하는 함수.
    """
    load_cosmicman_pipeline()

    # 1차 생성 (base)
    image = cosmicman_pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=40,
        guidance_scale=6,
        height=1024,
        width=1024,
        output_type="latent"
    ).images[0]

    # 2차 리파이너
    image = cosmicman_refiner(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image[None, :]
    ).images[0]

    image.save(output_filename)
    return image