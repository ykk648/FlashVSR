from __future__ import annotations

import os
import sys
import gc
from pathlib import Path
from typing import Callable

from .config import Settings
from .video import (
    VideoInfo,
    decode_conditioning_video,
    encode_video_with_audio,
    padded_frame_count,
)


ProgressCallback = Callable[[str], None]


class FlashVSREngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.pipe = None
        self.torch = None
        self.loaded_version: str | None = None

    @property
    def loaded(self) -> bool:
        return self.pipe is not None

    def unload(self) -> None:
        if self.pipe is None:
            return
        self.pipe = None
        self.loaded_version = None
        gc.collect()
        if self.torch is not None and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
            self.torch.cuda.ipc_collect()

    def load(self, model_version: str) -> None:
        if self.loaded and self.loaded_version == model_version:
            return
        if self.loaded:
            self.unload()

        os.environ["CUDA_VISIBLE_DEVICES"] = self.settings.device
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable to the FlashVSR API process")
        model_root = self.settings.model_root_for(model_version)
        if not model_root.is_dir():
            raise FileNotFoundError(f"Model directory is missing: {model_root}")

        wan_dir = self.settings.repo_root / "examples/WanVSR"
        if os.fspath(wan_dir) not in sys.path:
            sys.path.insert(0, os.fspath(wan_dir))
        from diffsynth import FlashVSRTinyLongPipeline, ModelManager
        from utils.TCDecoder import build_tcdecoder
        from utils.utils import Buffer_LQ4x_Proj, Causal_LQ4x_Proj

        manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        manager.load_models(
            [os.fspath(model_root / "diffusion_pytorch_model_streaming_dmd.safetensors")]
        )
        pipe = FlashVSRTinyLongPipeline.from_model_manager(manager, device="cuda")
        projector_class = Buffer_LQ4x_Proj if model_version == "v1" else Causal_LQ4x_Proj
        pipe.denoising_model().LQ_proj_in = projector_class(
            in_dim=3, out_dim=1536, layer_num=1
        ).to("cuda", dtype=torch.bfloat16)
        lq_state = torch.load(
            model_root / "LQ_proj_in.ckpt", map_location="cpu"
        )
        pipe.denoising_model().LQ_proj_in.load_state_dict(lq_state, strict=True)
        pipe.denoising_model().LQ_proj_in.to("cuda")

        pipe.TCDecoder = build_tcdecoder(
            new_channels=[512, 256, 128, 128], new_latent_channels=16 + 768
        )
        decoder_state = torch.load(
            model_root / "TCDecoder.ckpt", map_location="cpu"
        )
        pipe.TCDecoder.load_state_dict(decoder_state, strict=False)

        pipe.to("cuda")
        pipe.enable_vram_management(num_persistent_param_in_dit=None)
        context = torch.load(wan_dir / "prompt_tensor/posi_prompt.pth", map_location="cpu")
        pipe.init_cross_kv(context_tensor=context)
        pipe.load_models_to_device(["dit", "vae"])
        self.pipe = pipe
        self.torch = torch
        self.loaded_version = model_version

    def upscale(
        self,
        input_path: Path,
        output_path: Path,
        info: VideoInfo,
        model_version: str,
        scale: float,
        target_width: int,
        target_height: int,
        progress: ProgressCallback,
        cancelled: Callable[[], bool],
    ) -> str:
        self.load(model_version)
        assert self.pipe is not None and self.torch is not None
        torch = self.torch

        progress("preprocessing")
        lq_video = decode_conditioning_video(
            input_path, info, scale, target_width, target_height
        )
        if cancelled():
            raise InterruptedError("Job cancelled after preprocessing")

        progress("inference")
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        try:
            result = self.pipe(
                prompt="",
                negative_prompt="",
                cfg_scale=1.0,
                num_inference_steps=1,
                seed=0,
                LQ_video=lq_video,
                num_frames=padded_frame_count(info.frames),
                height=target_height,
                width=target_width,
                is_full_block=False,
                if_buffer=True,
                topk_ratio=2.0 * 768 * 1280 / (target_height * target_width),
                kv_ratio=3.0,
                local_range=11,
                color_fix=True,
            )
        except BaseException:
            if hasattr(self.pipe.TCDecoder, "clean_mem"):
                self.pipe.TCDecoder.clean_mem()
            if hasattr(self.pipe.denoising_model(), "LQ_proj_in"):
                self.pipe.denoising_model().LQ_proj_in.clear_cache()
            torch.cuda.empty_cache()
            raise
        del lq_video
        if result.shape[1] < info.frames:
            raise RuntimeError(
                f"Pipeline returned {result.shape[1]} frames, expected {info.frames}"
            )
        result = result[:, : info.frames].contiguous()
        if cancelled():
            del result
            raise InterruptedError("Job cancelled after inference")

        progress("encoding")
        encoder = encode_video_with_audio(result, input_path, output_path, info.fps)
        del result
        torch.cuda.empty_cache()
        return encoder
