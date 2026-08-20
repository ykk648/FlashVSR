from __future__ import annotations

import json
import math
import os
import subprocess
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    frames: int
    fps: str
    has_audio: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def probe_video(path: Path) -> VideoInfo:
    command = [
        "ffprobe", "-v", "error", "-count_frames", "-show_streams",
        "-of", "json", os.fspath(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    streams = json.loads(result.stdout).get("streams", [])
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError("Input contains no video stream")

    frame_value = video.get("nb_read_frames") or video.get("nb_frames")
    if not frame_value or str(frame_value) == "N/A":
        raise ValueError("ffprobe could not determine the input frame count")
    fps_value = video.get("avg_frame_rate") or video.get("r_frame_rate")
    fps = Fraction(str(fps_value))
    if fps <= 0:
        raise ValueError("Input has an invalid frame rate")

    return VideoInfo(
        width=int(video["width"]),
        height=int(video["height"]),
        frames=int(frame_value),
        fps=f"{fps.numerator}/{fps.denominator}",
        has_audio=any(stream.get("codec_type") == "audio" for stream in streams),
    )


def target_dimensions(width: int, height: int, scale: float) -> tuple[int, int]:
    if not math.isfinite(scale) or scale < 1:
        raise ValueError("scale must be a finite number greater than or equal to 1")
    target_width = int(round(width * scale)) // 128 * 128
    target_height = int(round(height * scale)) // 128 * 128
    if target_width <= 0 or target_height <= 0:
        raise ValueError("Scaled dimensions are too small")
    return target_width, target_height


def padded_frame_count(frames: int) -> int:
    """Return the smallest 8n+1 input that decodes at least `frames` outputs."""
    if frames <= 0:
        raise ValueError("frames must be positive")
    return math.ceil((frames + 3) / 8) * 8 + 1


def decode_conditioning_video(
    path: Path,
    info: VideoInfo,
    scale: float,
    target_width: int,
    target_height: int,
) -> "torch.Tensor":
    import torch

    padded_frames = padded_frame_count(info.frames)
    tensor = torch.empty(
        (1, 3, padded_frames, target_height, target_width),
        dtype=torch.bfloat16,
        device="cpu",
    )
    scaled_width = int(round(info.width * scale))
    scaled_height = int(round(info.height * scale))
    vf = (
        f"scale={scaled_width}:{scaled_height}:flags=bicubic,"
        f"crop={target_width}:{target_height}"
    )
    command = [
        "ffmpeg", "-v", "error", "-i", os.fspath(path),
        "-map", "0:v:0", "-vf", vf, "-fps_mode", "passthrough",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout is not None
    frame_bytes = target_width * target_height * 3
    last_frame = None
    try:
        for index in range(info.frames):
            raw = process.stdout.read(frame_bytes)
            if len(raw) != frame_bytes:
                stderr = process.stderr.read().decode("utf-8", "replace") if process.stderr else ""
                raise RuntimeError(
                    f"ffmpeg returned a short frame at index {index}: {stderr.strip()}"
                )
            array = np.frombuffer(raw, dtype=np.uint8).reshape(
                target_height, target_width, 3
            )
            frame = torch.from_numpy(array.copy()).permute(2, 0, 1)
            tensor[0, :, index].copy_(
                frame.to(dtype=torch.bfloat16).div_(127.5).sub_(1)
            )
            last_frame = tensor[0, :, index]
        return_code = process.wait()
        if return_code != 0:
            stderr = process.stderr.read().decode("utf-8", "replace") if process.stderr else ""
            raise RuntimeError(f"ffmpeg decode failed: {stderr.strip()}")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()

    if last_frame is None:
        raise RuntimeError("Input contains no decodable frames")
    for index in range(info.frames, padded_frames):
        tensor[0, :, index].copy_(last_frame)
    return tensor


def _write_frames_to_ffmpeg(
    frames: "torch.Tensor",
    output_path: Path,
    fps: str,
    encoder: str,
) -> None:
    import torch

    _, frame_count, height, width = frames.shape
    codec_args = (
        ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "18"]
        if encoder == "h264_nvenc"
        else ["-c:v", "libx264", "-preset", "medium", "-crf", "18"]
    )
    command = [
        "ffmpeg", "-y", "-v", "error", "-f", "rawvideo",
        "-pix_fmt", "rgb24", "-s:v", f"{width}x{height}",
        "-r", fps, "-i", "pipe:0", "-an", *codec_args,
        "-frames:v", str(frame_count), "-pix_fmt", "yuv420p",
        os.fspath(output_path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for index in range(frame_count):
            frame = (
                frames[:, index]
                .permute(1, 2, 0)
                .float()
                .add_(1)
                .mul_(127.5)
                .clamp_(0, 255)
                .to(dtype=torch.uint8)
                .numpy()
            )
            process.stdin.write(frame.tobytes())
        process.stdin.close()
        return_code = process.wait()
        stderr = process.stderr.read().decode("utf-8", "replace") if process.stderr else ""
        if return_code != 0:
            raise RuntimeError(stderr.strip() or f"ffmpeg {encoder} failed")
    except BaseException:
        if process.poll() is None:
            process.kill()
            process.wait()
        raise


def encode_video_with_audio(
    frames: "torch.Tensor",
    input_path: Path,
    output_path: Path,
    fps: str,
) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    video_only = output_path.with_suffix(".video-only.mp4")
    final_temp = output_path.with_suffix(".partial.mp4")
    for stale in (video_only, final_temp):
        stale.unlink(missing_ok=True)

    encoder = "h264_nvenc"
    try:
        try:
            _write_frames_to_ffmpeg(frames, video_only, fps, encoder)
        except (OSError, RuntimeError):
            encoder = "libx264"
            video_only.unlink(missing_ok=True)
            _write_frames_to_ffmpeg(frames, video_only, fps, encoder)

        command = [
            "ffmpeg", "-y", "-v", "error",
            "-i", os.fspath(video_only), "-i", os.fspath(input_path),
            "-map", "0:v:0", "-map", "1:a:0?", "-c", "copy",
            "-map_metadata", "1", "-movflags", "+faststart",
            os.fspath(final_temp),
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        final_temp.replace(output_path)
        return encoder
    finally:
        video_only.unlink(missing_ok=True)
        final_temp.unlink(missing_ok=True)
