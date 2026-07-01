import subprocess
from pathlib import Path
from core.domain.assets.execution import FrameManifest
from plugins.interfaces import VideoRendererProvider

class FFmpegVideoRenderer(VideoRendererProvider):
    def render_video(self, manifest: FrameManifest, audio_paths: list[Path], output_path: Path) -> Path:
        """
        Consumes a strictly ordered FrameManifest, normalizes the images, 
        and invokes FFmpeg to stitch them together.
        """
        if not manifest.frames:
            raise ValueError("FrameManifest is empty. Cannot render video.")
            
        output_path.parent.mkdir(parents=True, exist_ok=True)
            
        # 1. Create a concat list file
        concat_file = output_path.parent / "concat.txt"
        with open(concat_file, "w") as f:
            for entry in manifest.frames:
                # Normalizing paths for ffmpeg (forward slashes even on Windows)
                safe_path = entry.image_path.absolute().as_posix()
                f.write(f"file '{safe_path}'\n")
                f.write(f"duration 2.0\n")  # hardcoded 2s per frame for MVP
        
        # 2. Invoke FFmpeg
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", 
            "-i", str(concat_file),
            "-vsync", "vfr", "-pix_fmt", "yuv420p",
            str(output_path)
        ]
        
        try:
            print("[FFMPEG] Rendering video...")
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"[FFMPEG] Render complete: {output_path}")
        except subprocess.CalledProcessError as e:
            print("[FFMPEG] Render failed.")
            print(e.stderr.decode())
            raise e
            
        return output_path
