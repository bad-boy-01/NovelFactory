from core.pipeline.stage import PipelineStage, StageResult
from core.domain.assets.execution import ExecutionNode
from core.domain.timeline.models import Timeline
from core.rendering.render_queue import RenderQueue
import os
import json
import logging
import subprocess

logger = logging.getLogger(__name__)

class FFmpegAssemblyStage(PipelineStage):
    def __init__(self, output_dir="workspace"):
        self.output_dir = output_dir

    def get_providers(self) -> list:
        return []
        
    def execute(self, context) -> StageResult:
        timeline = None
        for node in context.execution_nodes:
            if isinstance(node.artifact, Timeline):
                timeline = node.artifact
                break
                
        if not timeline:
            raise ValueError("FFmpegAssembly: Missing Timeline.")
            
        # Get count of video clips
        video_track = timeline.tracks.get("video_main")
        clip_count = len(video_track.clips) if video_track else 0
        logger.info(f"Assembling video from Timeline ({clip_count} items)...")
        
        # 1. Create concat file for FFmpeg
        concat_file = os.path.join(self.output_dir, "concat.txt")
        
        import glob
        # Images are saved by ImageStage in workspace/Shot_XXX/image.png
        images = sorted(glob.glob(os.path.join(self.output_dir, "Shot_*", "image.png")))
        
        if not images:
            raise ValueError("FFmpegAssembly: No rendered assets found.")
            
        with open(concat_file, "w", encoding="utf-8") as f:
            for img in images:
                # Use forward slashes for FFmpeg path compatibility
                img_path = img.replace("\\", "/")
                f.write(f"file '{img_path}'\n")
                f.write(f"duration 4.0\n") # Static duration for demo
                
        output_video = os.path.join(self.output_dir, "final_video.mp4")
        srt_file = os.path.join(self.output_dir, "subtitles.srt")
        
        # 2. FFmpeg command: Subtitle multiplexing
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file
        ]
        
        if os.path.exists(srt_file):
            # Escape path for FFmpeg filter
            srt_escaped = srt_file.replace("\\", "/").replace(":", "\\:")
            cmd.extend(["-vf", f"subtitles={srt_escaped}"])
            
        cmd.extend([
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            output_video
        ])
        
        logger.info(f"Running FFmpeg: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Successfully rendered video: {output_video}")
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg failed: {e.stderr.decode()}")
            raise e
            
        # 3. Cleanup: Delete intermediate PNGs and VACUUM
        logger.info("Cleaning up intermediate assets...")
        for img in images:
            os.remove(os.path.join(self.output_dir, img))
            
        RenderQueue().vacuum()
        logger.info("RenderQueue vacuumed.")
        
        node = ExecutionNode(artifact=timeline, stage_name="FFmpegAssemblyStage")
        
        return StageResult(
            artifact=timeline,
            execution_node=node,
            metrics={"video_path": output_video},
            metadata={}
        )
