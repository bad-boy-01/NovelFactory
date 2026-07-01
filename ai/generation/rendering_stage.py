from core.pipeline.stage import PipelineStage, StageResult
from core.domain.asset import ExecutionNode
from core.domain.timeline import Timeline
from core.pipeline.render_queue import RenderQueue
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
        for node in context.execution_nodes.values():
            if isinstance(node.artifact, Timeline):
                timeline = node.artifact
                break
                
        if not timeline:
            raise ValueError("FFmpegAssembly: Missing Timeline.")
            
        logger.info(f"Assembling video from Timeline ({len(timeline.items)} items)...")
        
        # 1. Create concat file for FFmpeg
        concat_file = os.path.join(self.output_dir, "concat.txt")
        images = sorted([f for f in os.listdir(self.output_dir) if f.endswith(".png")])
        
        if not images:
            raise ValueError("FFmpegAssembly: No rendered assets found.")
            
        with open(concat_file, "w", encoding="utf-8") as f:
            for img in images:
                f.write(f"file '{img}'\n")
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
