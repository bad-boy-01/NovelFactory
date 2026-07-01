from core.pipeline.stage import PipelineStage, StageResult
from core.domain.assets.execution import ExecutionNode
from core.domain.timeline.models import Timeline, TimelineItem
from core.domain.scene.manifest import ShotManifest
import json
import logging
import os

logger = logging.getLogger(__name__)

class TimelineBuilderStage(PipelineStage):
    def __init__(self, output_dir="workspace"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def get_providers(self) -> list:
        return []
        
    def execute(self, context) -> StageResult:
        shot_manifest = None
        for node in context.execution_nodes.values():
            if isinstance(node.artifact, ShotManifest):
                shot_manifest = node.artifact
                break
                
        if not shot_manifest:
            raise ValueError("TimelineBuilder: Missing ShotManifest.")
            
        items = []
        current_time = 0.0
        
        # Build SRT content simultaneously
        srt_lines = []
        srt_index = 1
        
        for shot in shot_manifest.shots:
            start = current_time
            end = current_time + shot.duration
            
            item = TimelineItem(
                start=start,
                end=end,
                layer=1,
                asset_id=f"asset_{shot.shot_id}",  # To be fulfilled by RenderQueue
                transition="crossfade"
            )
            items.append(item)
            
            # Subtitle formatting
            def format_time(seconds: float) -> str:
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                secs = int(seconds % 60)
                msecs = int((seconds * 1000) % 1000)
                return f"{hours:02d}:{minutes:02d}:{secs:02d},{msecs:03d}"
                
            srt_lines.append(str(srt_index))
            srt_lines.append(f"{format_time(start)} --> {format_time(end)}")
            srt_lines.append(f"Shot: {shot.shot_id} [{shot.camera_type}]")
            srt_lines.append("")
            
            srt_index += 1
            current_time = end
            
        timeline = Timeline(
            duration=current_time,
            items=items,
            schema_version="1.0"
        )
        
        # Output artifacts to disk
        timeline_path = os.path.join(self.output_dir, "Timeline.json")
        srt_path = os.path.join(self.output_dir, "subtitles.srt")
        
        with open(timeline_path, "w", encoding="utf-8") as f:
            f.write(timeline.model_dump_json(indent=2))
            
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(srt_lines))
            
        node = ExecutionNode(artifact=timeline, stage_name="TimelineBuilderStage")
        
        return StageResult(
            artifact=timeline,
            execution_node=node,
            metrics={"duration": timeline.duration, "items": len(items)},
            metadata={"files_generated": [timeline_path, srt_path]}
        )
