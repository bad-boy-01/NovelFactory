from core.pipeline.stage import PipelineStage, StageResult
from core.domain.assets.execution import ExecutionNode
from core.domain.scene.manifest import SceneManifest
from core.domain.base import DomainModel
from pydantic import BaseModel
from typing import Dict
import os
import hashlib

class AudioAsset(BaseModel):
    asset_id: str
    path: str
    duration: float
    text: str

class AudioManifest(DomainModel):
    voiceovers: Dict[str, AudioAsset] = {} # Keyed by beat_id

class AudioGenerationStage(PipelineStage):
    def __init__(self, tts_provider=None, output_dir="workspace/audio"):
        self.tts = tts_provider
        self.output_dir = output_dir

    def get_providers(self) -> list:
        return [self.tts] if self.tts else []

    def execute(self, context) -> StageResult:
        if not self.tts:
            from plugins.audio.tts_provider import KokoroTTSProvider
            self.tts = KokoroTTSProvider()
            
        os.makedirs(self.output_dir, exist_ok=True)
            
        scene_manifest = None
        for node in context.execution_nodes.values():
            if isinstance(node.artifact, SceneManifest):
                scene_manifest = node.artifact
                break
                
        if not scene_manifest:
            raise ValueError("AudioGeneration: Missing SceneManifest.")
            
        manifest = AudioManifest()
            
        for scene in scene_manifest.scenes:
            for beat in scene.beats:
                text = beat.description
                
                # Mock generation
                filename = f"vo_{hashlib.md5(text.encode('utf-8')).hexdigest()[:8]}.wav"
                output_path = os.path.join(self.output_dir, filename)
                
                duration = self.tts.generate_voice(text=text, voice_id="default", output_path=output_path)
                
                asset = AudioAsset(
                    asset_id=f"audio_{beat.beat_id}",
                    path=output_path,
                    duration=duration,
                    text=text
                )
                manifest.voiceovers[beat.beat_id] = asset
                
        node = ExecutionNode(artifact=manifest, stage_name="AudioGenerationStage")
        
        return StageResult(
            artifact=manifest,
            execution_node=node,
            metrics={"audio_generated": len(manifest.voiceovers)},
            metadata={}
        )
