class TTSProvider:
    def generate_voice(self, text: str, voice_id: str, output_path: str) -> float:
        raise NotImplementedError

class KokoroTTSProvider(TTSProvider):
    def generate_voice(self, text: str, voice_id: str, output_path: str) -> float:
        """
        Mock Kokoro TTS.
        Returns the duration of the audio in seconds.
        """
        # In reality, this would load the Kokoro model and synthesize the wave
        duration = max(1.0, len(text) * 0.1) # Roughly 10 chars per second
        
        # We would save to output_path here
        with open(output_path, "w") as f:
            f.write("mock audio data")
            
        return duration
