class StoryBibleGeneratorStage:

    def __init__(self, llm_provider):
        self.llm = llm_provider

    def execute(self, context):

        schema = {
            "characters": [
                {
                    "name": "string",
                    "appearance": "string",
                    "outfit": "string"
                }
            ]
        }

        prompt = f"""
Extract structured character data from this story:

{getattr(context, 'raw_text', '')}
"""

        result = self.llm.generate_json(prompt, schema)

        context.story_bible = result

        return {
            "identity_blob": result,
            "frames": []
        }
