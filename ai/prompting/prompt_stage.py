from core.prompt.ast import PromptAST


class PromptBuilderStage:

    def __init__(self, compiler):
        self.compiler = compiler

    def execute(self, context):

        ast = PromptAST(
            character=context.story_bible["characters"][0]["name"],
            outfit=context.story_bible["characters"][0]["outfit"],
            scene=context.current_scene.description,
            camera="medium shot, centered",
            lighting="soft cinematic lighting",
            style="anime cinematic, high detail",
            negative="low quality, blurry, distorted"
        )

        compiled_prompt = self.compiler.compile(ast)

        context.current_prompt = compiled_prompt

        return {
            "prompt": compiled_prompt,
            "identity_blob": context.story_bible
        }
