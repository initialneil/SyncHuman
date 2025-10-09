
from SyncHuman.pipelines.SyncHumanOneStagePipeline import SyncHumanOneStagePipeline

      


pipeline = SyncHumanOneStagePipeline.from_pretrained(
    './ckpts/OneStage',
    )
pipeline.SyncHuman_2D3DCrossSpaceDiffusion.enable_xformers_memory_efficient_attention()


pipeline.run(
            image_path='./examples/000.png',
            save_path=f'./outputs/OneStage',
 
            )




    