
from SyncHuman.pipelines.SyncHumanTwoStage import SyncHumanTwoStagePipeline



pipeline=SyncHumanTwoStagePipeline.from_pretrained('./ckpts/SecondStage')
pipeline.cuda()
pipeline.run(image_path='./outputs/OneStage',outpath='./outputs/SecondStage') 


