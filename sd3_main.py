import ipdb

st = ipdb.set_trace
import builtins
import time
import os

builtins.st = ipdb.set_trace
from dataclasses import dataclass, field
import prompts as prompts_file
import numpy as np
from transformers import HfArgumentParser

from config.alignprop_config import AlignPropConfig
from alignprop_trainer_sd3 import AlignPropTrainerSD3
from modeling_sd3 import AlignPropDiffusionPipeline3
from trl.models.auxiliary_modules import aesthetic_scorer


@dataclass
class ScriptArguments:
    pretrained_model: str = field(
        default="runwayml/stable-diffusion-v1-5", metadata={"help": "the pretrained model to use"}
    )
    pretrained_revision: str = field(default="main", metadata={"help": "the pretrained model revision to use"})
    use_lora: bool = field(default=True, metadata={"help": "Whether to use LoRA."})


def image_outputs_logger(image_pair_data, global_step, accelerate_logger):
    # For the sake of this example, we will only log the last batch of images
    # and associated data
    result = {}
    images, prompts = [image_pair_data["images"], image_pair_data["prompts"]]
    for i, image in enumerate(images[:4]):
        prompt = prompts[i]
        result[f"{prompt}"] = image.unsqueeze(0).float()
    accelerate_logger.log_images(
        result,
        step=global_step,
    )


if __name__ == "__main__":
    # parser = HfArgumentParser((ScriptArguments, AlignPropConfig))
    # script_args, training_args = parser.parse_args_into_dataclasses()
    project_dir = f"alignprop_{int(time.time())}"
    os.makedirs(f"checkpoints/{project_dir}", exist_ok=True)

    # training_args.project_kwargs = {
    #     "logging_dir": "./logs",
    #     "automatic_checkpoint_naming": True,
    #     "total_limit": 5,
    #     "project_dir": f"checkpoints/{project_dir}",
    # }
    config = AlignPropConfig(backprop_strategy="fixed", tracker_project_name="stable_diffusion_training", num_epochs=20, train_gradient_accumulation_steps=4, project_kwargs= {
        "logging_dir": "./logs",
        "automatic_checkpoint_naming": True,
        "total_limit": 5,
        "project_dir": f"checkpoints/{project_dir}",
    })
    prompt_fn = getattr(prompts_file, 'hps_v2_all')

    pipeline = AlignPropDiffusionPipeline3(
         "stabilityai/stable-diffusion-3.5-medium",
        use_lora=False,
    )
    trainer = AlignPropTrainerSD3(
        config=config,
        prompt_function=prompt_fn,
        sd_pipeline=pipeline,
        image_samples_hook=image_outputs_logger,
    )

    trainer.train()