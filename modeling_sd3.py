# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This module extends the original *modeling_sd_base.py* to add **Stable Diffusion 3 / 3.5**
compatibility. The key design decisions are:

*   **Scheduler** – SD‑3.5 uses ``FlowMatchEulerDiscreteScheduler``.  We therefore keep a
generic ``scheduler_step`` wrapper that 
    − delegates to ``DDIMScheduler.step`` (legacy SD‑1.x) **unchanged**;
    − delegates to ``FlowMatchEulerDiscreteScheduler.step`` and synthesises a constant
      ``log_prob = 0`` tensor (Euler update is deterministic, hence latent likelihood w.r.t
      additive Gaussian noise is 0 in log‑prob terms).

*   **Pipeline hooks** – the original ``pipeline_step``/``pipeline_step_with_grad`` are
    preserved for SD‑1.x.  New variants suffixed **_sd3** replicate the SD‑3 denoising
    loop (copied and abridged from ``StableDiffusion3Pipeline.__call__``) while retaining
    AlignProp‑specific features (truncated/randomised back‑prop, gradient checkpointing,
    log‑prob tracking).

*   **Default pipeline wrapper** – ``DefaultDDPOStableDiffusion3Pipeline`` mirrors the
    SD‑1.x wrapper but binds a SD‑3.5 checkpoint, swaps the scheduler, and exposes the
    new step helpers.

This file is *drop‑in*: import it **instead of** the original ``modeling_sd_base`` when
working with SD‑3.5.
"""
from __future__ import annotations
import warnings
import contextlib
import os
import random

from dataclasses import dataclass
from typing import Any, Callable, Optional, Union, List, Dict

import torch
import torch.utils.checkpoint as checkpoint
from diffusers import (
    DDIMScheduler,
    FlowMatchEulerDiscreteScheduler,
    StableDiffusion3Pipeline, SD3Transformer2DModel
)
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    calculate_shift, retrieve_timesteps,
)
from transformers.utils import is_peft_available
from trl.core import randn_tensor
from trl.models.sd_utils import convert_state_dict_to_diffusers
from trl.models.modeling_sd_base import (
    DDPOPipelineOutput,
    DDPOSchedulerOutput,
    _left_broadcast,  # re‑export – used by legacy branch
)

if is_peft_available():
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict

# -------------------------------------------------------------------------------------
# 1. Scheduler helper – now supports both DDIM & Flow‑Match Euler
# -------------------------------------------------------------------------------------

def scheduler_step(
    scheduler: Union[DDIMScheduler, FlowMatchEulerDiscreteScheduler],
    model_output: torch.FloatTensor,
    timestep: Union[int, torch.FloatTensor],
    sample: torch.FloatTensor,
    eta: float = 0.0,
    **extra_kwargs,
) -> DDPOSchedulerOutput:
    """Unified AlignProp wrapper returning (prev_latents, log_prob).

    * **DDIM** – identical to the original implementation (code inlined for clarity).
    * **Flow‑Match Euler** – calls native ``scheduler.step``.  The update is deterministic
      (no additive variance term → no latent stochasticity), so we define
      ``log_prob = 0``.
    """
    if isinstance(scheduler, FlowMatchEulerDiscreteScheduler):
        # Native step → tuple(prev_sample,) when return_dict=False.
        prev_latents = scheduler.step(model_output, timestep, sample, **extra_kwargs, return_dict=False)[0]
        log_prob = torch.zeros(sample.shape[0], device=sample.device, dtype=sample.dtype)
        return DDPOSchedulerOutput(prev_latents, log_prob)

    if isinstance(scheduler, DDIMScheduler):
        # --- original (slightly refactored) DDIM branch --------------------------------
        prev_timestep = timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
        prev_timestep = torch.clamp(prev_timestep, 0, scheduler.config.num_train_timesteps - 1)

        alpha_prod_t = _left_broadcast(scheduler.alphas_cumprod.gather(0, timestep.cpu()), sample.shape).to(sample.device)
        alpha_prod_t_prev = _left_broadcast(
            torch.where(prev_timestep.cpu() >= 0,
                        scheduler.alphas_cumprod.gather(0, prev_timestep.cpu()),
                        scheduler.final_alpha_cumprod),
            sample.shape,
        ).to(sample.device)
        beta_prod_t = 1 - alpha_prod_t

        if scheduler.config.prediction_type == "epsilon":
            pred_x0 = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
            pred_eps = model_output
        elif scheduler.config.prediction_type == "sample":
            pred_x0 = model_output
            pred_eps = (sample - alpha_prod_t.sqrt() * pred_x0) / beta_prod_t.sqrt()
        elif scheduler.config.prediction_type == "v_prediction":
            pred_x0 = alpha_prod_t.sqrt() * sample - beta_prod_t.sqrt() * model_output
            pred_eps = alpha_prod_t.sqrt() * model_output + beta_prod_t.sqrt() * sample
        else:
            raise ValueError("Unsupported prediction_type")

        if scheduler.config.clip_sample:
            pred_x0 = pred_x0.clamp(-scheduler.config.clip_sample_range, scheduler.config.clip_sample_range)

        variance = _left_broadcast(
            scheduler._get_variance(timestep, prev_timestep),
            sample.shape,
        ).to(sample.device)
        std_dev_t = eta * variance.sqrt()
        pred_dir = (1 - alpha_prod_t_prev - std_dev_t**2).sqrt() * pred_eps
        prev_latents_mean = alpha_prod_t_prev.sqrt() * pred_x0 + pred_dir

        if eta > 0:
            noise = randn_tensor(model_output.shape, generator=extra_kwargs.get("generator"), device=model_output.device, dtype=model_output.dtype)
            prev_latents = prev_latents_mean + std_dev_t * noise
        else:
            prev_latents = prev_latents_mean

        log_prob = -((prev_latents.detach() - prev_latents_mean) ** 2) / (2 * std_dev_t**2 + 1e-8)
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
        return DDPOSchedulerOutput(prev_latents.type(sample.dtype), log_prob)

    raise TypeError(f"Unsupported scheduler type {type(scheduler)}")


# -------------------------------------------------------------------------------------
# 2. SD‑3.5 pipeline helpers (no‑grad + with‑grad) – truncated‑BP aware
# -------------------------------------------------------------------------------------

def _encode_prompt_sd3(pipeline: StableDiffusion3Pipeline, prompt, negative_prompt, num_images_per_prompt, device):
    """Minimal wrapper that re‑uses the same text for all three encoders (CLIP‑L, CLIP‑G, T5)."""
    return pipeline.encode_prompt(
        prompt=prompt,
        prompt_2=prompt,
        prompt_3=prompt,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt,
        negative_prompt_3=negative_prompt,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        do_classifier_free_guidance=True,
    )


@torch.no_grad()
def pipeline_step_sd3(
    pipeline: StableDiffusion3Pipeline,
    prompt: Optional[Union[str, List[str]]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    prompt_3: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    callback_steps: int = 1,
    num_inference_steps: int = 28,
    guidance_scale: float = 7.0,

    latents: Optional[torch.FloatTensor] = None,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    negative_prompt: Union[str, list[str]] | None = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    negative_prompt_3: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    sigmas: Optional[List[float]] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    ip_adapter_image: Optional[PipelineImageInput] = None,
    ip_adapter_image_embeds: Optional[torch.Tensor] = None,
    output_type: str = "pil",
    return_dict: bool = True,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    clip_skip: Optional[int] = None,
    max_sequence_length: int = 256,
    skip_guidance_layers: List[int] = None,
    skip_layer_guidance_scale: float = 2.8,
    skip_layer_guidance_stop: float = 0.2,
    skip_layer_guidance_start: float = 0.01,
    mu: Optional[float] = None,
):
    do_classifier_free_guidance = guidance_scale > 1.0
    pipeline.check_inputs(
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt,
        prompt_embeds,
        negative_prompt_embeds,
    )
    device = pipeline._execution_device
    height = height or pipeline.default_sample_size * pipeline.vae_scale_factor
    width = width or pipeline.default_sample_size * pipeline.vae_scale_factor

    # Encode prompt (CDF guidance ON)
    lora_scale = (
        pipeline.joint_attention_kwargs.get("scale", None) if joint_attention_kwargs is not None else None
    )
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipeline.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_3=prompt_3,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        negative_prompt_3=negative_prompt_3,
        do_classifier_free_guidance=do_classifier_free_guidance,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        device=device,
        clip_skip=clip_skip,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        lora_scale=lora_scale,
    )
    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
    if do_classifier_free_guidance:
        if skip_guidance_layers is not None:
            original_prompt_embeds = prompt_embeds
            original_pooled_prompt_embeds = pooled_prompt_embeds
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
    pipeline._joint_attention_kwargs = joint_attention_kwargs
    batch_size = 1 if isinstance(prompt, str) or prompt is None else len(prompt)

    latents = pipeline.prepare_latents(
        batch_size * num_images_per_prompt,
        pipeline.transformer.config.in_channels,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # 5. Prepare timesteps
    scheduler_kwargs = {}
    if pipeline.scheduler.config.get("use_dynamic_shifting", None) and mu is None:
        _, _, height, width = latents.shape
        image_seq_len = (height // pipeline.transformer.config.patch_size) * (
                width // pipeline.transformer.config.patch_size
        )
        mu = calculate_shift(
            image_seq_len,
            pipeline.scheduler.config.get("base_image_seq_len", 256),
            pipeline.scheduler.config.get("max_image_seq_len", 4096),
            pipeline.scheduler.config.get("base_shift", 0.5),
            pipeline.scheduler.config.get("max_shift", 1.16),
        )
        scheduler_kwargs["mu"] = mu
    elif mu is not None:
        scheduler_kwargs["mu"] = mu

    timesteps, num_inference_steps = retrieve_timesteps(
        pipeline.scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        **scheduler_kwargs,
    )
    num_warmup_steps = max(len(timesteps) - num_inference_steps * pipeline.scheduler.order, 0)
    pipeline._num_timesteps = len(timesteps)
    timesteps, _ = pipeline.scheduler.set_timesteps(num_inference_steps, device=device), num_inference_steps
    all_latents, all_logprobs = [latents], []

    # 6. Prepare image embeddings
    if (ip_adapter_image is not None and pipeline.is_ip_adapter_active) or ip_adapter_image_embeds is not None:
        ip_adapter_image_embeds = pipeline.prepare_ip_adapter_image_embeds(
            ip_adapter_image,
            ip_adapter_image_embeds,
            device,
            batch_size * num_images_per_prompt,
            do_classifier_free_guidance,
        )

        if joint_attention_kwargs is None:
            pipeline._joint_attention_kwargs = {"ip_adapter_image_embeds": ip_adapter_image_embeds}
        else:
            pipeline._joint_attention_kwargs.update(ip_adapter_image_embeds=ip_adapter_image_embeds)


    with pipeline.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(pipeline.scheduler.timesteps):
            latent_input = torch.cat([latents] * 2)
            noise_pred = pipeline.transformer(
                hidden_states=latent_input,
                timestep=t.expand(latent_input.shape[0]),
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                joint_attention_kwargs=None,
                return_dict=return_dict,
            )[0]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                should_skip_layers = (
                    True
                    if i > num_inference_steps * skip_layer_guidance_start
                       and i < num_inference_steps * skip_layer_guidance_stop
                    else False
                )
                if skip_guidance_layers is not None and should_skip_layers:
                    timestep = t.expand(latents.shape[0])
                    latent_model_input = latents
                    noise_pred_skip_layers = pipeline.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=original_prompt_embeds,
                        pooled_projections=original_pooled_prompt_embeds,
                        joint_attention_kwargs=pipeline.joint_attention_kwargs,
                        return_dict=return_dict,
                        skip_layers=skip_guidance_layers,
                    )[0]
                    noise_pred = (
                            noise_pred + (noise_pred_text - noise_pred_skip_layers) * skip_layer_guidance_scale
                    )

            step_out = scheduler_step(pipeline.scheduler, noise_pred, t, latents)
            latents, log_prob = step_out.latents, step_out.log_probs
            latents_dtype = latents.dtype

            all_latents.append(latents)
            all_logprobs.append(log_prob)

            if latents.dtype != latents_dtype:
                if torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                    latents = latents.to(latents_dtype)

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipeline.scheduler.order == 0):
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)

    if output_type == "latent":
        image = latents
    else:
        image = pipeline.vae.decode(latents / pipeline.vae.config.scaling_factor, return_dict=return_dict)[0]
        image = pipeline.image_processor.postprocess(image, output_type=output_type)

    return DDPOPipelineOutput(image, all_latents, all_logprobs)


def pipeline_step_sd3_with_grad(
    pipeline: StableDiffusion3Pipeline,
    *args,
    truncated_backprop: bool = True,
    truncated_backprop_rand: bool = True,
    truncated_backprop_timestep: int = 27,
    truncated_rand_backprop_minmax: tuple[int, int] = (0, 28),
    gradient_checkpoint: bool = True,
    **kwargs,
):
    """Same as ``pipeline_step_sd3`` but back‑prop friendly (no @torch.no_grad)."""
    # The code mirrors the no‑grad version but keeps the transformer forward pass in‑graph
    prompt = kwargs.get("prompt")
    negative_prompt = kwargs.get("negative_prompt")
    guidance_scale = kwargs.get("guidance_scale", 7.0)
    num_images_per_prompt = kwargs.get("num_images_per_prompt", 1)
    height = kwargs.get("height")
    width = kwargs.get("width")
    num_inference_steps = kwargs.get("num_inference_steps", 28)
    generator = kwargs.get("generator")
    latents = kwargs.get("latents")
    output_type = kwargs.get("output_type", "pil")

    device = pipeline._execution_device
    height = height or pipeline.default_sample_size * pipeline.vae_scale_factor
    width = width or pipeline.default_sample_size * pipeline.vae_scale_factor

    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = _encode_prompt_sd3(
        pipeline, prompt, negative_prompt, num_images_per_prompt, device
    )
    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    batch_size = 1 if isinstance(prompt, str) or prompt is None else len(prompt)
    latents = pipeline.prepare_latents(
        batch_size * num_images_per_prompt,
        pipeline.transformer.config.in_channels,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
    all_latents, all_logprobs = [latents], []

    for i, t in enumerate(pipeline.scheduler.timesteps):
        latent_input = torch.cat([latents] * 2)
        if gradient_checkpoint:
            noise_pred = checkpoint.checkpoint(
                pipeline.transformer,
                latent_input,
                t.expand(latent_input.shape[0]),
                prompt_embeds,
                None,
                None,
                use_reentrant=False,
            )[0]
        else:
            noise_pred = pipeline.transformer(
                hidden_states=latent_input,
                timestep=t.expand(latent_input.shape[0]),
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]

        if truncated_backprop:
            if truncated_backprop_rand:
                if i < random.randint(*truncated_rand_backprop_minmax):
                    noise_pred = noise_pred.detach()
            elif i < truncated_backprop_timestep:
                noise_pred = noise_pred.detach()

        noise_uncond, noise_text = noise_pred.chunk(2)
        noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

        step_out = scheduler_step(pipeline.scheduler, noise_pred, t, latents)
        latents, log_prob = step_out.latents, step_out.log_probs
        all_latents.append(latents)
        all_logprobs.append(log_prob)

    if output_type == "latent":
        image = latents
    else:
        image = pipeline.vae.decode(latents / pipeline.vae.config.scaling_factor, return_dict=False)[0]
        image = pipeline.image_processor.postprocess(image, output_type=output_type)

    return DDPOPipelineOutput(image, all_latents, all_logprobs)


# -------------------------------------------------------------------------------------
# 3. Convenience wrapper class analogous to DefaultDDPOStableDiffusionPipeline
# -------------------------------------------------------------------------------------

class DefaultDDPOStableDiffusion3Pipeline:
    """Factory that wires a SD‑3.5 checkpoint into AlignProp‑compatible helpers."""

    def __init__(
        self,
        pretrained_model_name: str,
        *,
        pretrained_model_revision: str = "main", use_lora: bool = False
    ):
        self.sd3_pipeline = StableDiffusion3Pipeline.from_pretrained(
            pretrained_model_name, revision=pretrained_model_revision
        )
        transformer: SD3Transformer2DModel = self.sd3_pipeline.transformer

        # Replace scheduler with fresh Flow‑Match Euler (safer than in‑place reload).
        self.sd3_pipeline.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            self.sd3_pipeline.scheduler.config
        )
        self.sd3_pipeline.safety_checker = None  # disable for RL training
        self.use_lora = use_lora

        # freeze VAE & encoders; enable LoRA or full fine‑tune on transformer only
        self.sd3_pipeline.vae.requires_grad_(False)
        self.sd3_pipeline.text_encoder.requires_grad_(False)
        self.sd3_pipeline.text_encoder_2.requires_grad_(False)
        self.sd3_pipeline.text_encoder_3.requires_grad_(False)
        self.sd3_pipeline.transformer.requires_grad_(not self.use_lora)

        if use_lora:
            try:
                self.sd3_pipeline.load_lora_weights(
                    pretrained_model_name,
                    weight_name="pytorch_lora_weights.safetensors",
                    revision=pretrained_model_revision,
                )
            except OSError:
                warnings.warn(
                    "Trying to load LoRA weights but no LoRA weights found. Set `use_lora=False` or check that "
                    "`pytorch_lora_weights.safetensors` exists in the model folder.",
                    UserWarning,
                )


    # --- AlignProp public API --------------------------------------------------------
    def __call__(self, *args, **kwargs):
        return pipeline_step_sd3(self.sd3_pipeline, *args, **kwargs)

    def rgb_with_grad(self, *args, **kwargs):
        return pipeline_step_sd3_with_grad(self.sd3_pipeline, *args, **kwargs)

    def scheduler_step(self, *args, **kwargs):
        return scheduler_step(self.sd3_pipeline.scheduler, *args, **kwargs)

    # property passthroughs -----------------------------------------------------------
    @property
    def transformer(self):
        return self.sd3_pipeline.transformer

    @property
    def vae(self):
        return self.sd3_pipeline.vae

    @property
    def tokenizer(self):
        return self.sd3_pipeline.tokenizer

    @property
    def scheduler(self):
        return self.sd3_pipeline.scheduler

    @property
    def text_encoder(self):
        return self.sd3_pipeline.text_encoder

    @property
    def autocast(self):
        return contextlib.nullcontext if self.use_lora else None

    # save / load helpers -------------------------------------------------------------
    def save_pretrained(self, output_dir: str):
        if self.use_lora:
            state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(self.sd3_pipeline.unet))
            self.sd3_pipeline.save_lora_weights(save_directory=output_dir, transformer_lora_layers=state_dict)
        self.sd3_pipeline.save_pretrained(output_dir)
        return


    def set_progress_bar_config(self, *args, **kwargs):
        self.sd3_pipeline.set_progress_bar_config(*args, **kwargs)
        return

    def get_trainable_layers(self):
        if self.use_lora:
            lora_config = LoraConfig(
                    r=4,
                    lora_alpha=4,
                    init_lora_weights="gaussian",
                    target_modules=["to_k", "to_q", "to_v", "to_out.0"],
                )
            self.sd3_pipeline.transformer.add_adapter(lora_config)
            # To avoid accelerate unscaling problems in FP16.
            for param in self.sd3_pipeline.transformer.parameters():
                # only upcast trainable parameters (LoRA) into fp32
                if param.requires_grad:
                    param.data = param.to(torch.float32)
            return self.sd3_pipeline.transformer
        else:
            return self.sd3_pipeline.transformer


    def save_checkpoint(self, models, weights, output_dir):
        if len(models) != 1:
            raise ValueError("Given how the trainable params were set, this should be of length 1")
        if self.use_lora and hasattr(models[0], "peft_config") and getattr(models[0], "peft_config", None) is not None:
            state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(models[0]))
            self.sd3_pipeline.save_lora_weights(save_directory=output_dir, transformer_lora_layers=state_dict)
        elif not self.use_lora and isinstance(models[0], SD3Transformer2DModel):
            models[0].save_pretrained(os.path.join(output_dir, "transformer"))
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")


    def load_checkpoint(self, models, input_dir):
        if len(models) != 1:
            raise ValueError("Given how the trainable params were set, this should be of length 1")
        if self.use_lora:
            lora_state_dict = self.sd3_pipeline.lora_state_dict(
                input_dir, weight_name="pytorch_lora_weights.safetensors", local_files_only=True
            )
            self.sd3_pipeline.load_lora_into_transformer(state_dict=lora_state_dict, transformer=self.transformer)
        elif not self.use_lora and isinstance(models[0], self.transformer.__class__):
            load_model = self.transformer.__class__.from_pretrained(os.path.join(input_dir, "transformer"))
            models[0].register_to_config(**load_model.config)
            models[0].load_state_dict(load_model.state_dict())
            del load_model
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")



def pipeline_step_with_grad(
    pipeline: StableDiffusion3Pipeline,
    prompt: Optional[Union[str, List[str]]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    prompt_3: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 28,
    guidance_scale: float = 7.0,
    latents: Optional[torch.FloatTensor] = None,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    negative_prompt_3: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    sigmas: Optional[List[float]] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    ip_adapter_image: Optional[PipelineImageInput] = None,
    ip_adapter_image_embeds: Optional[torch.Tensor] = None,
    gradient_checkpoint: bool = True,
    output_type: Optional[str] = "pil",
    backprop_strategy: str = 'gaussian',
    backprop_kwargs: Dict[str, Any] = None,
    guidance_rescale: float = 0.0,
    return_dict: bool = True,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    clip_skip: Optional[int] = None,
    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    callback_steps: int = 1,
    max_sequence_length: int = 256,
    skip_guidance_layers: List[int] = None,
    skip_layer_guidance_scale: float = 2.8,
    skip_layer_guidance_stop: float = 0.2,
    skip_layer_guidance_start: float = 0.01,
    mu: Optional[float] = None,
):
    patch_size = (
        pipeline.transformer.config.patch_size if hasattr(pipeline, "transformer") and pipeline.transformer is not None else 2
    )
    height = height or pipeline.default_sample_size * pipeline.vae_scale_factor
    width = width or pipeline.default_sample_size * pipeline.vae_scale_factor
    height -= height % (pipeline.vae_scale_factor * patch_size)
    width -= width % (pipeline.vae_scale_factor * patch_size)
    with torch.no_grad():

        pipeline.check_inputs(
            prompt=prompt,
            prompt_2=None,
            prompt_3=None,
            height=height,
            width=width,
            # callback_steps,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds
            # negative_prompt_embeds=negative_prompt_embeds,
        )
        # The code mirrors the no‑grad version but keeps the transformer forward pass in‑graph
        do_classifier_free_guidance = guidance_scale > 1.0
        backprop_timestep = -1

        while backprop_timestep >= num_inference_steps or backprop_timestep < 1:
            if backprop_strategy == 'gaussian':
                backprop_timestep = int(
                    torch.distributions.Normal(backprop_kwargs['mean'], backprop_kwargs['std']).sample().item())
            elif backprop_strategy == 'uniform':
                backprop_timestep = int(torch.randint(backprop_kwargs['min'], backprop_kwargs['max'], (1,)).item())
            elif backprop_strategy == 'fixed':
                backprop_timestep = int(backprop_kwargs['value'])

        device = pipeline._execution_device
        prompt_embeds = prompt_embeds.to(device=device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device=device)
        lora_scale = (
            pipeline.joint_attention_kwargs.get("scale", None) if joint_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            do_classifier_free_guidance=do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            # negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device,
            clip_skip=clip_skip,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        if do_classifier_free_guidance:
            if skip_guidance_layers is not None:
                original_prompt_embeds = prompt_embeds
                original_pooled_prompt_embeds = pooled_prompt_embeds
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        pipeline._joint_attention_kwargs = joint_attention_kwargs
        batch_size = 1 if isinstance(prompt, str) or prompt is None else len(prompt)

        latents = pipeline.prepare_latents(
            batch_size * num_images_per_prompt,
            pipeline.transformer.config.in_channels,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 5. Prepare timesteps
        scheduler_kwargs = {}
        if pipeline.scheduler.config.get("use_dynamic_shifting", None) and mu is None:
            _, _, height, width = latents.shape
            image_seq_len = (height // pipeline.transformer.config.patch_size) * (
                    width // pipeline.transformer.config.patch_size
            )
            mu = calculate_shift(
                image_seq_len,
                pipeline.scheduler.config.get("base_image_seq_len", 256),
                pipeline.scheduler.config.get("max_image_seq_len", 4096),
                pipeline.scheduler.config.get("base_shift", 0.5),
                pipeline.scheduler.config.get("max_shift", 1.16),
            )
            scheduler_kwargs["mu"] = mu
        elif mu is not None:
            scheduler_kwargs["mu"] = mu

        timesteps, num_inference_steps = retrieve_timesteps(
            pipeline.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            **scheduler_kwargs,
        )
    num_warmup_steps = max(len(timesteps) - num_inference_steps * pipeline.scheduler.order, 0)
    pipeline._num_timesteps = len(timesteps)
    pipeline.scheduler.set_timesteps(num_inference_steps, device=device)

    # 6. Prepare image embeddings
    if (ip_adapter_image is not None and pipeline.is_ip_adapter_active) or ip_adapter_image_embeds is not None:
        ip_adapter_image_embeds = pipeline.prepare_ip_adapter_image_embeds(
            ip_adapter_image,
            ip_adapter_image_embeds,
            device,
            batch_size * num_images_per_prompt,
            do_classifier_free_guidance,
        )

        if joint_attention_kwargs is None:
            pipeline._joint_attention_kwargs = {"ip_adapter_image_embeds": ip_adapter_image_embeds}
        else:
            pipeline._joint_attention_kwargs.update(ip_adapter_image_embeds=ip_adapter_image_embeds)

    all_latents, all_logprobs = [latents], []

    with pipeline.progress_bar(total=num_inference_steps) as progress_bar:

        for i, t in enumerate(pipeline.scheduler.timesteps):
            latent_input = torch.cat([latents] * 2)
            if gradient_checkpoint:
                noise_pred = checkpoint.checkpoint(
                    pipeline.transformer,
                    latent_input,
                    t.expand(latent_input.shape[0]),
                    prompt_embeds,
                    None,
                    None,
                    use_reentrant=False,
                )[0]
            else:
                noise_pred = pipeline.transformer(
                    hidden_states=latent_input,
                    timestep=t.expand(latent_input.shape[0]),
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    joint_attention_kwargs=joint_attention_kwargs,
                    return_dict=return_dict,
                )[0]

            if i < backprop_timestep:
                noise_pred = noise_pred.detach()

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                should_skip_layers = (
                    True
                    if i > num_inference_steps * skip_layer_guidance_start
                       and i < num_inference_steps * skip_layer_guidance_stop
                    else False
                )
                if skip_guidance_layers is not None and should_skip_layers:
                    timestep = t.expand(latents.shape[0])
                    latent_model_input = latents
                    noise_pred_skip_layers = pipeline.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=original_prompt_embeds,
                        pooled_projections=original_pooled_prompt_embeds,
                        joint_attention_kwargs=pipeline.joint_attention_kwargs,
                        return_dict=False,
                        skip_layers=skip_guidance_layers,
                    )[0]
                    noise_pred = (
                            noise_pred + (noise_pred_text - noise_pred_skip_layers) * skip_layer_guidance_scale
                    )


            step_out = scheduler_step(pipeline.scheduler, noise_pred, t, latents)
            latents, log_prob = step_out.latents, step_out.log_probs
            all_latents.append(latents)
            all_logprobs.append(log_prob)

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipeline.scheduler.order == 0):
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)

    if output_type == "latent":
        image = latents
    else:
        image = pipeline.vae.decode(latents / pipeline.vae.config.scaling_factor, return_dict=False)[0]
        image = pipeline.image_processor.postprocess(image, output_type=output_type)

    return DDPOPipelineOutput(image, all_latents, all_logprobs)



class AlignPropDiffusionPipeline3(DefaultDDPOStableDiffusion3Pipeline):
    def __init__(self, pretrained_model_name: str, pretrained_model_revision: str = "main", use_lora: bool = True):
        super().__init__(pretrained_model_name,pretrained_model_revision=pretrained_model_revision,use_lora=use_lora)

    def rgb_with_grad(self, *args, **kwargs) -> DDPOPipelineOutput:
        return pipeline_step_with_grad(self.sd3_pipeline, *args, **kwargs)