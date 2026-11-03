# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from typing import List, Dict, Optional, Union, Any

from PIL import Image
import torch

from data.data_utils import pil_img2rgb
from modeling.bagel.qwen2_navit import NaiveCache


VLM_THINK_SYSTEM_PROMPT = """You should first think about the reasoning process in the mind and then provide the user with the answer. 
The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here"""

GEN_THINK_SYSTEM_PROMPT = """You should first think about the planning process in the mind and then generate the image. 
The planning process is enclosed within <think> </think> tags, i.e. <think> planning process here </think> image here"""


class InterleaveInferencer:
    def __init__(
        self, model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids
    ):
        self.model = model
        self.vae_model = vae_model
        self.tokenizer = tokenizer
        self.vae_transform = vae_transform
        self.vit_transform = vit_transform
        self.new_token_ids = new_token_ids

    def init_gen_context(self):
        gen_context = {
            "kv_lens": [0],
            "ropes": [0],
            "past_key_values": NaiveCache(
                self.model.config.llm_config.num_hidden_layers
            ),
            "segments": [],
            "trace": {},
        }
        self._sync_cache_metadata(gen_context)
        return gen_context

    @staticmethod
    def _sync_cache_metadata(gen_context):
        past_key_values = gen_context.get("past_key_values")
        if past_key_values is None:
            return
        past_key_values.segment_ledger = list(gen_context.get("segments", []))
        past_key_values.metric_trace = dict(gen_context.get("trace", {}))

    def update_trace(self, gen_context, **trace_fields):
        trace = gen_context.setdefault("trace", {})
        trace.update({k: v for k, v in trace_fields.items() if v is not None})
        self._sync_cache_metadata(gen_context)
        return gen_context

    @staticmethod
    def _append_segment(
        gen_context,
        kind: str,
        start: int,
        end: int,
        segment_meta: Optional[Dict[str, Any]] = None,
    ):
        if end <= start:
            return
        meta = dict(segment_meta or {})
        segments = gen_context.setdefault("segments", [])
        segments.append(
            {
                "kind": kind,
                "start": int(start),
                "end": int(end),
                **meta,
            }
        )

    @torch.no_grad()
    def drop_generated_image_context(self, gen_context):
        segments = list(gen_context.get("segments", []))
        if not segments:
            return gen_context
        remove_ranges = []
        for seg in segments:
            if str(seg.get("source") or "") == "generated_image":
                start = int(seg.get("start", 0))
                end = int(seg.get("end", 0))
                if end > start:
                    remove_ranges.append((start, end))
        if not remove_ranges:
            return gen_context
        remove_ranges.sort()
        merged = []
        for start, end in remove_ranges:
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        cache = gen_context["past_key_values"]
        total_len = (
            gen_context["kv_lens"][-1] if gen_context.get("kv_lens") else cache.seq_lens
        )
        keep = torch.ones(total_len, dtype=torch.bool)
        for start, end in merged:
            keep[start:end] = False
        keep_idx = torch.nonzero(keep, as_tuple=False).flatten()
        removed_before = [0] * (total_len + 1)
        removed = 0
        mr_iter = iter(merged)
        cur = next(mr_iter, None)
        for idx in range(total_len):
            while cur is not None and idx >= cur[1]:
                cur = next(mr_iter, None)
            if cur is not None and cur[0] <= idx < cur[1]:
                removed += 1
            removed_before[idx + 1] = removed
        new_segments = []
        for seg in segments:
            if str(seg.get("source") or "") == "generated_image":
                continue
            start = int(seg.get("start", 0))
            end = int(seg.get("end", 0))
            new_start = start - removed_before[start]
            new_end = end - removed_before[end]
            if new_end > new_start:
                new_seg = dict(seg)
                new_seg["start"] = int(new_start)
                new_seg["end"] = int(new_end)
                new_segments.append(new_seg)
        for layer_idx in cache.key_cache.keys():
            if cache.key_cache[layer_idx] is not None:
                cache.key_cache[layer_idx] = cache.key_cache[layer_idx][keep_idx]
            if cache.value_cache[layer_idx] is not None:
                cache.value_cache[layer_idx] = cache.value_cache[layer_idx][keep_idx]
        kept_len = int(keep_idx.numel())
        if gen_context.get("kv_lens"):
            gen_context["kv_lens"][-1] = kept_len
        if gen_context.get("ropes"):
            gen_context["ropes"][-1] = kept_len
        gen_context["segments"] = new_segments
        self._sync_cache_metadata(gen_context)
        return gen_context

    @torch.no_grad()
    def update_context_text(
        self,
        text,
        gen_context,
        skip_prefill=False,
        segment_meta: Optional[Dict[str, Any]] = None,
    ):
        # used for interleave data, currently only support 1 data inference,

        past_key_values = gen_context["past_key_values"]
        # torch.save(past_key_values.key_cache[27], 'past_key_values_before_prefill.pt')
        kv_lens = gen_context["kv_lens"]
        ropes = gen_context["ropes"]
        prev_kv = kv_lens[-1] if kv_lens else 0
        generation_input, kv_lens, ropes = self.model.prepare_prompts(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            prompts=[text],
            tokenizer=self.tokenizer,
            new_token_ids=self.new_token_ids,
        )
        past_key_values = self.model.forward_cache_update_text(
            past_key_values, **generation_input
        )
        gen_context["kv_lens"] = kv_lens  # text_length
        gen_context["ropes"] = ropes
        gen_context["past_key_values"] = past_key_values
        self._append_segment(
            gen_context,
            "text",
            prev_kv,
            kv_lens[-1] if kv_lens else prev_kv,
            segment_meta,
        )
        self._sync_cache_metadata(gen_context)
        # torch.save(past_key_values.key_cache[27], 'past_key_values_after_prefill.pt')
        return gen_context

    @torch.no_grad()
    def update_context_image(
        self,
        image,
        gen_context,
        vae=True,
        vit=True,
        segment_meta: Optional[Dict[str, Any]] = None,
    ):
        # used for interleave data, currently only support 1 data inference,

        assert vae or vit
        past_key_values = gen_context["past_key_values"]
        kv_lens = gen_context["kv_lens"]
        ropes = gen_context["ropes"]
        trace = gen_context.setdefault("trace", {})
        original_query_kind_present = "query_kind" in trace
        original_query_kind = trace.get("query_kind")

        if vae:
            ## update vae kvlens:1134;ropes:109
            prev_kv = kv_lens[-1] if kv_lens else 0
            trace["query_kind"] = "vae"
            self._sync_cache_metadata(gen_context)
            generation_input, kv_lens, ropes = self.model.prepare_vae_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes,
                images=[image],
                transforms=self.vae_transform,
                new_token_ids=self.new_token_ids,
            )
            past_key_values = self.model.forward_cache_update_vae(
                self.vae_model, past_key_values, **generation_input
            )
            vae_meta = dict(segment_meta or {})
            vae_meta.setdefault("modality", "vae")
            self._append_segment(
                gen_context,
                "vae",
                prev_kv,
                kv_lens[-1] if kv_lens else prev_kv,
                vae_meta,
            )

        if vit:
            ## update vit
            prev_kv = kv_lens[-1] if kv_lens else 0
            trace["query_kind"] = "vit"
            self._sync_cache_metadata(gen_context)
            generation_input, kv_lens, ropes = self.model.prepare_vit_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes,
                images=[image],
                transforms=self.vit_transform,
                new_token_ids=self.new_token_ids,
            )
            past_key_values = self.model.forward_cache_update_vit(
                past_key_values, **generation_input
            )
            vit_meta = dict(segment_meta or {})
            vit_meta.setdefault("modality", "vit")
            self._append_segment(
                gen_context,
                "vit",
                prev_kv,
                kv_lens[-1] if kv_lens else prev_kv,
                vit_meta,
            )

        gen_context["kv_lens"] = kv_lens
        gen_context["ropes"] = ropes
        gen_context["past_key_values"] = past_key_values
        if original_query_kind_present:
            trace["query_kind"] = original_query_kind
        else:
            trace.pop("query_kind", None)
        self._sync_cache_metadata(gen_context)

        return gen_context

    @torch.no_grad()
    def gen_image(
        self,
        image_shape,
        gen_context,
        cfg_text_scale=4.0,
        cfg_img_scale=1.5,
        cfg_text_precontext=None,
        cfg_img_precontext=None,
        cfg_interval=(0.4, 1.0),
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        num_timesteps=50,
        timestep_shift=3.0,
        enable_taylorseer=False,
        early_stop_sampled_steps: Optional[int] = None,
        verbose: bool = True,
    ):
        # print(cfg_renorm_type)
        past_key_values = gen_context["past_key_values"]  # [2597,4,128]
        kv_lens = gen_context["kv_lens"]
        ropes = gen_context["ropes"]
        # s1 = torch.get_rng_state()
        generation_input = self.model.prepare_vae_latent(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            image_sizes=[image_shape],
            new_token_ids=self.new_token_ids,
        )
        # s2 = torch.get_rng_state()
        # changed = not torch.equal(s1, s2)
        # text cfg
        cfg_text_past_key_values = cfg_text_precontext[
            "past_key_values"
        ]  # [2397,4,128]
        kv_lens_cfg = cfg_text_precontext["kv_lens"]
        ropes_cfg = cfg_text_precontext["ropes"]
        generation_input_cfg_text = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg,
            image_sizes=[image_shape],
        )

        # img cfg
        cfg_img_past_key_values = cfg_img_precontext["past_key_values"]  # [200,4,128]
        kv_lens_cfg = cfg_img_precontext["kv_lens"]
        ropes_cfg = cfg_img_precontext["ropes"]
        generation_input_cfg_img = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg,
            image_sizes=[image_shape],
        )

        unpacked_latent = self.model.generate_image(
            past_key_values=past_key_values,
            cfg_text_past_key_values=cfg_text_past_key_values,
            cfg_img_past_key_values=cfg_img_past_key_values,
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,  # 4
            cfg_img_scale=cfg_img_scale,  # 2
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            timestep_shift=timestep_shift,
            **generation_input,
            cfg_text_packed_position_ids=generation_input_cfg_text[
                "cfg_packed_position_ids"
            ],
            cfg_text_packed_query_indexes=generation_input_cfg_text[
                "cfg_packed_query_indexes"
            ],
            cfg_text_key_values_lens=generation_input_cfg_text["cfg_key_values_lens"],
            cfg_text_packed_key_value_indexes=generation_input_cfg_text[
                "cfg_packed_key_value_indexes"
            ],
            cfg_img_packed_position_ids=generation_input_cfg_img[
                "cfg_packed_position_ids"
            ],
            cfg_img_packed_query_indexes=generation_input_cfg_img[
                "cfg_packed_query_indexes"
            ],
            cfg_img_key_values_lens=generation_input_cfg_img["cfg_key_values_lens"],
            cfg_img_packed_key_value_indexes=generation_input_cfg_img[
                "cfg_packed_key_value_indexes"
            ],
            enable_taylorseer=enable_taylorseer,
            early_stop_sampled_steps=early_stop_sampled_steps,
            verbose=verbose,
        )

        image = self.decode_image(unpacked_latent[0], image_shape)
        return image

    def decode_image(self, latent, image_shape):
        H, W = image_shape
        h, w = H // self.model.latent_downsample, W // self.model.latent_downsample

        latent = latent.reshape(
            1,
            h,
            w,
            self.model.latent_patch_size,
            self.model.latent_patch_size,
            self.model.latent_channel,
        )
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(
            1,
            self.model.latent_channel,
            h * self.model.latent_patch_size,
            w * self.model.latent_patch_size,
        )

        latent = latent.to(self.model.device)
        image = self.vae_model.decode(latent)

        image = (image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255
        image = Image.fromarray((image).to(torch.uint8).cpu().numpy())

        return image

    @torch.no_grad()
    def gen_text(
        self,
        gen_context,
        max_length: int = 500,
        do_sample: bool = True,
        temperature: float = 1.0,
        return_ids=False,
        generate_next_token=False,
    ):
        gen_context = deepcopy(gen_context)
        past_key_values = gen_context["past_key_values"]
        kv_lens = gen_context["kv_lens"]
        ropes = gen_context["ropes"]
        # Prepend the BOS token.
        generation_input = self.model.prepare_start_tokens(
            kv_lens, ropes, self.new_token_ids
        )
        model_outputs = self.model.generate_text(
            past_key_values=past_key_values,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            end_token_id=self.new_token_ids["eos_token_id"],
            **generation_input,
        )  # [93,1]
        # torch.save(past_key_values.key_cache[27],"past_key_cache_after_decode.pt")
        if return_ids:
            return model_outputs[:, 0].tolist()

        output = self.tokenizer.decode(model_outputs[:, 0])
        if not generate_next_token:
            output = output.split("<|im_end|>")[0].split("<|im_start|>")[1]
        return output

    @torch.no_grad()
    def interleave_inference(
        self,
        input_lists: List[Union[str, Image.Image]],
        think=False,
        understanding_output=False,
        max_think_token_n=1000,
        do_sample=False,
        text_temperature=0.7,
        cfg_text_scale=3.0,
        cfg_img_scale=1.5,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        image_shapes=(1024, 1024),
        enable_taylorseer=False,
    ) -> List[Union[str, Image.Image]]:

        output_list = []
        gen_context = self.init_gen_context()
        cfg_text_context = self.init_gen_context()
        cfg_img_context = self.init_gen_context()

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            if think:
                if understanding_output:
                    system_prompt = VLM_THINK_SYSTEM_PROMPT
                else:
                    system_prompt = GEN_THINK_SYSTEM_PROMPT
                gen_context = self.update_context_text(system_prompt, gen_context)
                cfg_img_context = self.update_context_text(
                    system_prompt, cfg_img_context
                )

            print(
                f"Start interleave inference with {len(input_lists)} inputs: {input_lists}"
            )
            for input_term in input_lists:
                if isinstance(input_term, str):
                    gen_context = self.update_context_text(input_term, gen_context)
                    cfg_img_context = self.update_context_text(
                        input_term, cfg_img_context
                    )

                elif isinstance(input_term, Image.Image):
                    input_term = self.vae_transform.resize_transform(
                        pil_img2rgb(input_term)
                    )
                    gen_context = self.update_context_image(
                        input_term, gen_context, vae=not understanding_output
                    )
                    cfg_text_context = self.update_context_image(
                        input_term, cfg_text_context, vae=not understanding_output
                    )

                    image_shapes = input_term.size[::-1]
                else:
                    raise ValueError(f"Unsupported input type: {type(input_term)}")

            if understanding_output:
                gen_text = self.gen_text(
                    gen_context,
                    do_sample=do_sample,
                    temperature=text_temperature,
                    max_length=max_think_token_n,
                )
                output_list.append(gen_text)

            else:
                if think:
                    gen_text = self.gen_text(
                        gen_context,
                        do_sample=do_sample,
                        temperature=text_temperature,
                        max_length=max_think_token_n,
                    )
                    gen_context = self.update_context_text(gen_text, gen_context)
                    cfg_img_context = self.update_context_text(
                        gen_text, cfg_img_context
                    )

                    output_list.append(gen_text)

                img = self.gen_image(
                    image_shapes,
                    gen_context,
                    cfg_text_precontext=cfg_text_context,
                    cfg_img_precontext=cfg_img_context,
                    cfg_text_scale=cfg_text_scale,
                    cfg_img_scale=cfg_img_scale,
                    cfg_interval=cfg_interval,
                    timestep_shift=timestep_shift,
                    num_timesteps=num_timesteps,
                    cfg_renorm_min=cfg_renorm_min,
                    cfg_renorm_type=cfg_renorm_type,
                    enable_taylorseer=enable_taylorseer,
                )

                output_list.append(img)

        return output_list

    def __call__(
        self,
        image: Optional[Image.Image] = None,
        text: Optional[str] = None,
        input_list=None,
        **kargs,
    ) -> Dict[str, Any]:
        output_dict = {"image": None, "text": None}

        if input_list is None:
            if image is None and text is None:
                print("Please provide at least one input: either an image or text.")
                return output_dict

            input_list = []
            if image is not None:
                input_list.append(image)
            if text is not None:
                input_list.append(text)
        else:
            if (
                any([not isinstance(i, (str, Image.Image)) for i in input_list])
                or len(input_list) == 0
            ):
                print(f"Invalid input_list: {input_list}")
                return output_dict

        output_list = self.interleave_inference(input_list, **kargs)

        for i in output_list:
            if isinstance(i, Image.Image):
                output_dict["image"] = i
            elif isinstance(i, str):
                output_dict["text"] = i
        return output_dict
