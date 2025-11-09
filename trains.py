import os
import utils
from train_tools import mlm
import numpy as np
import torch
import torch.nn.functional as Fnn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F  
import random
import warnings
from torchcontrib.optim import SWA
from lookahead import Lookahead
from torch.cuda.amp import autocast


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb=128")



def _is_cuda_oom(err: Exception) -> bool:
    msg = str(err).lower()
    return ("out of memory" in msg and "cuda" in msg) or ("cublas" in msg and "alloc" in msg)



def apply_gaussian_blur(image):
    gaussian_blur = transforms.GaussianBlur(5, sigma=(0.1, 2.0))
    return gaussian_blur(image)


def apply_random_illumination(image, sat_min=0.8, sat_max=1.2):
    factor = random.uniform(sat_min, sat_max)
    return transforms.ColorJitter(saturation=factor)(image)



class EmbeddingMemoryBank:
    def __init__(self, capacity, dim, device, dtype=torch.float16, store_ids=True):
        self.capacity = int(capacity)
        self.dim = int(dim)
        self.device = device
        self.dtype = dtype
        self.store_ids = store_ids

        self.img = torch.zeros((self.capacity, self.dim), device=self.device, dtype=self.dtype)
        self.txt = torch.zeros((self.capacity, self.dim), device=self.device, dtype=self.dtype)
        self.ids = torch.full((self.capacity,), -1, device=self.device, dtype=torch.long) if store_ids else None

        self.size = 0
        self.ptr = 0  

    def __len__(self):
        return self.size

    def _write_slice(self, start, tensor, where):
        n = tensor.size(0)
        end = start + n
        if end <= self.capacity:
            where[start:end] = tensor
        else:
            first = self.capacity - start
            where[start:] = tensor[:first]
            remain = n - first
            if remain > 0:
                where[:remain] = tensor[first:first+remain]

    @torch.no_grad()
    def push(self, z_img, z_txt, ids=None):
        n = min(z_img.size(0), z_txt.size(0))
        if n <= 0:
            return
        z_i = z_img[:n].detach().to(self.device, dtype=self.dtype)
        z_t = z_txt[:n].detach().to(self.device, dtype=self.dtype)

        if n >= self.capacity:
            self.img.copy_(z_i[-self.capacity:])
            self.txt.copy_(z_t[-self.capacity:])
            if self.store_ids and ids is not None:
                self.ids.copy_(ids[-self.capacity:].to(self.device))
            self.size = self.capacity
            self.ptr = 0
            return

        self._write_slice(self.ptr, z_i, self.img)
        self._write_slice(self.ptr, z_t, self.txt)
        if self.store_ids and ids is not None:
            ids = ids[:n].to(self.device)
            self._write_slice(self.ptr, ids, self.ids)

        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    @torch.no_grad()
    def sample_random(self, batch_size, avoid_ids=None):
        if self.size == 0:
            return None
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        if self.store_ids and avoid_ids is not None and self.ids is not None:
            avoid_ids = avoid_ids.to(self.device)
            mask_same = (self.ids[idx].unsqueeze(1) == avoid_ids.unsqueeze(0)).any(dim=1)
            if mask_same.any():
                idx[mask_same] = torch.randint(0, self.size, (int(mask_same.sum()),), device=self.device)
        return self.img[idx], self.txt[idx], (self.ids[idx] if self.store_ids else None)

    @torch.no_grad()
    def sample_hard_by_similarity(self, z_img_query, avoid_ids=None, topk=64):
        if self.size == 0:
            return None
        zq = z_img_query.to(self.device, dtype=torch.float32, non_blocking=True)
        mem_txt_all = self.txt[:self.size].to(dtype=torch.float32)

        sims = zq @ mem_txt_all.T  # (B, Nmem)

        if self.store_ids and avoid_ids is not None and self.ids is not None:
            avoid_ids = avoid_ids.to(self.device, non_blocking=True)
            avoid = (self.ids[:self.size].unsqueeze(0) == avoid_ids.unsqueeze(1))
            sims = sims.masked_fill(avoid, float('-inf'))

        if torch.isneginf(sims).all():
            return self.sample_random(z_img_query.size(0), avoid_ids=avoid_ids)

        B, N = sims.size()
        k = min(topk, N)
        _, topk_idx = torch.topk(sims, k=k, dim=1, largest=True)
        pick = torch.randint(0, k, (B,), device=sims.device)
        chosen_idx = topk_idx[torch.arange(B, device=sims.device), pick]

        mem_img_sel = self.img[chosen_idx]
        mem_txt_sel = self.txt[chosen_idx]
        return mem_img_sel, mem_txt_sel, (self.ids[chosen_idx] if self.store_ids else None)



def _unwrap_ddp(m):
    return m.module if hasattr(m, "module") else m

def _keep_half_or_bf16(z):
    if z.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        z = z.to(torch.float16)
    return z

def get_img_emb(model, images):

    m = _unwrap_ddp(model)

    if hasattr(m, "encode_image") and callable(getattr(m, "encode_image")):
        z = m.encode_image(images)
    else:
        enc = None
        for name in ["vision_encoder", "visual_encoder", "image_encoder"]:
            if hasattr(m, name):
                enc = getattr(m, name)
                break
        if enc is None:
            raise RuntimeError("找不到视觉编码器（vision_encoder/visual_encoder/image_encoder）。")

        if hasattr(enc, "forward_features") and callable(getattr(enc, "forward_features")):
            feats = enc.forward_features(images)
        else:
            feats = enc(images)

        if isinstance(feats, (list, tuple)):
            feats = feats[-1]
        if feats.dim() == 4:
            feats = feats.mean(dim=(2, 3))

        if hasattr(m, "vision_proj"):
            z = m.vision_proj(feats)
        elif hasattr(m, "visual_proj"):
            z = m.visual_proj(feats)
        elif hasattr(m, "proj"):
            z = m.proj(feats)
        else:
            z = feats

    z = _keep_half_or_bf16(z)
    return Fnn.normalize(z, dim=-1)


def get_txt_emb(model, input_ids, attention_mask):

    m = _unwrap_ddp(model)


    if hasattr(m, "get_text_embeds") and hasattr(m, "get_features"):
        try:
            text_embeds = m.get_text_embeds(input_ids, attention_mask)  # (B, L, H)
            z = m.get_features(image_embeds=None, text_embeds=text_embeds)  # (B, D)
            z = _keep_half_or_bf16(z)
            return Fnn.normalize(z, dim=-1)
        except Exception:

            pass

    if hasattr(m, "encode_text") and callable(getattr(m, "encode_text")):
        z = m.encode_text(input_ids=input_ids, attention_mask=attention_mask)
        z = _keep_half_or_bf16(z)
        return Fnn.normalize(z, dim=-1)

    txt = None
    for name in ["text_encoder", "bert"]:
        if hasattr(m, name):
            txt = getattr(m, name)
            break
    if txt is None:
        raise RuntimeError("找不到文本编码器（text_encoder/bert）。")

    backbone = txt.bert if hasattr(txt, "bert") else txt
    try:
        out = backbone(input_ids=input_ids,
                       attention_mask=attention_mask,
                       return_dict=True,
                       mode='text')  # 关键：显式 text-only 模式
    except TypeError:

        out = backbone(input_ids=input_ids,
                       attention_mask=attention_mask,
                       return_dict=True)

    cls = out.last_hidden_state[:, 0] if hasattr(out, "last_hidden_state") else out[0][:, 0]

    if hasattr(m, "text_proj"):
        z = m.text_proj(cls)
    elif hasattr(txt, "text_proj"):
        z = txt.text_proj(cls)
    else:
        z = cls

    z = _keep_half_or_bf16(z)
    return Fnn.normalize(z, dim=-1)



def mixup_embed(a, b, alpha=0.4):
 
    B = a.size(0)
    lam = torch.distributions.Beta(alpha, alpha).sample((B, 1)).to(a.device)
    out = lam * a + (1 - lam) * b
    return Fnn.normalize(out, dim=-1)


def compute_clip_itc_loss(z_img, z_txt_all, model=None, default_temperature=0.07):

    if (model is not None) and hasattr(model, 'logit_scale'):
        logits = torch.exp(model.logit_scale) * (z_img @ z_txt_all.T)
    else:
        logits = (z_img @ z_txt_all.T) / default_temperature
    B = z_img.size(0)
    labels = torch.arange(B, device=z_img.device)
    return Fnn.cross_entropy(logits, labels)


def memory_itc_augment(
    model, images_or_emb, text_input, idx_tensor,
    embed_bank, mem_cfg, want_emb_from_model=True
):

    if embed_bank is None or len(embed_bank) == 0:
        return 0.0, None, None, None


    z_img_cur = images_or_emb if not want_emb_from_model else get_img_emb(model, images_or_emb)
    z_txt_cur = get_txt_emb(model, text_input.input_ids, text_input.attention_mask)

    if mem_cfg['hard_negatives']:
        sampled = embed_bank.sample_hard_by_similarity(z_img_cur.detach(), avoid_ids=idx_tensor, topk=mem_cfg['topk'])
    else:
        sampled = embed_bank.sample_random(z_img_cur.size(0), avoid_ids=idx_tensor)
    if sampled is None:
        return 0.0, None, z_img_cur, z_txt_cur

    _, mem_txt, _ = sampled
    mem_txt = mem_txt.to(z_img_cur.device, non_blocking=True).detach()  

    backprop_keys = bool(mem_cfg.get('backprop_keys', False))
    kbp = float(mem_cfg.get('keys_backprop_prob', 0.0))
    if (not backprop_keys) and kbp > 0.0:
        backprop_keys = (torch.rand(()) < kbp).item()

    if backprop_keys:
        tilde_txt = mixup_embed(z_txt_cur, mem_txt, alpha=mem_cfg['alpha'])
        keys_txt = torch.cat([z_txt_cur, tilde_txt], dim=0)
    else:
        tilde_txt = mixup_embed(z_txt_cur.detach(), mem_txt, alpha=mem_cfg['alpha']).detach()
        keys_txt = torch.cat([z_txt_cur.detach(), tilde_txt], dim=0)  


    use_scale = bool(mem_cfg.get('use_model_logit_scale', False))
    model_for_temp = model if use_scale else None


    if mem_cfg.get('amp', True):
        with autocast(dtype=torch.float16):
            loss_itc_ext = compute_clip_itc_loss(
                z_img_cur, keys_txt, model=model_for_temp, default_temperature=mem_cfg['temperature']
            )
    else:
        loss_itc_ext = compute_clip_itc_loss(
            z_img_cur, keys_txt, model=model_for_temp, default_temperature=mem_cfg['temperature']
        )

    loss_add = mem_cfg['weight'] * loss_itc_ext
    return loss_add, loss_itc_ext.detach(), z_img_cur, z_txt_cur


def in_batch_mixup(images, alpha=0.8):
    if alpha <= 0:
        return images
    B = images.size(0)
    perm = torch.randperm(B, device=images.device)
    lam = torch.distributions.Beta(alpha, alpha).sample((1,)).to(images.device).item()
    return lam * images + (1 - lam) * images[perm]



def train(model, data_loader, optimizer, tokenizer, epoch, device, scheduler, config, mask_generator=None):
    model.train()

    base_opt = optimizer
    lookahead_optimizer = Lookahead(base_opt, alpha=0.5, k=6)
    if config.get('swa', False):
        optimizer = SWA(lookahead_optimizer, config['swa_start'], config['swa_freq'], config['swa_lr'])
    else:
        optimizer = lookahead_optimizer


    meta_epochs = int(config.get('meta_epochs', 10))
    mem_start_epoch = int(config.get('mem_start_epoch', meta_epochs))


    use_embed_memory = bool(config.get('use_embed_memory', True))
    mem_capacity = int(config.get('mem_capacity', 2048))
    mem_alpha = float(config.get('mem_alpha', 0.4))
    mem_temperature = float(config.get('mem_temperature', 0.07))
    mem_weight = float(config.get('mem_weight', 0.15))
    mem_topk = int(config.get('mem_topk', 32))
    mem_hard = bool(config.get('mem_hard_negatives', True))
    mem_warmup = int(config.get('mem_warmup_steps', 500))
    mem_stride = int(config.get('mem_stride', 2))
    mem_device = torch.device(config.get('mem_device', 'cpu'))


    mem_weight_start = float(config.get('mem_weight_start', mem_weight))
    mem_weight_ramp_epochs = int(config.get('mem_weight_ramp_epochs', 0))
    mem_use_model_logit_scale = bool(config.get('mem_use_model_logit_scale', False))
    mem_keys_backprop_prob = float(config.get('mem_keys_backprop_prob', 0.0))


    img_only_cfg = config.get('ext_backward_on_img_only', None)
    keys_cfg = config.get('ext_backward_on_keys', None)
    if img_only_cfg is not None:
        backprop_keys_base = not bool(img_only_cfg)
    elif keys_cfg is not None:
        backprop_keys_base = bool(keys_cfg)
    else:
        backprop_keys_base = False  

    mixup3_alpha = float(config.get('mixup3_alpha', 0.8))

    embed_bank = None
    embed_bank_warned = False
    global_step = 0

    def _mem_weight_at(e):
        if e < mem_start_epoch or mem_weight_ramp_epochs <= 0:
            return (mem_weight_start if e >= mem_start_epoch else 0.0)
        t = min(1.0, (e - mem_start_epoch + 1) / float(mem_weight_ramp_epochs))
        return (1.0 - t) * mem_weight_start + t * mem_weight

    def mk_mem_cfg(e):
        return {
            'alpha': mem_alpha,
            'temperature': mem_temperature,
            'weight': _mem_weight_at(e),
            'topk': mem_topk,
            'hard_negatives': mem_hard,
            'amp': True,
            'use_model_logit_scale': mem_use_model_logit_scale,
            'backprop_keys': backprop_keys_base,
            'keys_backprop_prob': mem_keys_backprop_prob,
        }

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_itc', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    if config.get('mlm', False):
        metric_logger.add_meter('loss_mlm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50

    for i, (image, text_eda, text, idx) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True)
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=config['max_tokens'],
                               return_tensors="pt").to(device)
        text_input_eda = tokenizer(text_eda, padding='max_length', truncation=True, max_length=config['max_tokens'],
                                   return_tensors="pt").to(device)


        if epoch < meta_epochs:
            if config.get('mlm', False):
                text_ids_masked, masked_pos, masked_ids = mlm(text, text_input, tokenizer, device, mask_generator, config)

            initial_params = {k: v.clone() for k, v in model.state_dict().items()}

            # ---- Task 1: blur ----
            image_blur = apply_gaussian_blur(image)
            if config.get('mlm', False):
                loss_itc, loss_itm, loss_mlm = model(
                    image_blur, text_input.input_ids, text_input.attention_mask,
                    text_ids_masked=text_ids_masked, masked_pos=masked_pos, masked_ids=masked_ids,
                    idx=idx, text_ids_eda=text_input_eda.input_ids, text_atts_eda=text_input_eda.attention_mask
                )
                loss1 = loss_itc + loss_itm + loss_mlm
            else:
                loss_itc, loss_itm = model(
                    image_blur, text_input.input_ids, text_input.attention_mask, idx=idx,
                    text_ids_eda=text_input_eda.input_ids, text_atts_eda=text_input_eda.attention_mask
                )
                loss1 = loss_itc + loss_itm

            itc_display_total = loss_itc
            enable_mem = (
                use_embed_memory and (epoch >= mem_start_epoch) and
                (global_step >= mem_warmup) and (global_step % mem_stride == 0)
            )
            if enable_mem:
                try:
                    if embed_bank is None:
                        z_img_tmp = get_img_emb(model, image_blur)
                        embed_bank = EmbeddingMemoryBank(mem_capacity, z_img_tmp.size(1), mem_device, dtype=torch.float16, store_ids=True)
                        z_txt_tmp = get_txt_emb(model, text_input.input_ids, text_input.attention_mask)
                        embed_bank.push(z_img_tmp.detach(), z_txt_tmp.detach(), ids=idx)

                    loss_add, itc_ext, z_img_cur, z_txt_cur = memory_itc_augment(
                        model, image_blur, text_input, idx, embed_bank, mk_mem_cfg(epoch), want_emb_from_model=True
                    )
                    if loss_add != 0.0:
                        loss1 = loss1 + loss_add
                        itc_display_total = loss_itc + itc_ext
                    embed_bank.push(z_img_cur.detach(), z_txt_cur.detach(), ids=idx)

                except Exception as e:
                    if _is_cuda_oom(e):
                        torch.cuda.empty_cache()
                    else:
                        if not embed_bank_warned:
                            warnings.warn(f"[Memory disabled] {e}")
                            embed_bank_warned = True
                        use_embed_memory = False

            optimizer.zero_grad()
            loss1.backward()
            optimizer.step()
            params_after_task1 = {k: v.clone() for k, v in model.state_dict().items()}

            # ---- Task 2: illumination ----
            model.load_state_dict(initial_params)

            sat_range = (0.8, 1.2) if epoch < 3 else (0.6, 1.4)
            image_illumination = apply_random_illumination(image, *sat_range)

            if config.get('mlm', False):
                loss_itc, loss_itm, loss_mlm = model(
                    image_illumination, text_input.input_ids, text_input.attention_mask,
                    text_ids_masked=text_ids_masked, masked_pos=masked_pos, masked_ids=masked_ids,
                    idx=idx, text_ids_eda=text_input_eda.input_ids, text_atts_eda=text_input_eda.attention_mask
                )
                loss2 = loss_itc + loss_itm + loss_mlm
            else:
                loss_itc, loss_itm = model(
                    image_illumination, text_input.input_ids, text_input.attention_mask, idx=idx,
                    text_ids_eda=text_input_eda.input_ids, text_atts_eda=text_input_eda.attention_mask
                )
                loss2 = loss_itc + loss_itm

            itc_display_total = loss_itc
            enable_mem = (
                use_embed_memory and (epoch >= mem_start_epoch) and
                (global_step >= mem_warmup) and (global_step % mem_stride == 0)
            )
            if enable_mem and embed_bank is not None:
                try:
                    loss_add, itc_ext, z_img_cur, z_txt_cur = memory_itc_augment(
                        model, image_illumination, text_input, idx, embed_bank, mk_mem_cfg(epoch), want_emb_from_model=True
                    )
                    if loss_add != 0.0:
                        loss2 = loss2 + loss_add
                        itc_display_total = loss_itc + itc_ext
                    embed_bank.push(z_img_cur.detach(), z_txt_cur.detach(), ids=idx)
                except Exception as e:
                    if _is_cuda_oom(e):
                        torch.cuda.empty_cache()
                    else:
                        if not embed_bank_warned:
                            warnings.warn(f"[Memory disabled] {e}")
                            embed_bank_warned = True
                        use_embed_memory = False

            optimizer.zero_grad()
            loss2.backward()
            optimizer.step()
            params_after_task2 = {k: v.clone() for k, v in model.state_dict().items()}

            # ---- Task 3: in-batch mixup ----
            model.load_state_dict(initial_params)
            mixed_images = in_batch_mixup(image, alpha=mixup3_alpha)

            if config.get('mlm', False):
                loss_itc, loss_itm, loss_mlm = model(
                    mixed_images, text_input.input_ids, text_input.attention_mask,
                    text_ids_masked=text_ids_masked, masked_pos=masked_pos, masked_ids=masked_ids,
                    idx=idx, text_ids_eda=text_input_eda.input_ids, text_atts_eda=text_input_eda.attention_mask
                )
                loss3 = loss_itc + loss_itm + loss_mlm
            else:
                loss_itc, loss_itm = model(
                    mixed_images, text_input.input_ids, text_input.attention_mask, idx=idx,
                    text_ids_eda=text_input_eda.input_ids, text_atts_eda=text_input_eda.attention_mask
                )
                loss3 = loss_itc + loss_itm

            itc_display_total = loss_itc
            enable_mem = (
                use_embed_memory and (epoch >= mem_start_epoch) and
                (global_step >= mem_warmup) and (global_step % mem_stride == 0)
            )
            if enable_mem and embed_bank is not None:
                try:
                    loss_add, itc_ext, z_img_cur, z_txt_cur = memory_itc_augment(
                        model, mixed_images, text_input, idx, embed_bank, mk_mem_cfg(epoch), want_emb_from_model=True
                    )
                    if loss_add != 0.0:
                        loss3 = loss3 + loss_add
                        itc_display_total = loss_itc + itc_ext
                    embed_bank.push(z_img_cur.detach(), z_txt_cur.detach(), ids=idx)
                except Exception as e:
                    if _is_cuda_oom(e):
                        torch.cuda.empty_cache()
                    else:
                        if not embed_bank_warned:
                            warnings.warn(f"[Memory disabled] {e}")
                            embed_bank_warned = True
                        use_embed_memory = False

            optimizer.zero_grad()
            loss3.backward()
            optimizer.step()
            params_after_task3 = {k: v.clone() for k, v in model.state_dict().items()}

            # ---- Reptile 聚合 ----
            new_params = {}
            meta_lr = float(config.get('meta_lr', 0.1))
            for k in initial_params.keys():
                new_params[k] = initial_params[k] + meta_lr * (
                    params_after_task1[k] + params_after_task2[k] + params_after_task3[k] - 3 * initial_params[k]
                )
            model.load_state_dict(new_params)

        # ========= 常规阶段 =========
        else:
            if config.get('mlm', False):
                text_ids_masked, masked_pos, masked_ids = mlm(text, text_input, tokenizer, device, mask_generator, config)
                loss_itc, loss_itm, loss_mlm = model(
                    image, text_input.input_ids, text_input.attention_mask,
                    text_ids_masked=text_ids_masked, masked_pos=masked_pos, masked_ids=masked_ids, idx=idx,
                    text_ids_eda=text_input_eda.input_ids, text_atts_eda=text_input_eda.attention_mask
                )
                loss = loss_itc + loss_itm + loss_mlm
            else:
                loss_itc, loss_itm = model(
                    image, text_input.input_ids, text_input.attention_mask, idx=idx,
                    text_ids_eda=text_input_eda.input_ids, text_atts_eda=text_input_eda.attention_mask
                )
                loss = loss_itc + loss_itm

            itc_display_total = loss_itc
            enable_mem = (
                use_embed_memory and (epoch >= mem_start_epoch) and
                (global_step >= mem_warmup) and (global_step % mem_stride == 0)
            )
            if enable_mem:
                try:
                    if embed_bank is None:
                        z_img_tmp = get_img_emb(model, image)
                        embed_bank = EmbeddingMemoryBank(mem_capacity, z_img_tmp.size(1), mem_device, dtype=torch.float16, store_ids=True)
                        z_txt_tmp = get_txt_emb(model, text_input.input_ids, text_input.attention_mask)
                        embed_bank.push(z_img_tmp.detach(), z_txt_tmp.detach(), ids=idx)

                    loss_add, itc_ext, z_img_cur, z_txt_cur = memory_itc_augment(
                        model, image, text_input, idx, embed_bank, mk_mem_cfg(epoch), want_emb_from_model=True
                    )
                    if loss_add != 0.0:
                        loss = loss + loss_add
                        itc_display_total = loss_itc + itc_ext
                    embed_bank.push(z_img_cur.detach(), z_txt_cur.detach(), ids=idx)
                except Exception as e:
                    if _is_cuda_oom(e):
                        torch.cuda.empty_cache()
                    else:
                        if not embed_bank_warned:
                            warnings.warn(f"[Memory disabled] {e}")
                            embed_bank_warned = True
                        use_embed_memory = False

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 调度器 & 日志
        scheduler.step()
        global_step += 1

        metric_logger.update(loss_itc=float(itc_display_total.item()))
        metric_logger.update(loss_itm=float(loss_itm.item()))
        if config.get('mlm', False):
            metric_logger.update(loss_mlm=float(locals().get('loss_mlm', itc_display_total).item()))
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    if config.get('swa', False):
        optimizer.swap_swa_sgd()

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.5f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}



def train_attr(model, data_loader, optimizer, tokenizer, epoch, device, scheduler, config, mask_generator=None):
    model.train()

    base_opt = optimizer
    lookahead_optimizer = Lookahead(base_opt, alpha=0.5, k=6)
    if config.get('swa', False):
        optimizer = SWA(lookahead_optimizer, config['swa_start'], config['swa_freq'], config['swa_lr'])
    else:
        optimizer = lookahead_optimizer


    use_embed_memory = config.get('use_embed_memory', True)
    mem_capacity = int(config.get('mem_capacity', 2048))
    mem_alpha = float(config.get('mem_alpha', 0.4))
    mem_temperature = float(config.get('mem_temperature', 0.07))
    mem_weight = float(config.get('mem_weight', 0.2))
    mem_replace_itc = bool(config.get('mem_replace_itc', False))
    mixup3_alpha = float(config.get('mixup3_alpha', 0.8))

    embed_bank = None
    embed_bank_enabled = use_embed_memory
    embed_bank_warned = False

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_itc', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_attr', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    if config.get('mlm', False):
        metric_logger.add_meter('loss_mlm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50


    base_attr = ['the person is a woman', 'the person is a man',
                 'the person is younger than 18 years old', 'the person is older than 18 years old',
                 'the person with short hair', 'the person with long hair',
                 'the person with a hat', 'the person without a hat',
                 'the person with a backpack', 'the person without a backpack',
                 'the person with a handbag', 'the person without a handbag',
                 'the person with a bag', 'the person without a bag',
                 'the person wears long sleeved upper clothes', 'the person wears short sleeved upper clothes',
                 'the person wears long dress or long pants', 'the person wears short dress or short pants',
                 'the person wears dress or skirt', 'the person wears pants or shorts',
                 'the person wears black upper clothes', 'the person does not wear black upper clothes',
                 'the person wears white upper clothes', 'the person does not wear white upper clothes',
                 'the person wears red upper clothes', 'the person does not wear red upper clothes',
                 'the person wears purple upper clothes', 'the person does not wear purple upper clothes',
                 'the person wears yellow upper clothes', 'the person does not wear yellow upper clothes',
                 'the person wears blue upper clothes', 'the person does not wear blue upper clothes',
                 'the person wears green upper clothes', 'the person does not wear green upper clothes',
                 'the person wears gray upper clothes', 'the person does not wear gray upper clothes',
                 'the person wears black lower clothes', 'the person does not wear black lower clothes',
                 'the person wears white lower clothes', 'the person does not wear white lower clothes',
                 'the person wears purple lower clothes', 'the person does not wear purple lower clothes',
                 'the person wears yellow lower clothes', 'the person does not wear yellow lower clothes',
                 'the person wears blue lower clothes', 'the person does not wear blue lower clothes',
                 'the person wears green lower clothes', 'the person does not wear green lower clothes',
                 'the person wears pink lower clothes', 'the person does not wear pink lower clothes',
                 'the person wears gray lower clothes', 'the person does not wear gray lower clothes',
                 'the person wears brown lower clothes', 'the person does not wear brown lower clothes']

    for i, (image, text, idx, label) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=config['max_tokens'],
                               return_tensors="pt").to(device)
        attr_input = tokenizer(base_attr, padding='longest', max_length=config['max_tokens'],
                               return_tensors="pt").to(device)


        if epoch < 10:
            initial_params = {k: v.clone() for k, v in model.state_dict().items()}

            # ---- Task 1: blur ----
            image_blur = apply_gaussian_blur(image)
            if config.get('mlm', False):
                text_ids_masked, masked_pos, masked_ids = mlm(text, text_input, tokenizer, device, mask_generator, config)
                attr_text_ids_masked, attr_masked_pos, attr_masked_ids = mlm(base_attr, attr_input, tokenizer, device, mask_generator, config, True)
                loss_itc, loss_itm, loss_mlm, loss_attr = model(
                    image_blur, text_input.input_ids, text_input.attention_mask,
                    text_ids_masked=text_ids_masked, masked_pos=masked_pos, masked_ids=masked_ids, idx=idx,
                    attr_text_ids=attr_input.input_ids, attr_text_atts=attr_input.attention_mask,
                    attr_text_ids_masked=attr_text_ids_masked, attr_masked_pos=attr_masked_pos,
                    attr_masked_ids=attr_masked_ids, label=label
                )
                loss1 = loss_itc + loss_itm + loss_mlm + config['t'] * loss_attr
            else:
                loss_itc, loss_itm, loss_attr = model(
                    image_blur, text_input.input_ids, text_input.attention_mask, idx=idx,
                    attr_text_ids=attr_input.input_ids, attr_text_atts=attr_input.attention_mask, label=label
                )
                loss1 = loss_itc + loss_itm + config['t'] * loss_attr

            if embed_bank_enabled:
                try:
                    with torch.no_grad():
                        z_img_cur = get_img_emb(model, image_blur)
                        z_txt_cur = get_txt_emb(model, text_input.input_ids, text_input.attention_mask)
                    if embed_bank is None:
                        embed_bank = EmbeddingMemoryBank(mem_capacity, z_img_cur.size(1), device, dtype=torch.float16)
                    mem = embed_bank.sample(z_img_cur.size(0))
                    if mem is not None:
                        _, mem_txt = mem
                        tilde_txt = mixup_embed(z_txt_cur, mem_txt, alpha=mem_alpha)
                        keys_txt = torch.cat([z_txt_cur, tilde_txt.detach()], dim=0)
                        loss_itc_ext = compute_clip_itc_loss(z_img_cur, keys_txt, model=model,
                                                             default_temperature=mem_temperature)
                        if mem_replace_itc:
                            loss1 = loss1 - loss_itc + loss_itc_ext
                            loss_itc = loss_itc_ext
                        else:
                            loss1 = loss1 + mem_weight * loss_itc_ext
                            loss_itc = (loss_itc + mem_weight * loss_itc_ext).detach()
                    embed_bank.push(z_img_cur, z_txt_cur)
                except Exception as e:
                    if not embed_bank_warned:
                        warnings.warn(f"[Memory disabled] {e}")
                        embed_bank_warned = True
                    embed_bank_enabled = False

            optimizer.zero_grad()
            loss1.backward()
            optimizer.step()
            params_after_task1 = {k: v.clone() for k, v in model.state_dict().items()}

            # ---- Task 2: illumination ----
            model.load_state_dict(initial_params)
            image_illumination = apply_random_illumination(image)
            if config.get('mlm', False):
                text_ids_masked, masked_pos, masked_ids = mlm(text, text_input, tokenizer, device, mask_generator, config)
                attr_text_ids_masked, attr_masked_pos, attr_masked_ids = mlm(base_attr, attr_input, tokenizer, device, mask_generator, config, True)
                loss_itc, loss_itm, loss_mlm, loss_attr = model(
                    image_illumination, text_input.input_ids, text_input.attention_mask,
                    text_ids_masked=text_ids_masked, masked_pos=masked_pos, masked_ids=masked_ids, idx=idx,
                    attr_text_ids=attr_input.input_ids, attr_text_atts=attr_input.attention_mask,
                    attr_text_ids_masked=attr_text_ids_masked, attr_masked_pos=attr_masked_pos,
                    attr_masked_ids=attr_masked_ids, label=label
                )
                loss2 = loss_itc + loss_itm + loss_mlm + config['t'] * loss_attr
            else:
                loss_itc, loss_itm, loss_attr = model(
                    image_illumination, text_input.input_ids, text_input.attention_mask, idx=idx,
                    attr_text_ids=attr_input.input_ids, attr_text_atts=attr_input.attention_mask, label=label
                )
                loss2 = loss_itc + loss_itm + config['t'] * loss_attr

            if embed_bank_enabled:
                try:
                    with torch.no_grad():
                        z_img_cur = get_img_emb(model, image_illumination)
                        z_txt_cur = get_txt_emb(model, text_input.input_ids, text_input.attention_mask)
                    if embed_bank is None:
                        embed_bank = EmbeddingMemoryBank(mem_capacity, z_img_cur.size(1), device, dtype=torch.float16)
                    mem = embed_bank.sample(z_img_cur.size(0))
                    if mem is not None:
                        _, mem_txt = mem
                        tilde_txt = mixup_embed(z_txt_cur, mem_txt, alpha=mem_alpha)
                        keys_txt = torch.cat([z_txt_cur, tilde_txt.detach()], dim=0)
                        loss_itc_ext = compute_clip_itc_loss(z_img_cur, keys_txt, model=model,
                                                             default_temperature=mem_temperature)
                        if mem_replace_itc:
                            loss2 = loss2 - loss_itc + loss_itc_ext
                            loss_itc = loss_itc_ext
                        else:
                            loss2 = loss2 + mem_weight * loss_itc_ext
                            loss_itc = (loss_itc + mem_weight * loss_itc_ext).detach()
                    embed_bank.push(z_img_cur, z_txt_cur)
                except Exception as e:
                    if not embed_bank_warned:
                        warnings.warn(f"[Memory disabled] {e}")
                        embed_bank_warned = True
                    embed_bank_enabled = False

            optimizer.zero_grad()
            loss2.backward()
            optimizer.step()
            params_after_task2 = {k: v.clone() for k, v in model.state_dict().items()}

            # ---- Task 3: in-batch mixup ----
            model.load_state_dict(initial_params)
            mixed_images = in_batch_mixup(image, alpha=mixup3_alpha)
            if config.get('mlm', False):
                text_ids_masked, masked_pos, masked_ids = mlm(text, text_input, tokenizer, device, mask_generator, config)
                attr_text_ids_masked, attr_masked_pos, attr_masked_ids = mlm(base_attr, attr_input, tokenizer, device, mask_generator, config, True)
                loss_itc, loss_itm, loss_mlm, loss_attr = model(
                    mixed_images, text_input.input_ids, text_input.attention_mask,
                    text_ids_masked=text_ids_masked, masked_pos=masked_pos, masked_ids=masked_ids, idx=idx,
                    attr_text_ids=attr_input.input_ids, attr_text_atts=attr_input.attention_mask,
                    attr_text_ids_masked=attr_text_ids_masked, attr_masked_pos=attr_masked_pos,
                    attr_masked_ids=attr_masked_ids, label=label
                )
                loss3 = loss_itc + loss_itm + loss_mlm + config['t'] * loss_attr
            else:
                loss_itc, loss_itm, loss_attr = model(
                    mixed_images, text_input.input_ids, text_input.attention_mask, idx=idx,
                    attr_text_ids=attr_input.input_ids, attr_text_atts=attr_input.attention_mask, label=label
                )
                loss3 = loss_itc + loss_itm + config['t'] * loss_attr

            if embed_bank_enabled:
                try:
                    with torch.no_grad():
                        z_img_cur = get_img_emb(model, mixed_images)
                        z_txt_cur = get_txt_emb(model, text_input.input_ids, text_input.attention_mask)
                    if embed_bank is None:
                        embed_bank = EmbeddingMemoryBank(mem_capacity, z_img_cur.size(1), device, dtype=torch.float16)
                    mem = embed_bank.sample(z_img_cur.size(0))
                    if mem is not None:
                        _, mem_txt = mem
                        tilde_txt = mixup_embed(z_txt_cur, mem_txt, alpha=mem_alpha)
                        keys_txt = torch.cat([z_txt_cur, tilde_txt.detach()], dim=0)
                        loss_itc_ext = compute_clip_itc_loss(z_img_cur, keys_txt, model=model,
                                                             default_temperature=mem_temperature)
                        if mem_replace_itc:
                            loss3 = loss3 - loss_itc + loss_itc_ext
                            loss_itc = loss_itc_ext
                        else:
                            loss3 = loss3 + mem_weight * loss_itc_ext
                            loss_itc = (loss_itc + mem_weight * loss_itc_ext).detach()
                    embed_bank.push(z_img_cur, z_txt_cur)
                except Exception as e:
                    if not embed_bank_warned:
                        warnings.warn(f"[Memory disabled] {e}")
                        embed_bank_warned = True
                    embed_bank_enabled = False

            optimizer.zero_grad()
            loss3.backward()
            optimizer.step()
            params_after_task3 = {k: v.clone() for k, v in model.state_dict().items()}

            # Reptile 聚合
            new_params = {}
            meta_lr = float(config.get('meta_lr', 0.1))
            for k in initial_params.keys():
                new_params[k] = initial_params[k] + meta_lr * (
                    params_after_task1[k] + params_after_task2[k] + params_after_task3[k] - 3 * initial_params[k]
                )
            model.load_state_dict(new_params)

        # ========= 10 之后：常规训练 =========
        else:
            if config.get('mlm', False):
                text_ids_masked, masked_pos, masked_ids = mlm(text, text_input, tokenizer, device, mask_generator, config)
                attr_text_ids_masked, attr_masked_pos, attr_masked_ids = mlm(base_attr, attr_input, tokenizer, device, mask_generator, config, True)
                loss_itc, loss_itm, loss_mlm, loss_attr = model(
                    image, text_input.input_ids, text_input.attention_mask,
                    text_ids_masked=text_ids_masked, masked_pos=masked_pos, masked_ids=masked_ids, idx=idx,
                    attr_text_ids=attr_input.input_ids, attr_text_atts=attr_input.attention_mask,
                    attr_text_ids_masked=attr_text_ids_masked, attr_masked_pos=attr_masked_pos,
                    attr_masked_ids=attr_masked_ids, label=label
                )
                loss = loss_itc + loss_itm + loss_mlm + config['t'] * loss_attr
            else:
                loss_itc, loss_itm, loss_attr = model(
                    image, text_input.input_ids, text_input.attention_mask, idx=idx,
                    attr_text_ids=attr_input.input_ids, attr_text_atts=attr_input.attention_mask, label=label
                )
                loss = loss_itc + loss_itm + config['t'] * loss_attr

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        metric_logger.update(loss_itc=float(loss_itc.item()))
        metric_logger.update(loss_itm=float(loss_itm.item()))
        if config.get('mlm', False):
            metric_logger.update(loss_mlm=float(loss_mlm.item()))
        metric_logger.update(loss_attr=float(loss_attr.item()))
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    if config.get('swa', False):
        optimizer.swap_swa_sgd()

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.5f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}
