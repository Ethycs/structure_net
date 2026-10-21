"""NCM generate shim: two scalar allocations changed to capture-safe GPU fills.

Adapted from HazyResearch/ncm, commit c70274273f489abc73832b2b98584a8483b6b203.
Original is MIT-licensed; see the adjacent ncm_capture_LICENSE and study record.
Every other statement in this function is unchanged from the official source.
"""
import torch
import torch.nn.functional as F

@torch.no_grad()
def generate(self, idx, prefix_tokens, temperature=1.0, top_k=50, top_p=0.9,
                 use_kv_cache=False, cfg_scale=0.0, cfg_null_eos=50256):
        assert not (self.query_level0 and use_kv_cache), \
            "query_level0 generation supports the eager path only"

        dev = prefix_tokens.device

        self._cfg = None
        # CFG: double the batch to [document; null] before the prefix is encoded; cfg_scale=0 builds no null branch.
        if cfg_scale > 0.0:
            if self.prefix_pad_id is None:
                raise ValueError("cfg_scale needs prefix_pad_id to build the null prefix")
            _cB = prefix_tokens.shape[0]
            _null = torch.full_like(prefix_tokens, self.prefix_pad_id)
            _null[:, 0] = cfg_null_eos
            prefix_tokens = torch.cat([prefix_tokens, _null], dim=0)
            if idx is not None:
                idx = torch.cat([idx, idx], dim=0)
            self._cfg = {"scale": float(cfg_scale), "B": _cB,
                         "nlev": len(self.levels), "level": 0}

        B = prefix_tokens.shape[0]
        vq = self.vqvae_local[0].vq
        z_hat = torch.zeros(B, self.levels[-1], self.vqvae_local[0].embed_dim, device=dev)

        if self.use_ctx_encoder:
            prefix_emb = self._encode_context(prefix_tokens[:, :self.prefix_len])
        else:
            if bool(getattr(self.config, 'raw_token_prefix', False)):
                _enc = self.vqvae_local[0].encoder
                if bool(getattr(self.config, 'contextual_prefix', False)) and \
                        len(getattr(_enc, 'blocks', [])) > 0:
                    z_prefix = _enc.forward_nopos(prefix_tokens)[:, :self.prefix_len, :]
                else:
                    z_prefix = _enc.wte(prefix_tokens)[:, :self.prefix_len, :]
            else:
                z_prefix = self.vqvae_local[0].encode_z(prefix_tokens)[:, :self.prefix_len, :]
            prefix_emb = self.transformer.wte(z_prefix)
        if not self.use_rope:
            prefix_emb = prefix_emb + self.wpe(self.wp_prefix)

        if self.config.use_level_encoding:
            level_pos_embed = self.transformer.wle(self.lvl_1L)
        else:
            level_pos_embed = torch.zeros(1, self.L - self.levels[0], self.config.n_embd, device=dev)

        if not self.use_rope:
            level_pos_embed = level_pos_embed + self.wpe(self.wp_1L)

        P = self.prefix_len

        _pbias = None
        if self.prefix_pad_id is not None:
            _pm = (prefix_tokens[:, :P] == self.prefix_pad_id)
            _pbias = torch.zeros(_pm.shape[0], 1, 1, P, device=dev)
            _pbias[:, :, :, :P] = torch.where(
                _pm[:, None, None, :],
                torch.full((), float('-inf'), device=dev),
                torch.full((), 0.0, device=dev))

        if idx is None and self.query_level0:
            l0 = self.levels[0]
            _q0 = self.build_l0_query_input(B, prefix_emb.dtype)
            if self.use_post_upsample_conv:
                _q0 = self._apply_post_upsample_conv_single(_q0, 0)
            q = self.transformer.wte(_q0)
            if not self.use_rope:
                q = q + self.wpe(self.wp_1L[:, :l0])
            if self.config.use_level_encoding:
                q = q + self.transformer.wle(self.lvl_1L[:, :l0])
            h = torch.cat([prefix_emb, q], dim=1)
            m0 = torch.zeros(1, 1, P + l0, P + l0, device=dev)
            m0[:, :, :P, P:] = float('-inf')
            if self.l0_ar:
                m0[:, :, P:, P:] = torch.triu(
                    torch.full((l0, l0), float('-inf'), device=dev), diagonal=1)
            if _pbias is not None:
                m0 = m0 + F.pad(_pbias, (0, l0))
            rope_0 = (self.rope_cos_gen_0, self.rope_sin_gen_0) if self.use_rope else None
            if self.l0_ar:
                _E0 = vq.embeddings[0]
                base_q = self.build_l0_query_input(B, prefix_emb.dtype)
                x_in, codes = base_q, []
                for t in range(l0):
                    q = self.transformer.wte(x_in)
                    if not self.use_rope:
                        q = q + self.wpe(self.wp_1L[:, :l0])
                    if self.config.use_level_encoding:
                        q = q + self.transformer.wle(self.lvl_1L[:, :l0])
                    h = torch.cat([prefix_emb, q], dim=1)
                    for block in self.transformer.blocks:
                        h = block(h, attn_mask=m0, rope_cos_sin=rope_0)
                    h = self.transformer.ln_f(h)
                    codes.append(self._sample_level(
                        self.lm_heads[0](h[:, P + t:P + t + 1, :]),
                        temperature, top_k, top_p))
                    if t + 1 < l0:
                        known = torch.cat(codes, dim=1)
                        x_in = torch.cat([base_q[:, :1], _E0(known).to(base_q.dtype),
                                          base_q[:, t + 2:]], dim=1)
                idx = torch.cat(codes, dim=1)
            else:
                for block in self.transformer.blocks:
                    h = block(h, attn_mask=m0, rope_cos_sin=rope_0)
                h = self.transformer.ln_f(h)
                if self._cfg is not None: self._cfg["level"] = 0
                idx = self._sample_level(self.lm_heads[0](h[:, P:, :]), temperature, top_k, top_p)

        if idx is None:
            assert self.predict_level0, "generate(idx=None) requires model.predict_level0 or query_level0"
            rope_p = (self.rope_cos_gen_0, self.rope_sin_gen_0) if self.use_rope else None
            h = prefix_emb
            pfx_mask = torch.zeros(1, 1, P, P, device=dev)
            if _pbias is not None:
                pfx_mask = pfx_mask + _pbias
            for block in self.transformer.blocks:
                h = block(h, attn_mask=pfx_mask, rope_cos_sin=rope_p)
            h = self.transformer.ln_f(h)
            l0_logits = self.lm_head_l0(self._level0_slice(h, _pm if self.prefix_pad_id is not None else None))
            if self._cfg is not None: self._cfg["level"] = 0
            idx = self._sample_level(l0_logits, temperature, top_k, top_p)

        generated_idx = [idx]
        idx_next = idx
        if use_kv_cache:
            self._set_kv_caching(True)

        try:
            if use_kv_cache:
                rope_p = (
                    self.rope_cos_gen_0, self.rope_sin_gen_0,
                ) if self.use_rope else None
                h = prefix_emb
                pfx_mask = torch.zeros(1, 1, P, P, device=prefix_emb.device)
                for block in self.transformer.blocks:
                    h = block(h, attn_mask=pfx_mask, rope_cos_sin=rope_p)
                prefix_kv = [(b.attn.cached_k, b.attn.cached_v) for b in self.transformer.blocks]

            level_acum = self.levels[0] if self.query_level0 else 0
            for level_idx in range(len(self.levels)-1):
                if self._cfg is not None:
                    self._cfg["level"] = level_idx + 1

                z_q_level = vq.embeddings[level_idx](idx_next)
                z_hat, x = vq.get_next_autoregressive_input(level_idx, z_hat, z_q_level)

                if self.use_post_upsample_conv:
                    x = self._apply_post_upsample_conv_single(x, level_idx + (1 - self._lvl_first))

                x = self.transformer.wte(x) + level_pos_embed[:, level_acum:level_acum + self.levels[level_idx+1], :]

                L_cur = self.levels[level_idx + 1]
                # Heads and RoPE buffers are indexed by absolute level when level 0 occupies slot 0.
                _hi = level_idx + (1 - self._lvl_first)
                rope_k = (
                    getattr(self, f'rope_cos_gen_{_hi}'),
                    getattr(self, f'rope_sin_gen_{_hi}'),
                ) if self.use_rope else None

                if use_kv_cache:
                    for blk, (pk, pv) in zip(self.transformer.blocks, prefix_kv):
                        blk.attn.cached_k, blk.attn.cached_v = pk, pv
                    lvl_mask = torch.zeros(1, 1, L_cur, P + L_cur, device=x.device)
                    for block in self.transformer.blocks:
                        x = block(x, attn_mask=lvl_mask, rope_cos_sin=rope_k)
                    x = self.transformer.ln_f(x)
                    logits = self.lm_heads[_hi](x)
                    idx_next = self._sample_level(logits, temperature, top_k, top_p)
                else:
                    logits = self._decode_level_logits(x, prefix_emb, level_idx, _pbias, hidx=_hi)
                    idx_next = self._sample_level(logits, temperature, top_k, top_p)

                generated_idx.append(idx_next)
                level_acum += self.levels[level_idx+1]
        finally:
            if use_kv_cache:
                self._set_kv_caching(False)

        out = torch.cat(generated_idx, dim=1)
        if self._cfg is not None:
            out = out[:self._cfg["B"]]
        self._cfg = None
        return out
