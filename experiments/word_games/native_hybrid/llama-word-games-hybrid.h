#pragma once
#include "ggml.h"
#include "ggml-backend.h"
struct llama_model;
void word_games_hybrid_setup(const llama_model & model, int threads);
bool word_games_hybrid_active(int layer, int64_t tokens);
bool word_games_hybrid_enabled();
void word_games_hybrid_set_user(ggml_backend_sched_eval_callback cb, void * ud);
bool word_games_hybrid_callback(ggml_tensor * t, bool ask, void * ud);
ggml_tensor * word_games_hybrid_build(ggml_context * ctx, ggml_cgraph * gf,
    ggml_backend_sched_t sched, ggml_tensor * input, ggml_tensor * ids,
    ggml_tensor * weights, int layer);
