#include "llama.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <sstream>

// LLM_CHAT_TEMPLATE_GRANITE

static struct llama_model_params mparams;
static struct llama_model* model;

static struct llama_context_params cparams;
static struct llama_context* ctx;

struct llama_sampler_chain_params sampler_chain_params;
struct llama_sampler * sampler;

static void init() {
#if 0
    sampler = llama_sampler_init_greedy();
#else
    sampler_chain_params = llama_sampler_chain_default_params();
    sampler = llama_sampler_chain_init(sampler_chain_params);
    const float temperature = 1.00f;
    const uint32_t seed = 153; // LLAMA_DEFAULT_SEED will actually use a random seed
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(50));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.99f, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(seed));
#endif
}

static void deinit() {
    llama_sampler_free(sampler);    
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
}

static bool load_model(const char* model_path) {
    mparams = llama_model_default_params();
    model = llama_load_model_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "Failed to load model '%s'\n", model_path);
        return false;
    }
    printf("Model loaded\n");
    cparams = llama_context_default_params();
    ctx = llama_new_context_with_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "Failed to create context\n");
        llama_free_model(model);
        model = nullptr;
        return false;
    }
    printf("Model loaded and context created.\n");
    return true;
}

static std::string token_to_piece(const struct llama_context * ctx, llama_token token, bool special = false) {
    std::string piece;
    piece.resize(piece.capacity());  // using string internal cache, 15 bytes + '\n'
    const int n_chars = llama_token_to_piece(llama_get_model(ctx), token, &piece[0], piece.size(), 0, special);
    if (n_chars < 0) {
        piece.resize(-n_chars);
        int check = llama_token_to_piece(llama_get_model(ctx), token, &piece[0], piece.size(), 0, special);
        GGML_ASSERT(check == -n_chars);
    } else {
        piece.resize(n_chars);
    }
    return piece;
}

static std::string string_from(const struct llama_context * ctx, const std::vector<llama_token> & tokens) {
    std::stringstream buf;
    buf << "[ ";
    bool first = true;
    for (const auto & token : tokens) {
        if (!first) {
            buf << ", ";
        } else {
            first = false;
        }
        auto detokenized = token_to_piece(ctx, token);
        detokenized.erase(
            std::remove_if(
                detokenized.begin(),
                detokenized.end(),
                [](const unsigned char c) { return !std::isprint(c); }),
            detokenized.end());
        buf << "'" << detokenized << "'"
            << ":" << std::to_string(token);
    }
    buf << " ]";
    return buf.str();
}

static std::string detokenize(llama_context * ctx, const std::vector<llama_token> & tokens, 
                              bool remove_special = false, bool unparse_special = true) {
    std::string text;
    text.resize(std::max(text.capacity(), tokens.size()));
    int32_t n_chars = llama_detokenize(llama_get_model(ctx), 
                                       tokens.data(), (int32_t)tokens.size(), 
                                       &text[0], (int32_t)text.size(), 
                                       remove_special, unparse_special);
    if (n_chars < 0) {
        text.resize(-n_chars);
        n_chars = llama_detokenize(llama_get_model(ctx), 
                                   tokens.data(), (int32_t)tokens.size(), 
                                   &text[0], (int32_t)text.size(), 
                                   remove_special, unparse_special);
        // whitespace trimming is performed after per-token detokenization
        GGML_ASSERT(n_chars <= (int32_t)text.size());  
    }
    text.resize(n_chars);
    // NOTE: the original tokenizer decodes bytes after collecting the pieces.
    return text;
}

static bool inference() {
    llama_chat_message conversation[] = {
        {"role", "User"},
        {"content", "Please list one IBM Research laboratory located in the United States. You should only output its name and location."}
    };
    std::vector<llama_chat_message> messages;
    std::vector<char> formatted(llama_n_ctx(ctx));
    int new_len = llama_chat_apply_template(model, "granite", 
                                            conversation, 2, 
                                            true, formatted.data(), formatted.size());
    if (new_len > (int)formatted.size()) {
        formatted.resize(new_len);
        new_len = llama_chat_apply_template(model, "granite", 
                                            conversation, 2, 
                                            true, formatted.data(), formatted.size());
    }
    if (new_len < 0) {
        fprintf(stderr, "failed to apply the chat template\n");
        return false;
    }
    formatted.resize(new_len);
    // this is what actually expected from formatted string:
    // https://huggingface.co/ibm-granite/granite-3.0-1b-a400m-instruct

    static const char* fs_0 = "<|start_of_role|>user<|end_of_role|>"
    "Please list one IBM Research laboratory located in the United States. "
    "You should only output its name and location."
    "<|end_of_text|>"
    "<|start_of_role|>assistant<|end_of_role|>";

    static const char* fs_1 = "<|start_of_role|>user<|end_of_role|>"
    "Explain battery charge level to voltage ratio for different types of batteries."
    "<|end_of_text|>"
    "<|start_of_role|>assistant<|end_of_role|>";

    static const char* fs  = "<|start_of_role|>user<|end_of_role|>"
    "Generate a story about Cinderela and her fairy godmother."
    "<|end_of_text|>"
    "<|start_of_role|>assistant<|end_of_role|>";

    formatted.resize(strlen(fs));
    memcpy(formatted.data(), fs, formatted.size());
    // Python sample code outputs:
    // "545 East 9th Street<|end_of_text|>"
    int32_t len = (int32_t)formatted.size();
    printf("formatted: \"%s\"\n", formatted.data());
    bool add_special = false;
    bool parse_special = true;
    int n_tokens = len + 2 * add_special;
    std::vector<llama_token> tokens(n_tokens);
    n_tokens = llama_tokenize(model, formatted.data(), len, tokens.data(), tokens.size(), 
                              add_special, parse_special);
    if (n_tokens < 0) {
        fprintf(stderr, "Failed to tokenize the prompt.\n");
        return false;
    }
    tokens.resize(n_tokens);
    printf("tokens: %s\n", string_from(ctx, tokens).c_str());
    printf("detokenize: \"%s\"\n", detokenize(ctx, tokens).c_str());
    uint32_t n_ctx = llama_n_ctx(ctx);
    printf("n_ctx: %d\n", n_ctx);
    int n_past = 0;
    std::vector<llama_token> embd;
    if (!tokens.empty()) {
        // Use tokens from the prompt
        embd.insert(embd.end(), tokens.begin(), tokens.end());
        tokens.clear();
    }
    int n_predict = 512; // Number of tokens to generate
    int n_remaining = n_predict;
    llama_token last = 0;
    while (n_remaining > 0) {
        int n_ctx_used = llama_get_kv_cache_used_cells(ctx);
        if (n_ctx_used + embd.size() > n_ctx) {
            fprintf(stderr, "context size exceeded\n");
            break;
        }
        if (llama_decode(ctx, llama_batch_get_one(embd.data(), embd.size())) != 0) {
            fprintf(stderr, "Failed to evaluate tokens. last=%d\n", last);
            break;
        }
        n_past += embd.size();
        llama_token id = llama_sampler_sample(sampler, ctx, -1);
        last = id;
        // printf("id: %d\n", id);
        if (llama_token_is_eog(model, id)) {
            printf("\n<end of text>\n");
            break;
        }
        std::string token_str = token_to_piece(ctx, id);
//      printf("%zd %d \"%s\"\n", embd.size(), id, token_str.c_str());
        embd.push_back(id);
        n_remaining--;
    }
    printf("\n");
    printf("result: \"%s\"\n", detokenize(ctx, embd).c_str());
    return true;
}

int main(int argc, char** argv) {
    assert(argc > 1);
    // only print errors
    llama_log_set([](enum ggml_log_level level, const char * text, void * /* user_data */) {
        if (level >= GGML_LOG_LEVEL_DEBUG) {
            fprintf(stderr, "%s", text);
        }
    }, nullptr);
    ggml_backend_load_all();
    ggml_backend_init_best();
    if (!load_model(argv[1])) { return 1; }
    printf("Model loaded and context created.\n");
    init();
    inference();
    deinit();
    return 0;
}

