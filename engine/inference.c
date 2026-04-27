/*
 * inference.c — GuppyLM CPU inference engine (single-file, static)
 *
 * Loads fp16 weights packed by model/export_guppy.py and runs
 * autoregressive sampling with a vanilla transformer:
 *   - LayerNorm (pre-norm)
 *   - Learned positional embeddings
 *   - ReLU FFN
 *   - Weight-tied LM head
 *
 * Compile via engine/Makefile (statically linked against musl).
 *
 * Weights are linked in via objcopy as a single blob:
 *   _binary_weights_bin_start .. _binary_weights_bin_end
 *
 * Usage: ./guppy "your prompt"
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "bytelevel.h"

/* ===== Embedded weights blob ===== */
extern const unsigned char _binary_weights_bin_start[];
extern const unsigned char _binary_weights_bin_end[];

/* ===== Binary format ===== */
#define MAGIC "GPPY"
#define FORMAT_VERSION 1u

/* ===== Runtime config (read from header) ===== */
static int VOCAB_SIZE;
static int D_MODEL;
static int N_HEADS;
static int N_LAYERS;
static int D_FF;
static int MAX_SEQ_LEN;
static int HEAD_DIM;

/* Special token IDs (read from header) */
static int BOS_ID;
static int EOS_ID;
static int PAD_ID;
static int UNK_ID;

/* ===== Vocabulary =====
 *
 * Stored as a flat blob of (len:u16, bytes:len) entries, in token-id order.
 * For phase 1 we use a simple longest-match BPE encoder; this is not
 * byte-identical to HuggingFace tokenizers but works for the chat path.
 */
typedef struct {
    char*  bytes;      /* pointer into vocab_blob */
    int    len;
} VocabEntry;

static VocabEntry* vocab;          /* [VOCAB_SIZE] */
static char*       vocab_blob;     /* contiguous storage of all token strings */

/* ===== Weights =====
 *
 * All matrices are row-major fp32 after load. We unpack from fp16 on disk
 * to keep the embedded blob small (~17 MB for an 8.7M-param model).
 */
static float* tok_emb;        /* [VOCAB_SIZE x D_MODEL] (also tied to LM head) */
static float* pos_emb;        /* [MAX_SEQ_LEN x D_MODEL] */

typedef struct {
    /* LayerNorm 1 (pre-attention) */
    float* ln1_g;             /* [D_MODEL] */
    float* ln1_b;             /* [D_MODEL] */
    /* QKV: [3*D_MODEL x D_MODEL] + bias [3*D_MODEL] */
    float* qkv_w;
    float* qkv_b;
    /* Output proj: [D_MODEL x D_MODEL] + bias [D_MODEL] */
    float* proj_w;
    float* proj_b;
    /* LayerNorm 2 (pre-FFN) */
    float* ln2_g;
    float* ln2_b;
    /* FFN: w1 [D_FF x D_MODEL] + b1 [D_FF], w2 [D_MODEL x D_FF] + b2 [D_MODEL] */
    float* w1;
    float* b1;
    float* w2;
    float* b2;
} Block;

static Block* blocks;
static float* ln_f_g;             /* final LayerNorm gain */
static float* ln_f_b;             /* final LayerNorm bias */

/* ===== Reader ===== */
typedef struct {
    const unsigned char* data;
    size_t offset;
    size_t size;
} Reader;

static void rd(Reader* r, void* dst, size_t n) {
    if (r->offset + n > r->size) {
        fprintf(stderr, "weights: short read at offset %zu (need %zu, have %zu)\n",
                r->offset, n, r->size - r->offset);
        exit(1);
    }
    memcpy(dst, r->data + r->offset, n);
    r->offset += n;
}
static uint32_t rd_u32(Reader* r) { uint32_t v; rd(r, &v, 4); return v; }
static int32_t  rd_i32(Reader* r) { int32_t  v; rd(r, &v, 4); return v; }
static uint16_t rd_u16(Reader* r) { uint16_t v; rd(r, &v, 2); return v; }

static float fp16_to_fp32(uint16_t h) {
    uint32_t s = (h >> 15) & 1;
    uint32_t e = (h >> 10) & 0x1F;
    uint32_t f = h & 0x3FF;
    if (e == 0) {
        if (f == 0) return s ? -0.0f : 0.0f;
        return ldexpf((float)f, -24) * (s ? -1.0f : 1.0f);
    }
    if (e == 31) {
        if (f == 0) return s ? -INFINITY : INFINITY;
        return NAN;
    }
    float v = ldexpf((float)(f + 1024), (int)e - 25);
    return s ? -v : v;
}

static void rd_fp16_array(Reader* r, float* out, size_t n) {
    for (size_t i = 0; i < n; i++) {
        uint16_t h; rd(r, &h, 2);
        out[i] = fp16_to_fp32(h);
    }
}

static float* alloc_f(size_t n) {
    float* p = (float*)malloc(n * sizeof(float));
    if (!p) { fprintf(stderr, "OOM (%zu floats)\n", n); exit(1); }
    return p;
}

/* ===== Load weights ===== */
static void load_weights(void) {
    Reader r = {
        .data   = _binary_weights_bin_start,
        .offset = 0,
        .size   = (size_t)(_binary_weights_bin_end - _binary_weights_bin_start),
    };

    char magic[5] = {0};
    rd(&r, magic, 4);
    if (memcmp(magic, MAGIC, 4) != 0) {
        fprintf(stderr, "bad magic: %s (expected %s)\n", magic, MAGIC);
        exit(1);
    }
    uint32_t version = rd_u32(&r);
    if (version != FORMAT_VERSION) {
        fprintf(stderr, "unsupported weights version: %u\n", version);
        exit(1);
    }

    VOCAB_SIZE  = rd_i32(&r);
    D_MODEL     = rd_i32(&r);
    N_HEADS     = rd_i32(&r);
    N_LAYERS    = rd_i32(&r);
    D_FF        = rd_i32(&r);
    MAX_SEQ_LEN = rd_i32(&r);
    HEAD_DIM    = D_MODEL / N_HEADS;

    BOS_ID = rd_i32(&r);
    EOS_ID = rd_i32(&r);
    PAD_ID = rd_i32(&r);
    UNK_ID = rd_i32(&r);

    /* Vocab: [u32 total_bytes] [for each token: u16 len, bytes] */
    uint32_t vocab_bytes = rd_u32(&r);
    vocab_blob = (char*)malloc(vocab_bytes);
    if (!vocab_blob) { fprintf(stderr, "OOM vocab\n"); exit(1); }
    vocab = (VocabEntry*)calloc(VOCAB_SIZE, sizeof(VocabEntry));

    size_t blob_pos = 0;
    for (int i = 0; i < VOCAB_SIZE; i++) {
        uint16_t L = rd_u16(&r);
        if (blob_pos + L > vocab_bytes) {
            fprintf(stderr, "vocab blob overflow\n"); exit(1);
        }
        rd(&r, vocab_blob + blob_pos, L);
        vocab[i].bytes = vocab_blob + blob_pos;
        vocab[i].len   = L;
        blob_pos += L;
    }

    /* Token + positional embeddings */
    tok_emb = alloc_f((size_t)VOCAB_SIZE * D_MODEL);
    rd_fp16_array(&r, tok_emb, (size_t)VOCAB_SIZE * D_MODEL);

    pos_emb = alloc_f((size_t)MAX_SEQ_LEN * D_MODEL);
    rd_fp16_array(&r, pos_emb, (size_t)MAX_SEQ_LEN * D_MODEL);

    /* Blocks */
    blocks = (Block*)calloc(N_LAYERS, sizeof(Block));
    for (int l = 0; l < N_LAYERS; l++) {
        Block* b = &blocks[l];
        b->ln1_g = alloc_f(D_MODEL); rd_fp16_array(&r, b->ln1_g, D_MODEL);
        b->ln1_b = alloc_f(D_MODEL); rd_fp16_array(&r, b->ln1_b, D_MODEL);

        b->qkv_w = alloc_f((size_t)3 * D_MODEL * D_MODEL);
        rd_fp16_array(&r, b->qkv_w, (size_t)3 * D_MODEL * D_MODEL);
        b->qkv_b = alloc_f(3 * D_MODEL);
        rd_fp16_array(&r, b->qkv_b, 3 * D_MODEL);

        b->proj_w = alloc_f((size_t)D_MODEL * D_MODEL);
        rd_fp16_array(&r, b->proj_w, (size_t)D_MODEL * D_MODEL);
        b->proj_b = alloc_f(D_MODEL);
        rd_fp16_array(&r, b->proj_b, D_MODEL);

        b->ln2_g = alloc_f(D_MODEL); rd_fp16_array(&r, b->ln2_g, D_MODEL);
        b->ln2_b = alloc_f(D_MODEL); rd_fp16_array(&r, b->ln2_b, D_MODEL);

        b->w1 = alloc_f((size_t)D_FF * D_MODEL);
        rd_fp16_array(&r, b->w1, (size_t)D_FF * D_MODEL);
        b->b1 = alloc_f(D_FF);
        rd_fp16_array(&r, b->b1, D_FF);

        b->w2 = alloc_f((size_t)D_MODEL * D_FF);
        rd_fp16_array(&r, b->w2, (size_t)D_MODEL * D_FF);
        b->b2 = alloc_f(D_MODEL);
        rd_fp16_array(&r, b->b2, D_MODEL);
    }

    ln_f_g = alloc_f(D_MODEL); rd_fp16_array(&r, ln_f_g, D_MODEL);
    ln_f_b = alloc_f(D_MODEL); rd_fp16_array(&r, ln_f_b, D_MODEL);
}

/* ===== Math kernels ===== */

/* LayerNorm: out = (x - mean) / sqrt(var + eps) * g + b */
static void layernorm(float* out, const float* x, const float* g, const float* b, int n) {
    float mean = 0.0f;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= n;
    float var = 0.0f;
    for (int i = 0; i < n; i++) { float d = x[i] - mean; var += d * d; }
    var /= n;
    float inv = 1.0f / sqrtf(var + 1e-5f);
    for (int i = 0; i < n; i++) out[i] = (x[i] - mean) * inv * g[i] + b[i];
}

/* y = W @ x + bias.   W is [out_dim x in_dim] row-major. */
static void matvec_bias(float* y, const float* W, const float* bias,
                        const float* x, int out_dim, int in_dim) {
    for (int i = 0; i < out_dim; i++) {
        const float* row = W + (size_t)i * in_dim;
        float s = bias ? bias[i] : 0.0f;
        for (int j = 0; j < in_dim; j++) s += row[j] * x[j];
        y[i] = s;
    }
}

static void softmax(float* x, int n) {
    float m = x[0];
    for (int i = 1; i < n; i++) if (x[i] > m) m = x[i];
    float s = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - m); s += x[i]; }
    float inv = 1.0f / s;
    for (int i = 0; i < n; i++) x[i] *= inv;
}

/* ===== Scratch buffers ===== */
static float* x_buf;          /* [MAX_SEQ_LEN x D_MODEL] */
static float* q_buf;          /* [MAX_SEQ_LEN x D_MODEL] */
static float* k_buf;          /* [MAX_SEQ_LEN x D_MODEL] */
static float* v_buf;          /* [MAX_SEQ_LEN x D_MODEL] */
static float* qkv_tmp;        /* [3*D_MODEL] */
static float* attn_out;       /* [D_MODEL] */
static float* proj_tmp;       /* [D_MODEL] */
static float* norm_tmp;       /* [D_MODEL] */
static float* ff_hidden;      /* [D_FF] */
static float* ff_out;         /* [D_MODEL] */
static float* logits;         /* [VOCAB_SIZE] */
static float* attn_scores;    /* [MAX_SEQ_LEN] */

static void alloc_scratch(void) {
    x_buf       = alloc_f((size_t)MAX_SEQ_LEN * D_MODEL);
    q_buf       = alloc_f((size_t)MAX_SEQ_LEN * D_MODEL);
    k_buf       = alloc_f((size_t)MAX_SEQ_LEN * D_MODEL);
    v_buf       = alloc_f((size_t)MAX_SEQ_LEN * D_MODEL);
    qkv_tmp     = alloc_f(3 * D_MODEL);
    attn_out    = alloc_f(D_MODEL);
    proj_tmp    = alloc_f(D_MODEL);
    norm_tmp    = alloc_f(D_MODEL);
    ff_hidden   = alloc_f(D_FF);
    ff_out      = alloc_f(D_MODEL);
    logits      = alloc_f(VOCAB_SIZE);
    attn_scores = alloc_f(MAX_SEQ_LEN);
}

/* ===== Forward pass =====
 *
 * Recomputes the full sequence each step (no KV cache). At ~9M params and
 * <=128 tokens this is cheap enough on a single CPU core for phase 1.
 */
static void forward(const int* tokens, int seq_len, float* out_logits) {
    /* 1. Embeddings (clamp id to [0, VOCAB_SIZE) defensively) */
    for (int t = 0; t < seq_len; t++) {
        int id = tokens[t];
        if (id < 0 || id >= VOCAB_SIZE) id = 0;
        const float* te = tok_emb + (size_t)id * D_MODEL;
        const float* pe = pos_emb + (size_t)t * D_MODEL;
        float* xt = x_buf + (size_t)t * D_MODEL;
        for (int d = 0; d < D_MODEL; d++) xt[d] = te[d] + pe[d];
    }

    float attn_scale = 1.0f / sqrtf((float)HEAD_DIM);

    /* 2. Transformer blocks */
    for (int l = 0; l < N_LAYERS; l++) {
        Block* b = &blocks[l];

        /* Project Q, K, V for every position */
        for (int t = 0; t < seq_len; t++) {
            float* xt = x_buf + (size_t)t * D_MODEL;
            layernorm(norm_tmp, xt, b->ln1_g, b->ln1_b, D_MODEL);
            matvec_bias(qkv_tmp, b->qkv_w, b->qkv_b, norm_tmp, 3 * D_MODEL, D_MODEL);
            memcpy(q_buf + (size_t)t * D_MODEL, qkv_tmp,                 D_MODEL * sizeof(float));
            memcpy(k_buf + (size_t)t * D_MODEL, qkv_tmp + D_MODEL,       D_MODEL * sizeof(float));
            memcpy(v_buf + (size_t)t * D_MODEL, qkv_tmp + 2 * D_MODEL,   D_MODEL * sizeof(float));
        }

        /* Causal multi-head attention */
        for (int t = 0; t < seq_len; t++) {
            memset(attn_out, 0, D_MODEL * sizeof(float));
            for (int h = 0; h < N_HEADS; h++) {
                const float* qh = q_buf + (size_t)t * D_MODEL + h * HEAD_DIM;
                for (int s = 0; s <= t; s++) {
                    const float* kh = k_buf + (size_t)s * D_MODEL + h * HEAD_DIM;
                    float dot = 0.0f;
                    for (int d = 0; d < HEAD_DIM; d++) dot += qh[d] * kh[d];
                    attn_scores[s] = dot * attn_scale;
                }
                softmax(attn_scores, t + 1);
                for (int s = 0; s <= t; s++) {
                    const float* vh = v_buf + (size_t)s * D_MODEL + h * HEAD_DIM;
                    float w = attn_scores[s];
                    float* oh = attn_out + h * HEAD_DIM;
                    for (int d = 0; d < HEAD_DIM; d++) oh[d] += w * vh[d];
                }
            }
            matvec_bias(proj_tmp, b->proj_w, b->proj_b, attn_out, D_MODEL, D_MODEL);
            float* xt = x_buf + (size_t)t * D_MODEL;
            for (int d = 0; d < D_MODEL; d++) xt[d] += proj_tmp[d];
        }

        /* FFN: ReLU(W1 x + b1), then W2 + b2 */
        for (int t = 0; t < seq_len; t++) {
            float* xt = x_buf + (size_t)t * D_MODEL;
            layernorm(norm_tmp, xt, b->ln2_g, b->ln2_b, D_MODEL);
            matvec_bias(ff_hidden, b->w1, b->b1, norm_tmp, D_FF, D_MODEL);
            for (int d = 0; d < D_FF; d++) if (ff_hidden[d] < 0) ff_hidden[d] = 0;
            matvec_bias(ff_out, b->w2, b->b2, ff_hidden, D_MODEL, D_FF);
            for (int d = 0; d < D_MODEL; d++) xt[d] += ff_out[d];
        }
    }

    /* 3. Final norm + tied LM head on the LAST position only */
    float* x_last = x_buf + (size_t)(seq_len - 1) * D_MODEL;
    layernorm(norm_tmp, x_last, ln_f_g, ln_f_b, D_MODEL);
    /* Weight-tied: logit_v = <norm_tmp, tok_emb[v]> */
    for (int v = 0; v < VOCAB_SIZE; v++) {
        const float* row = tok_emb + (size_t)v * D_MODEL;
        float s = 0.0f;
        for (int d = 0; d < D_MODEL; d++) s += norm_tmp[d] * row[d];
        out_logits[v] = s;
    }
}

/* ===== Tokenizer (ByteLevel BPE — matches HF tokenizers behavior) =====
 *
 * GuppyLM's tokenizer is a ByteLevel BPE (GPT-2 style):
 *   1. Three added/special tokens are matched literally before anything
 *      else: "<pad>" (id 0), "<|im_start|>" (id 1), "<|im_end|>" (id 2).
 *   2. The remaining bytes are passed through the byte_to_unicode table
 *      (engine/bytelevel.h) — e.g. ' ' -> 'Ġ', '\n' -> 'Ċ'.
 *   3. The encoded string is then segmented by greedy longest-match
 *      against the vocab. This is NOT byte-identical to the BPE merge
 *      algorithm, but for the small chat templates and short prompts
 *      we feed it, the longest-match almost always picks the merged
 *      token because longer merges shadow shorter ones.
 *
 * Fallback: if no vocab entry matches at the current encoded position
 * we advance by one UTF-8 codepoint and emit PAD_ID. This guarantees
 * forward progress and never produces an out-of-range token id.
 */

/* Length of the next UTF-8 codepoint at p (bytes). Defaults to 1 for
 * malformed input so we still make progress. */
static int utf8_skip(const char* p) {
    unsigned char c = (unsigned char)p[0];
    if ((c & 0x80) == 0x00) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}

static int safe_unk(void) {
    if (UNK_ID >= 0 && UNK_ID < VOCAB_SIZE) return UNK_ID;
    if (PAD_ID >= 0 && PAD_ID < VOCAB_SIZE) return PAD_ID;
    return 0;
}

/* Encode a byte-level segment [enc, enc+enc_len) into token ids using
 * greedy longest-match against the vocab. Appends to out[] starting at *n. */
static void bpe_longest_match(const char* enc, int enc_len,
                              int* out, int* n, int max_out) {
    int p = 0;
    while (p < enc_len && *n < max_out) {
        int best_len = 0, best_id = -1;
        for (int v = 0; v < VOCAB_SIZE; v++) {
            int vl = vocab[v].len;
            if (vl == 0 || vl <= best_len) continue;
            if (p + vl > enc_len) continue;
            if (memcmp(enc + p, vocab[v].bytes, vl) == 0) {
                best_len = vl;
                best_id  = v;
            }
        }
        if (best_len > 0) {
            out[(*n)++] = best_id;
            p += best_len;
        } else {
            out[(*n)++] = safe_unk();
            p += utf8_skip(enc + p);
        }
    }
}

/* Hardcoded order of added tokens. ids match config.json. */
typedef struct { const char* lit; int len; int id; } AddedTok;

static int find_added(const char* text, int pos, int len, AddedTok* hit) {
    static const AddedTok ADDED[] = {
        { "<|im_start|>", 12, 1 },
        { "<|im_end|>",   10, 2 },
        { "<pad>",         5, 0 },
    };
    for (size_t i = 0; i < sizeof(ADDED)/sizeof(ADDED[0]); i++) {
        if (pos + ADDED[i].len <= len &&
            memcmp(text + pos, ADDED[i].lit, ADDED[i].len) == 0) {
            *hit = ADDED[i];
            return 1;
        }
    }
    return 0;
}

/* Encode without auto-prepending BOS — caller controls the prompt template.
 * For GuppyLM the ChatML prefix already starts with <|im_start|> (= BOS). */
static int encode(const char* text, int* out, int max_out) {
    int n = 0;
    int len = (int)strlen(text);
    int pos = 0;

    /* Working buffer for byte-level encoded segments.
     * Worst case each input byte expands to ~2 UTF-8 bytes. */
    int    cap = len * 4 + 16;
    char*  enc = (char*)malloc(cap);
    if (!enc) { fprintf(stderr, "OOM encode\n"); exit(1); }

    while (pos < len && n < max_out) {
        AddedTok hit;
        if (find_added(text, pos, len, &hit)) {
            out[n++] = hit.id;
            pos += hit.len;
            continue;
        }

        /* Collect bytes until the next added token (or end of input),
         * applying byte-level encoding into enc[]. */
        int seg_start = pos;
        while (pos < len) {
            AddedTok dummy;
            if (find_added(text, pos, len, &dummy)) break;
            pos++;
        }

        int enc_len = 0;
        for (int i = seg_start; i < pos; i++) {
            unsigned char b = (unsigned char)text[i];
            int bl = bytelevel_len[b];
            memcpy(enc + enc_len, bytelevel[b], bl);
            enc_len += bl;
        }
        bpe_longest_match(enc, enc_len, out, &n, max_out);
    }

    free(enc);
    return n;
}

/* Decode a byte-level UTF-8 sequence into the original bytes.
 *
 * Inverse of the bytelevel encoding applied at tokenization time:
 *   - Single-byte ASCII (< 0x80) maps directly to its codepoint, which
 *     in the byte_to_unicode table corresponds to the same byte.
 *   - 2-byte UTF-8 sequences are codepoints in U+0080..U+0142; look up
 *     the original byte in bytelevel_inv[]. */
static void emit_token(int id) {
    if (id < 0 || id >= VOCAB_SIZE) return;
    /* Skip the three special tokens that have no on-screen representation. */
    if (id == BOS_ID || id == EOS_ID || id == PAD_ID) return;

    const unsigned char* p   = (const unsigned char*)vocab[id].bytes;
    const unsigned char* end = p + vocab[id].len;
    while (p < end) {
        int cp;
        int n;
        if (p[0] < 0x80) { cp = p[0]; n = 1; }
        else if ((p[0] & 0xE0) == 0xC0 && p + 1 < end) {
            cp = ((p[0] & 0x1F) << 6) | (p[1] & 0x3F); n = 2;
        }
        else if ((p[0] & 0xF0) == 0xE0 && p + 2 < end) {
            cp = ((p[0] & 0x0F) << 12) | ((p[1] & 0x3F) << 6) | (p[2] & 0x3F); n = 3;
        }
        else { cp = -1; n = 1; }

        int b = -1;
        if (cp >= 0 && cp < BYTELEVEL_INV_SIZE) b = bytelevel_inv[cp];
        if (b >= 0) fputc(b, stdout);
        p += n;
    }
    fflush(stdout);
}

/* ===== Sampling (temperature + top-k, matching reference inference.py) ===== */
static int cmp_desc(const void* a, const void* b) {
    float fa = *(const float*)a, fb = *(const float*)b;
    return (fa < fb) - (fa > fb);
}

static int sample(float* l, float temperature, int top_k) {
    if (temperature <= 0.0f) {
        int best = 0;
        for (int i = 1; i < VOCAB_SIZE; i++) if (l[i] > l[best]) best = i;
        return best;
    }
    for (int i = 0; i < VOCAB_SIZE; i++) l[i] /= temperature;

    /* top-k: mask everything below the k-th largest logit */
    if (top_k > 0 && top_k < VOCAB_SIZE) {
        float* sorted = (float*)malloc(VOCAB_SIZE * sizeof(float));
        memcpy(sorted, l, VOCAB_SIZE * sizeof(float));
        qsort(sorted, VOCAB_SIZE, sizeof(float), cmp_desc);
        float threshold = sorted[top_k - 1];
        free(sorted);
        for (int i = 0; i < VOCAB_SIZE; i++) {
            if (l[i] < threshold) l[i] = -INFINITY;
        }
    }

    softmax(l, VOCAB_SIZE);
    float r = (float)rand() / (float)RAND_MAX;
    float c = 0.0f;
    for (int i = 0; i < VOCAB_SIZE; i++) {
        c += l[i];
        if (c >= r) return i;
    }
    return VOCAB_SIZE - 1;
}

/* ===== Main ===== */
int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s \"prompt\"\n", argv[0]);
        return 1;
    }
    srand((unsigned)time(NULL));

    load_weights();
    alloc_scratch();

    /* GuppyLM was trained with the ChatML template:
     *   <|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n
     * Without this wrapper the model emits gibberish. The literal special
     * tokens are present in the vocab (ids 1 and 2), so the longest-match
     * encoder will pick them up cleanly. */
    char prompt[4096];
    snprintf(prompt, sizeof(prompt),
             "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n",
             argv[1]);

    int* tokens = (int*)malloc(MAX_SEQ_LEN * sizeof(int));
    int seq_len = encode(prompt, tokens, MAX_SEQ_LEN);

    float temperature = 0.7f;   /* matches reference inference.py defaults */
    int   top_k       = 50;
    int   max_new     = 64;

    for (int i = 0; i < max_new && seq_len < MAX_SEQ_LEN; i++) {
        forward(tokens, seq_len, logits);
        int next = sample(logits, temperature, top_k);
        if (next == EOS_ID) break;
        emit_token(next);
        tokens[seq_len++] = next;
    }
    fputc('\n', stdout);
    return 0;
}
