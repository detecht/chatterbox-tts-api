# Plan: Enable Native bfloat16 Support for NVIDIA L4 GPU

This plan outlines the steps required to transition the Chatterbox TTS API and the underlying library to use native `bfloat16` (BF16) precision. This will significantly improve performance and reduce VRAM usage on compatible hardware like the NVIDIA L4.

---

## 1. Chatterbox TTS API Changes (Current Repo)

### A. Configuration Updates
*   **File:** `app/config.py`
*   **Action:** Add a `MODEL_DTYPE` environment variable.
*   **Details:** Allow values `float32` (default) and `bfloat16`.

### B. Model Initialization
*   **File:** `app/core/tts_model.py`
*   **Action:** Explicitly cast model components to the configured dtype after loading.
*   **Implementation:**
    ```python
    if _device == 'cuda' and Config.MODEL_DTYPE == 'bfloat16':
        _model.t3.to(dtype=torch.bfloat16)
        _model.s3gen.to(dtype=torch.bfloat16)
        _model.ve.to(dtype=torch.bfloat16)
    ```

### C. Inference Optimization
*   **File:** `app/api/endpoints/speech.py`
*   **Action:** Wrap the `model.generate` calls in a `torch.cuda.amp.autocast` context.
*   **Implementation:**
    ```python
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
        audio_tensor = await loop.run_in_executor(...)
    ```

---

## 2. Chatterbox-TTS Library Changes (`@chatterbox/**`)

The library contains several hardcoded `float32` casts that will cause errors or unnecessary overhead when running in BF16. These must be updated to be "precision-aware."

### A. S3Gen Module
*   **File:** `src/chatterbox/models/s3gen/s3gen.py`
*   **Change:** Update `embed_ref` to use `self.dtype` instead of `.float()` or hardcoded `torch.float32`.
*   **Change:** Remove the `FIXME (fp16 mode)` comment and the associated cast in `inference`, ensuring the output mels match the expected precision.

### B. Mel Spectrogram Utilities
*   **File:** `src/chatterbox/models/s3gen/utils/mel.py`
*   **Change:** Update `mel_spectrogram` function.
*   **Details:** 
    *   Instead of `torch.tensor(y).float()`, use `.to(dtype=model_dtype)`.
    *   Ensure `mel_basis` and `hann_window` are created or cast to match the model's dtype.

### C. Speaker Encoder (CAMPPlus)
*   **File:** `src/chatterbox/models/s3gen/xvector.py`
*   **Change:** In `CAMPPlus.inference`, change `speech.to(torch.float32)` to use the model's current weight precision.

### D. Vocoder (HiFT-GAN)
*   **File:** `src/chatterbox/models/s3gen/hifigan.py`
*   **Change:** In `SineGen._f02uv`, change `.type(torch.float32)` to match the F0 tensor's dtype or allow it to be BF16.
*   **File:** `src/chatterbox/models/s3gen/decoder.py`
*   **Change:** Ensure `mask_to_bias` continues to support `bfloat16` (currently has an `assert` for it).

---

## 3. Verification Steps

1.  **Unit Tests:** Run existing tests in `tests/` to ensure audio quality hasn't degraded.
2.  **Benchmark:** Compare generation speed (RTF) on the L4 GPU between FP32 and BF16.
3.  **VRAM Audit:** Confirm that VRAM usage is reduced (expected ~40-50% reduction for model weights).
4.  **Precision Check:** Add a debug endpoint or log statement to verify `model.t3.text_head.weight.dtype` is `torch.bfloat16` at runtime.
