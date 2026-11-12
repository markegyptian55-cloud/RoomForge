# Week 2 Interactive Segmentation & 5-Tier Tracking Report (`s05_track.py`)

## 1. Warm Interactive SAM 2.1 Architecture

Week 2 interactive clicks (`segment_init`, `segment_refine`) cannot afford cold model weight initialization per click. `worker/room3d_worker/pipeline/s05_track.py` implements `WarmSamSession`:

- **Bounded LRU Frame Cache**: Calls `_to_cpu_if_tensor` on cache insertion so up to `max_cache_items` (default `8`) encoded image states sit in CPU RAM inside an `OrderedDict` (evicting the oldest item via `popitem(last=False)`), while active prediction executes on `_resolve_sam_device("cuda:1")` to stay within `Tesla T4` `15360 MiB` VRAM limits.
- **State Snapshot & Restoration**: `_Sam2Interactive` captures and restores `_features`, `_orig_hw`, and `_is_image_set` when switching between cached reference frames (`set_image` runs only on cache miss).
- **Binary Mask Normalization**: `normalize_binary_mask` converts boolean arrays, raw float logits (`threshold=0.0`), `0..1` probabilities, or `0..255` arrays into canonical 1-channel `uint8` (`{0, 255}`) masks matching `(height, width)`, saved via `save_binary_mask_png`.

---

## 2. 5-Tier Propagation & Recovery Pipeline (`run_tracking_pipeline`)

To prevent object drift, ID switches on occlusion, and dropped masks across wide camera sweeps, `run_tracking_pipeline` orchestrates five recovery tiers:

| Tier | Helper / Function | Condition / Threshold | Action Taken |
| :--- | :--- | :--- | :--- |
| **Tier 1** | `track_vos` (`_Sam2Video`) | Default pass on ordered frames | Stages ordered RGB JPEGs inside a `tempfile.TemporaryDirectory`, runs `propagate_in_video` with `inference_state_device="cpu"`. |
| **Tier 2** | `filter_neighbor_iou` & `_neighbor_drift_bad` | `min_iou=0.25`, `max_area_ratio=3.5` | Flags sudden neighbor mask drops or runaway area expansion relative to valid neighbors. |
| **Tier 3** | `_lift_foreground_points` & `lift_and_reproject_masks` | All active frames (`occlusion_tol=0.25`) | Lifts reference frame (`masks[ref_frame_idx]`) foreground pixels (`z_cam > 1e-4`), classifies per-frame visibility (`visible`, `occluded`, `out_of_frustum`) via `_classify_frame_visibility`, and re-prompts `WarmSamSession` when occluded objects re-enter view. |
| **Tier 4** | `score_reid_crops` (`_HistogramReidEncoder`) | Reference similarity `< min_similarity` (`0.65`) | Evaluates masked HSV histogram similarity (`_crop_masked_rgb`) against anchor appearance and rejects identity switches. |
| **Tier 5** | `apply_manual_mask_overrides` (+ standalone `project_3dgs_mask_hints`) | Unrecovered flagged frames / user overrides in UI | Zeroes unrecovered (`iou_flagged - recovered`) and `reid_rejected` masks, applies per-frame `ManualOverride(frame_idx, mask_path, skip_frame)` from `PUT /api/edits/{id}/masks/{frame_idx}`, and ships `project_3dgs_mask_hints` as a standalone 3DGS-to-2D projection helper (not yet auto-invoked inside `run_tracking_pipeline`). |

---

## 3. Verified Local Behavior (`worker/tests/test_track.py`)

All five tracking invariants are automated and passing on CPU (`5 passed` in `worker/tests/test_track.py`):
1. Warm SAM session cache hits bypass redundant encoder passes and evict excess states cleanly.
2. Staged video propagation preserves frame ordering in isolated temporary directories.
3. Multi-point frustum/depth lifting separates temporary occlusion from scene exit and re-prompts cleanly on re-entry.
4. Masked HSV Re-ID rejects drifted identities cleanly after simulated object swap.
5. Manual overrides (`skip_frame` + custom PNG masks) take precedence in `run_tracking_pipeline` and write a JSON tracking report via `write_tracking_report`.
