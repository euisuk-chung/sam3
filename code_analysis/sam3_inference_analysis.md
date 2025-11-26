# SAM 3 Inference Analysis

This document provides a detailed analysis of the inference process in SAM 3, focusing on how the model handles both image and video inputs. It covers the high-level architecture, key classes, and the step-by-step flow of data during prediction.

## 1. High-Level Architecture & Class Hierarchy

The SAM 3 inference system is built upon a hierarchical class structure that separates high-level session management from low-level model execution.

### Key Classes

*   **`Sam3VideoPredictor`** (`sam3/model/sam3_video_predictor.py`):
    *   **Role**: The main entry point for users. It acts as a high-level wrapper and session manager.
    *   **Functionality**: It manages multiple inference sessions (via `_ALL_INFERENCE_STATES`), handles multi-GPU coordination, and dispatches user requests (like adding prompts or resetting sessions) to the underlying model.
    *   **Relationship**: It *contains* the model instance (stored in `self.model`), which is created via `build_sam3_video_model`.

*   **`Sam3VideoInferenceWithInstanceInteractivity`** (`sam3/model/sam3_video_inference.py`):
    *   **Role**: The actual model class instantiated by the builder.
    *   **Functionality**: Extends the base inference logic to support interactive refinement (e.g., adding clicks to refine a track). It manages the "action history" (add, remove, refine) to optimize propagation (e.g., only re-tracking affected objects).
    *   **Inheritance**: Inherits from `Sam3VideoInference`.

*   **`Sam3VideoInference`** (`sam3/model/sam3_video_inference.py`):
    *   **Role**: Implements the core video inference logic.
    *   **Functionality**: Handles the main propagation loop (`propagate_in_video`), single-frame inference (`_run_single_frame_inference`), and memory management. It orchestrates the interaction between the **Detector** (SAM 3 Image Model) and the **Tracker**.
    *   **Inheritance**: Inherits from `Sam3VideoBase`.

*   **`Sam3TrackerPredictor`** (`sam3/model/sam3_tracking_predictor.py`):
    *   **Role**: The specific component responsible for tracking objects across frames.
    *   **Functionality**: Handles memory encoding, memory fusion (reading from past frames), and mask prediction based on temporal context.

## 2. Inference Workflow

The inference process can be broken down into three main stages: **Initialization**, **Prompting**, and **Propagation**.

### Stage 1: Initialization (`init_state`)

Before any inference can occur, the system must load the media and prepare the state.

1.  **Load Media**: The `init_state` method (in `Sam3VideoInference`) loads the video or image. It supports loading frames on demand or pre-loading them.
2.  **Create State Dictionary**: A dictionary `inference_state` is created to hold all session-specific data:
    *   `images`: The raw video frames.
    *   `input_batch`: A `BatchedDatapoint` containing preprocessed images and placeholders for prompts.
    *   `feature_cache`: Caches for vision backbone features to avoid re-computation.
    *   `tracker_inference_states`: States specific to the tracker (e.g., object memories).
    *   `action_history`: A log of user interactions (used to optimize re-propagation).

### Stage 2: Prompting (`add_prompt`)

SAM 3 supports multimodal prompts: text, boxes, and points.

*   **Text & Box Prompts**:
    *   Handled by `Sam3VideoInference.add_prompt`.
    *   **Process**:
        1.  The prompt is added to the `inference_state`.
        2.  `_run_single_frame_inference` is called immediately on the current frame to generate initial masks.
        3.  For text prompts, the text encoder generates embeddings which are used to query the image features.
        4.  For box prompts, they are converted to geometric embeddings.

*   **Point Prompts (Interactive Refinement)**:
    *   Handled by `Sam3VideoInferenceWithInstanceInteractivity.add_tracker_new_points`.
    *   **Process**:
        1.  Identifies if this is a new object or a refinement of an existing one.
        2.  If it's a new object, a new tracker state is initialized.
        3.  The point is passed to the **Tracker** (`self.tracker.add_new_points`).
        4.  The tracker updates its memory and refines the mask for that specific frame.

### Stage 3: Propagation (`propagate_in_video`)

This is the core loop where the model tracks objects across the video.

1.  **Action Analysis**:
    *   `Sam3VideoInferenceWithInstanceInteractivity` first checks the `action_history`.
    *   It determines if it needs to run a **Full Propagation** (re-track everything), a **Partial Propagation** (re-track only specific objects that were refined), or just **Fetch** existing results (if nothing changed).

2.  **The Propagation Loop**:
    *   The method iterates through the video frames (forward or backward).
    *   For each frame, it calls `_run_single_frame_inference`.

3.  **Single Frame Inference (`_run_single_frame_inference`)**:
    *   This method orchestrates the detection and tracking for a specific frame.
    *   **Detection**: Runs the SAM 3 Image Model (`self.detector`) to find objects matching the text/box prompts.
    *   **Tracking**: Runs the Tracker (`self.tracker`) to propagate masks from previous frames using memory attention.
    *   **Fusion**: It combines results from the detector (newly found objects) and the tracker (objects being followed).
    *   **Memory Update**: The output mask of the current frame is encoded and added to the tracker's memory bank to help track the object in subsequent frames.

## 3. Key Functions & Data Flow

### `_run_single_frame_inference` (in `Sam3VideoInference`)
This is the workhorse function.
```python
def _run_single_frame_inference(self, inference_state, frame_idx, reverse):
    # 1. Prepare inputs (images, prompts)
    # 2. Call _det_track_one_frame to get raw predictions
    obj_id_to_mask, obj_id_to_score, ... = self._det_track_one_frame(...)
    
    # 3. Cache the results
    self._cache_frame_outputs(...)
    
    # 4. Return formatted output
    return out
```

### `_det_track_one_frame`
This function (likely in `Sam3VideoBase` or inherited) manages the dual nature of SAM 3:
1.  **Detector Branch**: "Is there a 'cat' in this frame?" (using Text/Vision features).
2.  **Tracker Branch**: "Where did the 'cat' from frame $t-1$ go?" (using Memory features).
3.  **Conflict Resolution**: If the detector finds a new 'cat' and the tracker is following an existing 'cat', it resolves overlaps and assigns IDs.

### `propagate_in_video` (in `Sam3VideoInferenceWithInstanceInteractivity`)
Optimizes the process based on user actions.
```python
def propagate_in_video(self, inference_state, ...):
    # 1. Decide strategy based on history
    propagation_type, obj_ids = self.parse_action_history_for_propagation(...)
    
    # 2. Execute Strategy
    if propagation_type == "propagation_full":
        yield from super().propagate_in_video(...) # Run full loop
    elif propagation_type == "propagation_partial":
        # Only run tracker for specific objects (faster)
        ...
    elif propagation_type == "propagation_fetch":
        # Just return cached results (instant)
        ...
```

## 4. Summary of Data Structures

*   **`inference_state`**: The "global" context for a session.
*   **`BatchedDatapoint`**: The standard input format for the model, handling batching of images and prompts.
*   **`tracker_inference_states`**: A list of states, where each state corresponds to a tracked object (or a group of objects), containing their specific memory banks.
