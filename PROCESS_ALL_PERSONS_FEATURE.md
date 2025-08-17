# Process All Persons Feature

## Overview
Added a new `process_all_persons` parameter that allows you to detect and process **all people** in multi-person scenes instead of selecting just one person.

## Usage

### Option 1: Single Person Mode (Default)
- **`process_all_persons`**: `False` (default)
- **`person_index`**: `"0"` (or any number to select specific person)
- **Result**: Processes only the selected person with full feature support

### Option 2: All Persons Mode (NEW)
- **`process_all_persons`**: `True` 
- **`person_index`**: Any value (ignored when process_all_persons=True)
- **Result**: Detects and includes ALL people found in the image

## How It Works

When `process_all_persons=True`:

1. **Detection**: DWPose detects all people in the image
2. **No Person Selection**: The `person_index` parameter is ignored
3. **No Bone Validation**: Bone validation is skipped for performance (all persons)
4. **No Kalman Filtering**: Temporal smoothing is disabled (too complex for multiple people)
5. **Full Output**: All detected people are included in the JSON keypoints output

## Example Scenarios

### Family Photo (4 people)
```
process_all_persons: True
→ Result: All 4 people detected and included in output
→ JSON contains: people[0], people[1], people[2], people[3]
```

### Sports Scene (Multiple Athletes)
```
process_all_persons: True  
→ Result: All athletes detected and processed
→ Perfect for crowd analysis or team pose detection
```

### Portrait (1 person)
```
process_all_persons: False
person_index: "0"
→ Result: Single person with full Kalman filtering and bone validation
→ Best quality for single-person use cases
```

## Feature Limitations

When `process_all_persons=True`:

- ❌ **No Kalman Filtering**: Temporal smoothing is only available for single-person mode
- ❌ **No Bone Validation**: Hand bone validation is skipped for performance
- ✅ **Faster Processing**: No per-person post-processing overhead
- ✅ **Complete Detection**: All people in the scene are captured

## Technical Implementation

### Files Modified
- **`nodes.py`**: Added `process_all_persons` parameter to INPUT_TYPES and run() method
- **`loader.py`**: Modified `_run_dwpose_with_thresholds()` to force original DWPose method for all persons

### Code Changes
```python
# New parameter in nodes.py INPUT_TYPES
"process_all_persons": ("BOOLEAN", {"default": False}),

# Modified person selection logic  
if process_all_persons:
    person_index = None  # Special value for all persons mode
else:
    person_index = int(person_index)  # Normal single person mode

# Updated loader.py to force custom filtering for all persons
if person_index is None:
    # Process all persons mode - use very permissive thresholds to keep all people
    body_threshold = 0.01  # Very low threshold to keep all poses
    hands_face_threshold = 0.01
    # Disable bone validation (designed for single person)
    # Use draw_poses(filtered_poses, ...) which draws ALL people
```

### Key Fix Applied
**Issue**: When `process_all_persons=True`, the pose image was only showing one person (flickering between people) despite the JSON containing all persons.

**Root Cause**: The original DWPose method has a limitation where it only draws one person at a time, even when multiple people are detected.

**Solution**: Modified the **custom filtering path** to handle all persons mode:
1. **Force custom filtering path** for `process_all_persons=True`
2. **Use very permissive thresholds** (0.01) to keep ALL detected poses
3. **Disable bone validation** in all persons mode (designed for single person)
4. **Draw all filtered poses** using `draw_poses(filtered_poses, ...)` which properly renders ALL people

This ensures that both the JSON contains all people AND the pose image shows all people simultaneously.

## Migration Guide

### Existing Workflows
- **No changes needed**: All existing workflows continue to work unchanged
- **Default behavior**: `process_all_persons=False` maintains original single-person behavior

### New Multi-Person Workflows
1. Set `process_all_persons` to `True`
2. Set `person_index` to any value (it will be ignored)
3. Disable Kalman filtering for best performance
4. Parse the returned JSON to access all detected people

## Performance Considerations

### Single Person Mode (Original)
- **Memory**: Moderate (processes one person)
- **Speed**: Slower (bone validation + Kalman filtering)
- **Quality**: Highest (all post-processing enabled)

### All Persons Mode (New)
- **Memory**: Higher (processes all detected people)
- **Speed**: Faster per person (no post-processing)
- **Quality**: Good (raw DWPose output without refinements)

## Use Cases

### Perfect for All Persons Mode:
- 👥 Group photos and family pictures
- 🏃 Sports and action scenes with multiple people
- 🎭 Dance performances and choreography
- 📊 Crowd analysis and people counting
- 🎬 Multi-actor scenes in video processing

### Better with Single Person Mode:
- 📸 Portrait photography
- 🎯 Precise pose analysis requiring high accuracy
- 🎥 Video sequences needing temporal smoothing
- ✋ Hand gesture analysis (bone validation needed)
- 🔬 Research applications requiring maximum precision

The new feature provides the flexibility to choose between **maximum quality** (single person) and **complete coverage** (all persons) based on your specific use case!