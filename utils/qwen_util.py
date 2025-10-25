def patch_qwen_vl_utils(vision_process, cur_val = None):
    if hasattr(vision_process, '_patch'):
        return
    if os.getenv('VIDEO_MAX_PIXELS') and not os.getenv('VIDEO_TOTAL_PIXELS'):
        # https://github.com/QwenLM/Qwen2.5-VL/issues/1120
        os.environ['VIDEO_TOTAL_PIXELS'] = str(int(128000 * 28 * 28 * 0.9))
    res = {}
    for key in [
            'image_factor',
            'min_pixels',
            'max_pixels',
            'max_ratio',
            'video_min_pixels',
            'video_max_pixels',
            'video_total_pixels',
            'frame_factor',
            'fps',
            'fps_min_frames',
            'fps_max_frames',
    ]:
        type_func = float if key == 'fps' else int
        if cur_val is not None:
            val = cur_val[key]
        else:
            val = getattr(vision_process, key.upper(), None)
        if val is None:
            continue
        default_value = getattr(vision_process, key.upper(), None)
        if default_value is None:
            # Skip keys not supported by the specific vision_process implementation
            continue
        setattr(vision_process, key.upper(), val)
        res[key] = val
    # Patch decord video reader if available
    _read_video_decord = getattr(vision_process, '_read_video_decord', None)
    if _read_video_decord is not None:

        def _new_read_video_decord(ele: dict):
            from .vision_utils import load_file
            ele['video'] = load_file(ele['video'])
            return _read_video_decord(ele)

        backends = getattr(vision_process, 'VIDEO_READER_BACKENDS', None)
        if isinstance(backends, dict):
            backends['decord'] = _new_read_video_decord
        elif backends is None:  # keye_vl
            vision_process._read_video_decord = _new_read_video_decord
    vision_process._patch = True
    return res
