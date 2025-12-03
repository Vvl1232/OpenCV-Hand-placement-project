# TODO for Improving Hand Detection Accuracy and Sizes

- [x] Increase window/screen size: Update FRAME_WIDTH and FRAME_HEIGHT to 1280x720 for better resolution.
- [x] Increase rectangle size: Update RECT_WIDTH and RECT_HEIGHT to 400x300 to enlarge the virtual boundary.
- [x] Refine skin color ranges: Adjust LOWER_HSV, UPPER_HSV, LOWER_YCrCb, UPPER_YCrCb for more accurate skin detection and reduce false positives.
- [x] Expand face removal: Dilate the face mask to cover neck and ears areas, preventing misidentification.
- [x] Enhance hand detection filters: Tighten area threshold, aspect ratio, extent, and solidity in detect_hand function for better accuracy.
- [ ] Test real-time performance: Run the code and ensure FPS >=8, adjust parameters if needed.
