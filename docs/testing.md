This documentation describes how to test the code.

# Main steps

Run the automated tests. The tests are written using `unittest`.

Try all features exposed by the live preview (`preview_scene.py`).

Run the following manual tests:

* Look for peter panning and shadow acne in all scenes, using `preview_scene.py`
* Run `manual_test_shadows.py` and check for:
  * The presence of the shadows.
  * The absence of peter panning and shadow acne.
  * The correct application of shadow blur.
  * The originally white spot on the edge of the the scene should be blue-ish.
  * The tonemapping is correctly applied (i. e. matches to the tonemapping provided by the live preview).
* Run `manual_test_depth_predictor.py` for all depth predictors and check whether:
  * They pass all quantitative performance tests.
  * The shown example predictions of the models look correct.
* Run `manual_test_temporary_target_transforms.py` and check whether:
  * The target oject is visible and transformed on the image in both transformation modes. The two transformations do not have to yield the same results due to the differences in the algorithm.
* Run `manual_test_shadows.py` and check whether:
  * The viewpoint handling seems working.
  * The point cloud creation seems working.
  * The near plane is correctly configured.
  * The visible shadows look correct.
* Run `manual_test_masking.py` and check wheter:
  * Only a rectangular area around the scaled standing area of the target object is visible.