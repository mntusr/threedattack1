# Adversarial 3D Objects Against Monocular Depth Estimators (official implementation) (to be published)

## Usage

Create the environments for MiDaS and the main program. The MiDaS environment is described in the [ZoeDepth repository](https://github.com/isl-org/ZoeDepth). The main environment is described in the local [environment.yml](environment.yml) file.

Download the scene files from the releases (`scene_files.zip`) and extract it to the root of this repository.

Start the worker (`launch_dpt_beit_384_worker.py` or `launch_midas_large_worker.py`) for the target model in the **MiDaS-environment**.

Create a file, called `local_config.json` based on `local_config_default.json` in the same directory. Modify the `mlflow_tracking_url` key to specify the Mlflow tracking URL. The other two keys are not important for prediction.

Start Mlflow in the **main environment** and create a new experiment, called, `Experiment1`.

Start `manual_attack.py` in the **main environment** with the attack parameters. You can see the command for the attacks described in the paper below.

Command for hyperparameter group A in the paper, with MiDaS v2.1 Large<sub>384</sub> target model:

```
python manual_attack.py --n-control-points "108" --n-estim-viewpts "2" --freeze-estim 0 1 --max-pos-change-per-coord "0.3496993572440835" --cma-optimized-metric "median_reldelta_cropped_log10" --sigma0 "0.0457404277815225" --n-val-viewpoints "200" --n-train-viewpoints "400" --n-test-viewpoints "200" --target-model-name "midas_large" --target-scene-path "scenes\room1_subdivided3.glb" --free-area-multiplier "1.5332186838927129" --experiment-name "Experiment1" --maxiter "133" --transform-type "volume_based" --n-cubes-steps "20" --max-shape-change "0.2093192993924143" --eval-on-test
```

Command for hyperparameter group B in the paper, with MiDaS v2.1 Large<sub>384</sub> target model:

```
python manual_attack.py --n-control-points "191" --n-estim-viewpts "3" --freeze-estim 0 1 2 --max-pos-change-per-coord "0.3665534825115781" --cma-optimized-metric "min_reldelta_log10" --sigma0 "0.2039613547547968" --n-val-viewpoints "200" --n-train-viewpoints "400" --n-test-viewpoints "200" --target-model-name "midas_large" --target-scene-path "scenes\room1_subdivided3.glb" --free-area-multiplier "1.2225047450966795" --experiment-name "Experiment1" --maxiter "118" --transform-type "volume_based" --n-cubes-steps "20" --max-shape-change "0.1841823650223717" --eval-on-test
```

Command for hyperparameter group A in the paper, with MiDaS v3.1 BEiT<sub>L-384</sub>:

```
python manual_attack.py --n-control-points "108" --n-estim-viewpts "2" --freeze-estim 0 1 --max-pos-change-per-coord "0.3496993572440835" --cma-optimized-metric "median_reldelta_cropped_log10" --sigma0 "0.0457404277815225" --n-val-viewpoints "200" --n-train-viewpoints "400" --n-test-viewpoints "200" --target-model-name "dpt_beit_384" --target-scene-path "scenes\room1_subdivided3.glb" --free-area-multiplier "1.5332186838927129" --experiment-name "Experiment1" --maxiter "133" --transform-type "volume_based" --n-cubes-steps "20" --max-shape-change "0.2093192993924143" --eval-on-test
```

Command for hyperparameter group B in the paper, with MiDaS v3.1 BEiT<sub>L-384</sub>:

```
python manual_attack.py --n-control-points "191" --n-estim-viewpts "3" --freeze-estim 0 1 2 --max-pos-change-per-coord "0.3665534825115781" --cma-optimized-metric "min_reldelta_log10" --sigma0 "0.2039613547547968" --n-val-viewpoints "200" --n-train-viewpoints "400" --n-test-viewpoints "200" --target-model-name "dpt_beit_384" --target-scene-path "scenes\room1_subdivided3.glb" --free-area-multiplier "1.2225047450966795" --experiment-name "Experiment1" --maxiter "118" --transform-type "volume_based" --n-cubes-steps "20" --max-shape-change "0.1841823650223717" --eval-on-test
```

## Documentation

The documentation is contained by the `docs` folder. Topics:

* [Authoring scenes](docs/scenes.md)
* [Array formats](docs/array_formats.md)
* [Testing](docs/testing.md)
* [Hyperparameter tuning](docs/hyperparameter_tuning.md)
* [Known typos](docs/known_typos.md)\*

\*: We are aware of these typos, but we decided to keep them, for the sake of consistency with the code used for the paper.

# How to cite

Authors:

* Tamás Márk Fehér
* Márton Szemenyei

BibTex: Coming soon.
