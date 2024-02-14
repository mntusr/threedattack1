The hyperparameter tuning is implemented by `optuna_attack.py`.

The hyperparameter tuning relies on [Optuna](https://github.com/optuna/optuna). Its package is included in the main environmnet, just like [Optuna dashboard](https://github.com/optuna/optuna-dashboard).

The hyperparameter optimization script supports both NSGA II and TPE samplers.

# Usage

Launch the worker for the target model in the **MiDaS-environment** and configure Mlflow in `local_config.json`, as you would do for the manual attack.

Start Mlflow in the **main environment**.

Create experiment `Experiment2` in Mlflow. You do not have to create the Optuna study, the hyperparameter optimization script will take care about it.

Start `optuna_attack.py` **in the main environment**. We used this command, or similar ones for hyperparameter tuning:

```
python optuna_attack.py --experiment "Experiment2" --study "Study2" --study-timeout 86400 --target-model "midas_large" --max-maxiter 200  --max-n-control-points 200 --max-n-estim-viewpts 4 --meta-optimized exact_mean_reldelta_val_cropped_rmse --sampler tpe --no-predcall-target
```

You can later look up the Mlflow run for each Optuna trial using the user attribute `run_name`.

# Using an existing study

The hyperparameter optimization script supports using an existing study, but this behavior is not enabled by default. You should use, `--resume-with-retry` or `--resume-without-retry` to enable it. The first flag resumes the study, without retrying the last trial if that failed. The second one resumes the study with retrying the trial.

> [!WARNING]
> The hyperparameter optimization script does not serialize the state of the samplers, so you should keep in mind the related caveats described in the [Optuna documentation](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/001_rdb.html#resume-study).