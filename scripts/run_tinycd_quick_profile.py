import os
import sys
from pathlib import Path

# Ensure geoai package is in import path
root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir))

# Enable coordinate subsampling to make the training extremely fast (Quick Profile)
os.environ["DL_MAX_SAMPLES"] = "100"

from geoai.experiments.experiment_runner import ExperimentRunner

def main():
    config_path = root_dir / "configs" / "experiments" / "tinycd_quick_profile.yaml"
    runner = ExperimentRunner(outputs_dir=str(root_dir / "outputs" / "experiments"), configs_dir=str(root_dir / "configs"))
    
    print("Starting TinyCD Quick Profile run...")
    try:
        experiment = runner.run(config_path)
        print("Experiment State:", experiment.get_state())
        print("Metrics:", experiment.metrics)
        if experiment.get_state() == "Completed":
            print("Success! TinyCD pipeline ran successfully.")
            sys.exit(0)
        else:
            print("Failure: TinyCD pipeline execution failed.")
            sys.exit(1)
    except Exception as e:
        print("Error during Quick Profile execution:", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
