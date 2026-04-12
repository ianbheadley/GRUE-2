import json
import os
from datetime import datetime


class ExperimentLog:
    """Structured logging for concept extraction experiments."""

    def __init__(self, experiment_name, output_dir="results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.log = {
            "experiment": experiment_name,
            "timestamp": datetime.now().isoformat(),
            "config": {},
            "results": {},
            "controls": {},
            "summary": {}
        }

    def set_config(self, **kwargs):
        self.log["config"].update(kwargs)

    def log_result(self, method, key, value):
        if method not in self.log["results"]:
            self.log["results"][method] = {}
        self.log["results"][method][key] = value

    def log_control(self, method, key, value):
        if method not in self.log["controls"]:
            self.log["controls"][method] = {}
        self.log["controls"][method][key] = value

    def log_summary(self, key, value):
        self.log["summary"][key] = value

    def save(self):
        ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        name = self.log["experiment"].replace(" ", "_")
        filepath = os.path.join(self.output_dir, f"{name}_{ts}.json")
        with open(filepath, "w") as f:
            json.dump(self.log, f, indent=2, default=str)
        print(f"Experiment log saved to {filepath}")
        return filepath

    def print_summary(self):
        print("\n" + "=" * 70)
        print(f"EXPERIMENT: {self.log['experiment']}")
        print(f"TIMESTAMP: {self.log['timestamp']}")
        print("=" * 70)

        if self.log["config"]:
            print("\nCONFIG:")
            for k, v in self.log["config"].items():
                print(f"  {k}: {v}")

        for method, results in self.log["results"].items():
            print(f"\n--- {method} ---")
            for k, v in results.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")

            if method in self.log["controls"]:
                print(f"  Controls:")
                for k, v in self.log["controls"][method].items():
                    if isinstance(v, float):
                        print(f"    {k}: {v:.4f}")
                    else:
                        print(f"    {k}: {v}")

        if self.log["summary"]:
            print(f"\nSUMMARY:")
            for k, v in self.log["summary"].items():
                print(f"  {k}: {v}")

        print("=" * 70)
