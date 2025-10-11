# Authors: Jing He, Haodong Li
# Last modified: 2025-10-10

import os


def init_per_sample_csv(output_dir, alignment, metric_funcs):
    per_sample_csv = os.path.join(output_dir, f"per_sample_{alignment}.csv")
    with open(per_sample_csv, "w+") as f:
        f.write("filename,")
        f.write(",".join([m.__name__ for m in metric_funcs]))
        f.write("\n")
    return per_sample_csv

def write_per_sample_csv(per_sample_csv, pred_name, sample_metric):
    with open(per_sample_csv, "a+") as f:
        f.write(pred_name + ",")
        f.write(",".join(sample_metric))
        f.write("\n")

def write_metrics_txt(eval_dir, alignment, eval_text):
    metrics_filename = f"eval_metrics_{alignment}.txt"
    _save_to = os.path.join(eval_dir, metrics_filename)
    with open(_save_to, "w+") as f:
        f.write(eval_text)
