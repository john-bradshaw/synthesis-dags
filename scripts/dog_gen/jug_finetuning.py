"""
Script to use jug to run through the different tasks in parallel.
"""

from jug import TaskGenerator

from syn_dags.script_utils import opt_utils
import run_finetuning

@TaskGenerator
def run_ft(task):
    params = run_finetuning.Params(task)
    res = run_finetuning.main(params)
    return res


task_list = [
    opt_utils.GuacTask.RANOLAZINE,
    opt_utils.GuacTask.VALSARTAN,
    opt_utils.GuacTask.DECO,
]

out = [run_ft(task) for task in task_list]
