"""Tidy3D cloud access. Nothing here starts a solver run without an explicit confirmation.

estimate():  upload + web.estimate_cost(), printed; the draft task is deleted unless kept.
run():       upload, print the estimate, then ask for the task name to be typed back before
             web.start(); returns the loaded SimulationData.
"""

from __future__ import annotations


def estimate(sim, task_name: str, folder: str, keep_task: bool = False) -> tuple[str | None, float]:
    from tidy3d import web

    task_id = web.upload(sim, task_name=task_name, folder_name=folder, verbose=False)
    cost = web.estimate_cost(task_id, verbose=False)
    print(f"[estimate] {task_name}: {cost:.4f} FlexCredit (max, before shutoff)")
    if not keep_task:
        web.delete(task_id)
        task_id = None
    return task_id, cost


def run(sim, task_name: str, folder: str, path: str, confirm=input):
    """Estimate, print, ask, then run. `confirm` must return the task name typed back."""
    from tidy3d import web

    task_id, cost = estimate(sim, task_name, folder, keep_task=True)
    answer = confirm(f"Start {task_name} for up to {cost:.4f} FlexCredit? Type the task name to confirm: ")
    if answer.strip() != task_name:
        web.delete(task_id)
        print(f"[run] {task_name} not started (draft deleted).")
        return None
    web.start(task_id)
    web.monitor(task_id, verbose=True)
    data = web.load(task_id, path=path)
    print(f"[run] {task_name}: real cost {web.real_cost(task_id)} FlexCredit")
    return data
