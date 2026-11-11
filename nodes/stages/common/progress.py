import logging
import time

import comfy.model_management as model_management


def _emit_progress(cls, value: int, total: int, text: str = "") -> None:
    try:
        node_id = getattr(cls.hidden, "unique_id", None) if hasattr(cls, "hidden") else None

        from ....runners.remote_comfy import CURRENT_JOB, emit_remote_progress
        job = CURRENT_JOB.get()
        if job is not None:
            emit_remote_progress(node_id or job.stage_node_id, value, total,
                                 text=text)
            return

        from comfy.utils import ProgressBar
        pbar = ProgressBar(total, node_id=node_id)
        pbar.update_absolute(value, total)
        if text:
            from server import PromptServer
            if node_id is not None:
                PromptServer.instance.send_progress_text(text, node_id)
    except Exception as e:
        logging.warning("[ComfyTV] progress emit failed: %s", e)


def _check_interrupted() -> None:
    model_management.throw_exception_if_processing_interrupted()


def _is_processing_interrupt(exc: BaseException) -> bool:
    return type(exc).__name__ == "InterruptProcessingException"


def _fake_run_ticks(cls, steps: int = 4, delay_s: float = 0.12) -> None:
    for i in range(steps):
        _check_interrupted()
        _emit_progress(cls, i, steps, text=f"step {i + 1}/{steps}")
        time.sleep(delay_s)
