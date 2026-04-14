from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PipelineRunMode(str, Enum):
    DEMO = "demo"
    RESEARCH = "research"
    API = "api"


@dataclass(frozen=True)
class RunModeOptions:
    show: bool
    save_txt: bool
    collect_metrics: bool
    persist_events: bool
    persist_video: bool


MODE_DEFAULTS: dict[PipelineRunMode, RunModeOptions] = {
    PipelineRunMode.DEMO: RunModeOptions(
        show=True,
        save_txt=False,
        collect_metrics=False,
        persist_events=False,
        persist_video=False,
    ),
    PipelineRunMode.RESEARCH: RunModeOptions(
        show=False,
        save_txt=True,
        collect_metrics=True,
        persist_events=True,
        persist_video=True,
    ),
    PipelineRunMode.API: RunModeOptions(
        show=False,
        save_txt=False,
        collect_metrics=True,
        persist_events=True,
        persist_video=True,
    ),
}


def normalize_run_mode(mode: str | PipelineRunMode) -> PipelineRunMode:
    if isinstance(mode, PipelineRunMode):
        return mode
    return PipelineRunMode(str(mode).lower())


def build_run_mode_options(
    mode: str | PipelineRunMode,
    *,
    show: bool | None = None,
    save_txt: bool | None = None,
    collect_metrics: bool | None = None,
    persist_events: bool | None = None,
    persist_video: bool | None = None,
) -> RunModeOptions:
    resolved_mode = normalize_run_mode(mode)
    defaults = MODE_DEFAULTS[resolved_mode]
    return RunModeOptions(
        show=defaults.show if show is None else show,
        save_txt=defaults.save_txt if save_txt is None else save_txt,
        collect_metrics=defaults.collect_metrics if collect_metrics is None else collect_metrics,
        persist_events=defaults.persist_events if persist_events is None else persist_events,
        persist_video=defaults.persist_video if persist_video is None else persist_video,
    )
