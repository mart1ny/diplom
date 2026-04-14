from scripts.run_modes import PipelineRunMode, build_run_mode_options, normalize_run_mode


def test_normalize_run_mode_accepts_strings_and_enum() -> None:
    assert normalize_run_mode("demo") is PipelineRunMode.DEMO
    assert normalize_run_mode(PipelineRunMode.API) is PipelineRunMode.API


def test_research_mode_defaults_to_metrics_and_exports() -> None:
    options = build_run_mode_options(PipelineRunMode.RESEARCH)
    assert options.show is False
    assert options.save_txt is True
    assert options.collect_metrics is True
    assert options.persist_events is True
    assert options.persist_video is True


def test_demo_mode_can_be_overridden_explicitly() -> None:
    options = build_run_mode_options(
        PipelineRunMode.DEMO,
        show=False,
        persist_video=True,
    )
    assert options.show is False
    assert options.persist_video is True
    assert options.collect_metrics is False
