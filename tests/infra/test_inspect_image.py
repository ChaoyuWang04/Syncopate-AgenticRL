from syncopate.infra import real_training_scenario
import pytest


def test_inspect_image_uses_current_modal_shell_interface():
    builder = getattr(real_training_scenario, "build_inspection_command", None)
    assert callable(builder)
    command = builder("verlai/verl:vllm024.latest")
    assert command[:4] == ["modal", "shell", "--image", "verlai/verl:vllm024.latest"]
    assert command[-1] == "--no-pty"


def test_inspect_image_propagates_nested_modal_failure():
    checker = getattr(real_training_scenario, "check_inspection_result", None)
    assert callable(checker)
    with pytest.raises(RuntimeError, match="manifest unknown"):
        checker(1, "", "manifest unknown")
