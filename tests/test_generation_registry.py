"""Registries for the swappable generation stages exist and behave like the others."""

from inr_unet.registry import BACKGROUNDS, POTENTIALS, Registry


def test_generation_registries_exist():
    assert isinstance(POTENTIALS, Registry)
    assert isinstance(BACKGROUNDS, Registry)


def test_registry_register_and_get():
    reg = Registry("tmp")

    @reg.register("thing")
    def f():
        return 7

    assert "thing" in reg
    assert reg.get("thing")() == 7
