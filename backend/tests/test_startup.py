"""Startup safety checks that need no database."""


def test_default_secret_key_is_flagged_at_startup(caplog, monkeypatch):
    """The published default SECRET_KEY must never pass silently."""
    import logging

    from fastapi.testclient import TestClient

    import app.main as main

    monkeypatch.setattr(main.settings, "SECRET_KEY", main.INSECURE_DEFAULT_SECRET)
    monkeypatch.setattr(main.Base.metadata, "create_all", lambda **k: None)   # no DB needed
    with caplog.at_level(logging.WARNING, logger="urbaneye"):
        with TestClient(main.app):
            pass
    assert any("SECRET_KEY is the published default" in r.message for r in caplog.records)
