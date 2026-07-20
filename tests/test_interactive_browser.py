from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.browser
ROOT = Path(__file__).resolve().parents[1]
PAGES = (
    "index.html",
    "foundations-lab.html",
    "architecture-lab.html",
    "training-and-alignment.html",
    "serving-lab.html",
    "multimodal-flow.html",
)


@pytest.mark.skipif(
    os.environ.get("RUN_BROWSER_TESTS") != "1",
    reason="set RUN_BROWSER_TESTS=1 and install the webqa group",
)
def test_interactive_pages_run_without_console_errors_and_fit_mobile() -> None:
    sync_api = pytest.importorskip("playwright.sync_api")
    with sync_api.sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        for name in PAGES:
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            errors: list[str] = []
            page.on(
                "console",
                lambda message, errors=errors: (
                    errors.append(message.text) if message.type == "error" else None
                ),
            )
            page.goto((ROOT / "learning" / "readings" / "interactive" / name).as_uri())
            page.wait_for_load_state("domcontentloaded")
            control = page.locator("input, button, select").first
            if control.count():
                control.focus()
                if control.evaluate("element => element.tagName === 'INPUT'"):
                    control.press("ArrowRight")
            page.set_viewport_size({"width": 320, "height": 800})
            page.wait_for_timeout(50)
            overflow = page.evaluate(
                "() => document.documentElement.scrollWidth - document.documentElement.clientWidth"
            )
            assert overflow <= 1, f"{name} has {overflow}px horizontal overflow at 320px"
            assert not errors, f"{name} console errors: {errors}"
            page.close()
        browser.close()
