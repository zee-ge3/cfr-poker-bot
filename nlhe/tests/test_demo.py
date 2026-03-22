from streamlit.testing.v1 import AppTest


def test_app_has_two_tabs():
    at = AppTest.from_file("nlhe/demo/app.py", default_timeout=30)
    at.run()
    assert not at.exception, f"App raised: {at.exception}"
    assert len(at.tabs) == 2


def test_deal_button_exists():
    at = AppTest.from_file("nlhe/demo/app.py", default_timeout=30)
    at.run()
    buttons = [b.label for b in at.button]
    assert "Deal New Hand" in buttons
