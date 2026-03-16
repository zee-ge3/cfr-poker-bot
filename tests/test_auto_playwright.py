# tests/test_auto_playwright.py
import os, sys, zipfile, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_build_spy_zip_contains_correct_player():
    """spy zip: submission/player.py content is spy_player.py content."""
    from auto_playwright import build_zip
    with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as f:
        zip_path = f.name
    try:
        build_zip('spy', zip_path)
        with zipfile.ZipFile(zip_path) as z:
            names = z.namelist()
            assert any('submission/player.py' in n for n in names), \
                f"submission/player.py not in zip: {names}"
            content = z.read([n for n in names if n.endswith('submission/player.py')][0]).decode()
            assert 'SpyAgent' in content, "spy zip player.py doesn't contain SpyAgent"
    finally:
        os.unlink(zip_path)


def test_build_main_zip_contains_main_player():
    """main zip: submission/player.py content is the real player.py."""
    from auto_playwright import build_zip
    with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as f:
        zip_path = f.name
    try:
        build_zip('main', zip_path)
        with zipfile.ZipFile(zip_path) as z:
            names = z.namelist()
            content = z.read([n for n in names if n.endswith('submission/player.py')][0]).decode()
            assert 'SpyAgent' not in content, "main zip shouldn't contain SpyAgent"
    finally:
        os.unlink(zip_path)
