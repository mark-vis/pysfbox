# Releasing PySFBox to PyPI

Releases are published automatically by `.github/workflows/publish.yml` when a
version tag is pushed, using **PyPI Trusted Publishing** (OpenID Connect) — no
API tokens or repository secrets are stored.

## One-time setup (on PyPI)

1. Sign in at https://pypi.org (enable 2FA).
2. Go to **Account → Publishing → Add a pending publisher** and register:
   - **PyPI Project Name:** `pysfbox`
   - **Owner:** `mark-vis`
   - **Repository name:** `pysfbox`
   - **Workflow name:** `publish.yml`
   - **Environment name:** `pypi`

   (A *pending* publisher can be added before the project exists on PyPI; the
   first tagged release then creates it.)
3. Optional but recommended: in the GitHub repo, **Settings → Environments →
   New environment `pypi`**, and add a required reviewer so a human approves
   each publish.

## Cutting a release

The version lives in **one place**: `__version__` in `pysfbox/__init__.py`
(`pyproject.toml` reads it dynamically).

```bash
# 1. bump the version
#    edit pysfbox/__init__.py:  __version__ = "1.0.1"
git commit -am "Release 1.0.1"

# 2. tag it (v + the exact version) and push the tag
git tag v1.0.1
git push origin main --tags
```

Pushing the tag runs the workflow: it builds the sdist + wheel, checks the tag
matches the built version, and publishes to PyPI. Within a minute or two
`pip install pysfbox` (and `pipx install pysfbox`) serve the new version.

## Test run first (optional)

To rehearse without touching real PyPI, add a TestPyPI trusted publisher
(same fields, on https://test.pypi.org) and temporarily point the publish step
at it with `with: { repository-url: https://test.pypi.org/legacy/ }`, or upload
a local build by hand:

```bash
python -m build
twine upload -r testpypi dist/*
pip install -i https://test.pypi.org/simple/ pysfbox
```

## Notes

- Publish **this** (public) repository only — never the development tree.
- A PyPI version number can be uploaded **once**; you cannot overwrite it. Bump
  the version for any re-release.
- Before the first public PyPI release, confirm with F.A.M. Leermakers that a
  formal PyPI distribution of `pysfbox/sfnewton.py` (Wageningen copyright /
  reproduction notice, see `NOTICE`) is acceptable — it is already on public
  GitHub, but PyPI is a broader, permanent channel.
