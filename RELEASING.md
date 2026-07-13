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
(`pyproject.toml` reads it dynamically). Also update `CITATION.cff`
(`version:` and `date-released:`) so the citation matches the release.

```bash
# 1. bump the version
#    edit pysfbox/__init__.py :  __version__ = "1.0.1"
#    edit CITATION.cff        :  version: 1.0.1  and  date-released: "YYYY-MM-DD"
git commit -am "Release 1.0.1"

# 2. tag it (v + the exact version) and push the tag
git tag v1.0.1
git push origin main --tags
```

Pushing the tag runs the workflow: it builds the sdist + wheel, checks the tag
matches the built version, **publishes to PyPI**, and **creates a GitHub
Release** (with the wheel/sdist attached). Within a minute or two
`pip install pysfbox` (and `pipx install pysfbox`) serve the new version.

## Citation & DOI (Zenodo)

`CITATION.cff` gives GitHub a "Cite this repository" button and machine-readable
citation metadata. For a citable **DOI**, enable the Zenodo–GitHub archive:

1. One-time: sign in at https://zenodo.org with GitHub, open **GitHub** in the
   settings, and flip the toggle **on** for `mark-vis/pysfbox`. Do this
   *before* the release you want archived — Zenodo only archives releases
   created after the toggle is on (it does not back-fill existing ones).
2. Because the publish workflow now creates a GitHub Release for every tag,
   each new release is archived by Zenodo automatically and gets its own DOI,
   plus a **concept DOI** that always resolves to the latest version.
3. After the first archived release, copy the concept DOI into `CITATION.cff`
   (`doi:` line) and add a DOI badge to `README.md`.

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
