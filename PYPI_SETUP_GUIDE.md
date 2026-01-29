# PyPI Automatic Publishing Setup Guide

## Overview
Your repository already has a GitHub Actions workflow (`.github/workflows/release_to_pypi.yml`) that automatically publishes to PyPI when you push a tag starting with `v` (e.g., `v0.1.3`).

## Step 1: Create a PyPI API Token

### Option A: Trusted Publishing (Recommended - No token needed!)

PyPI now supports "Trusted Publishers" which is more secure and doesn't require storing tokens.

1. Go to https://pypi.org/ and log in
2. Navigate to your package: https://pypi.org/manage/project/scikit-explain/
3. Go to "Publishing" tab
4. Click "Add a new publisher"
5. Fill in:
   - **PyPI Project Name**: `scikit-explain`
   - **Owner**: `monte-flora` (your GitHub username)
   - **Repository name**: `scikit-explain`
   - **Workflow name**: `release_to_pypi.yml`
   - **Environment name**: (leave blank)
6. Save

Then update your workflow to remove the password (see Step 3 below).

### Option B: API Token (Current Method)

If you prefer using API tokens:

1. Go to https://pypi.org/manage/account/token/
2. Click "Add API token"
3. Give it a name like "scikit-explain-github-actions"
4. **Scope**: Choose "Project: scikit-explain" (more secure than account-wide)
5. Click "Add token"
6. **IMPORTANT**: Copy the token immediately (starts with `pypi-`). You won't see it again!

## Step 2: Add Token to GitHub Secrets

If using Option B (API Token):

1. Go to your GitHub repository: https://github.com/monte-flora/scikit-explain
2. Click "Settings" tab
3. In the left sidebar, click "Secrets and variables" → "Actions"
4. Click "New repository secret"
5. Name: `PYPI_API_TOKEN`
6. Value: Paste your PyPI token (starts with `pypi-`)
7. Click "Add secret"

## Step 3: Update GitHub Workflow (Optional Improvements)

Your current workflow works, but here are recommended updates:

### For Trusted Publishing (Option A):
Update `.github/workflows/release_to_pypi.yml` line 27-31 to:

```yaml
      - name: Publish distribution to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          # No password needed with Trusted Publishing!
```

### For API Token (Option B):
Keep current setup, but update action versions:

```yaml
      - name: Publish distribution to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          user: __token__
          password: ${{ secrets.PYPI_API_TOKEN }}
```

### Additional Improvements:
- Update `actions/checkout@v2` → `actions/checkout@v4`
- Update `actions/setup-python@v2` → `actions/setup-python@v5`
- Update Python version from 3.8 to 3.9 or higher

## Step 4: Update Version and Create Release

### Update Version in setup.py

Edit `setup.py` line 28:

```python
VERSION = "0.1.3"  # Increment from current 0.1.2
```

### Commit and Push Version Change

```bash
git add setup.py
git commit -m "Bump version to 0.1.3"
git push origin master
```

### Create and Push Tag

```bash
# Create an annotated tag
git tag -a v0.1.3 -m "Release v0.1.3 - Fix 2D ALE plotting with different bin counts"

# Push the tag to GitHub
git push origin v0.1.3
```

**Alternative**: Use a single command to tag the latest commit:
```bash
git tag -a v0.1.3 -m "Release v0.1.3" && git push origin v0.1.3
```

### Monitor the Release

1. Go to https://github.com/monte-flora/scikit-explain/actions
2. You should see the "Publish to PyPI" workflow running
3. Once complete (green checkmark), your package is on PyPI!
4. Check: https://pypi.org/project/scikit-explain/

## Workflow Explanation

When you push a tag starting with `v`:

1. ✅ GitHub Actions triggers the workflow
2. ✅ Checks out your code
3. ✅ Sets up Python 3.8
4. ✅ Builds source distribution (`.tar.gz`) and wheel (`.whl`)
5. ✅ Validates the distributions with `twine check`
6. ✅ Publishes to PyPI using your token or trusted publisher
7. ✅ Users can now install with: `pip install scikit-explain==0.1.3`

## Tag Naming Convention

- ✅ `v0.1.3` - Patch version (bug fixes)
- ✅ `v0.2.0` - Minor version (new features, backward compatible)
- ✅ `v1.0.0` - Major version (breaking changes)

Follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes

## Troubleshooting

### "File already exists" Error
- You can't overwrite versions on PyPI
- Increment the version number in `setup.py` and create a new tag

### "Authentication failed" Error
- Check that `PYPI_API_TOKEN` secret is correctly set in GitHub
- Verify the token hasn't expired or been revoked
- Try creating a new token

### "Invalid distribution" Error
- Ensure `setup.py` has correct metadata
- Check that `README.md` exists and is in `MANIFEST.in`
- Run locally: `python -m build && twine check dist/*`

### Workflow Doesn't Trigger
- Ensure tag starts with `v` (e.g., `v0.1.3`, not `0.1.3`)
- Check GitHub Actions permissions in Settings → Actions → General

## Testing Before Publishing

Test the build locally before pushing a tag:

```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build the package
python -m pip install --upgrade build twine
python -m build

# Check the distribution
twine check dist/*

# Test upload to TestPyPI (optional)
twine upload --repository testpypi dist/*

# Test install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ scikit-explain
```

## Quick Reference

### Full Release Process

```bash
# 1. Update version in setup.py
# VERSION = "0.1.3"

# 2. Commit version change
git add setup.py
git commit -m "Bump version to 0.1.3"
git push origin master

# 3. Create and push tag
git tag -a v0.1.3 -m "Release v0.1.3 - Brief description of changes"
git push origin v0.1.3

# 4. Monitor: https://github.com/monte-flora/scikit-explain/actions

# 5. Verify: https://pypi.org/project/scikit-explain/
```

## Current Status

- ✅ GitHub Actions workflow exists: `.github/workflows/release_to_pypi.yml`
- ⏳ PyPI API token needs to be added to GitHub Secrets (or use Trusted Publishing)
- ✅ Current version: 0.1.2
- ⏳ Next version: 0.1.3 (after adding the 2D ALE fix)

## For Your Recent Fix

Since you just merged the 2D ALE plotting fix, here's what to do:

```bash
# 1. Update version for the bug fix (patch version)
# Edit setup.py: VERSION = "0.1.3"

# 2. Commit
git add setup.py
git commit -m "Bump version to 0.1.3"
git push origin master

# 3. Tag and push
git tag -a v0.1.3 -m "Release v0.1.3 - Fix 2D ALE plotting with different bin counts"
git push origin v0.1.3
```

The GitHub Action will automatically build and publish to PyPI!
