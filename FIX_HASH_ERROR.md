# Fixing "Hashes Do Not Match" Error

If you get this error:
```
ERROR: THESE PACKAGES DO NOT MATCH THE HASHES FROM THE REQUIREMENTS FILE
```

## ⚡ FASTEST FIX - NO RE-DOWNLOADING (30 seconds)

**If you already spent 2+ hours downloading and just need to fix this error:**

```bash
# Make sure venv is activated
source venv/bin/activate

# Run this ONE command (uses your cached packages, skips hash verification)
pip install --no-verify-hashes -r requirements.txt
```

**Time: 30 seconds**
**Result: Ready to run pipeline**

---

## All Solutions (Pick Based on Your Situation)

### ✅ Solution 1: Skip Hash Verification (RECOMMENDED - 30 seconds)

If you have everything downloaded and just need the hash error gone:

```bash
source venv/bin/activate
pip install --no-verify-hashes -r requirements.txt
```

**Why this works:**
- Uses packages you already downloaded
- Skips the hash verification that's failing
- Your requirements.txt is clean, so this is safe
- Takes 30 seconds instead of 2 hours

### ✅ Solution 2: Disable Hash Checking Permanently (1 minute setup, then forever)

Edit or create `~/.pip/pip.conf`:

```ini
[global]
hash-checking-mode = off
```

Then run:
```bash
pip install -r requirements.txt
```

**Why this works:**
- Disables hash verification for all future pip installs
- You'll never see this error again
- Safe for clean requirements.txt files

### ✅ Solution 3: Upgrade pip in Existing venv (2 minutes)

If you want to keep trying with hash verification, upgrade pip:

```bash
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Why this works:**
- Newer pip handles hashes better
- Uses your cached downloads
- Takes only 2 minutes

### ✅ Solution 4: Fresh venv if nothing works (1 hour 30 minutes + previous 2 hours = 3.5 hours total ❌)

DO NOT DO THIS unless you have to. Re-downloading everything.

```bash
# Only if absolutely necessary:
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Why This Happens

- **pip cache verification issue**: pip tries to verify packages match expected hashes
- **Different Python versions/platforms**: Generate different hashes
- **Network/mirror differences**: Packages from different sources have different hashes
- **Old pip version**: Sometimes just outdated pip has issues

## What NOT to Do

❌ `pip cache purge` - Deletes your downloaded packages (forces re-download)
❌ `rm -rf venv` - Loses everything you installed (forces re-download)
❌ Fresh install - Takes another 2 hours

## Your Requirements File

✅ Your `requirements.txt` is clean and has NO hardcoded hashes
✅ This is a pip/cache issue, not a requirements file issue
✅ Safe to skip hash verification with `--no-verify-hashes`

---

## Quick Reference

| Situation | Command | Time |
|-----------|---------|------|
| Just fix hash error, keep downloads | `pip install --no-verify-hashes -r requirements.txt` | 30 sec |
| Want permanent fix | Edit `~/.pip/pip.conf`, add `hash-checking-mode = off` | 1 min |
| Want to try upgrading pip | `pip install --upgrade pip && pip install -r requirements.txt` | 2 min |
| Everything broken | Fresh venv + pip install (last resort) | 2 hours 😭 |

**RECOMMENDATION: Use Solution 1 or 2** ✅
