#!/usr/bin/env bash
# Publish a new version of google_gnm_model.tox as a GitHub release.
#
# Usage:
#   ./release.sh <version> [path/to/new.tox] [--notes notes.md] [--dry-run]
#
# Examples:
#   ./release.sh 1.0.3                          # .tox already saved into this folder
#   ./release.sh 1.0.3 ~/Desktop/gnm.tox        # copy it in from elsewhere first
#   ./release.sh 1.0.3 --notes notes.md         # release notes from a file
#
# Without --notes, your $EDITOR opens to write the release notes.
# The asset is always uploaded as google_gnm_model.tox so the README's
# "releases/latest/download/google_gnm_model.tox" link keeps working.
set -euo pipefail

TOX=google_gnm_model.tox
cd "$(dirname "$0")"

die() { echo "error: $*" >&2; exit 1; }

version="" src="" notes="" dry=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --notes) notes="${2:?--notes needs a file}"; shift 2 ;;
    --dry-run) dry=1; shift ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) if [[ -z "$version" ]]; then version="$1"; elif [[ -z "$src" ]]; then src="$1"; else die "unexpected argument: $1"; fi; shift ;;
  esac
done

[[ -n "$version" ]] || die "missing version (e.g. ./release.sh 1.0.3)"
version="${version#v}"
[[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "version must look like 1.2.3"
tag="v$version"

command -v gh >/dev/null || die "gh CLI not installed"
gh auth status >/dev/null 2>&1 || die "gh not logged in (run: gh auth login)"
[[ "$(git branch --show-current)" == main ]] || die "not on main"
git rev-parse -q --verify "refs/tags/$tag" >/dev/null && die "tag $tag already exists"
git fetch -q origin main
[[ "$(git rev-list --count HEAD..origin/main)" == 0 ]] || die "local main is behind origin — pull first"

# Only the .tox (and this script) may have uncommitted changes.
other=$(git status --porcelain | grep -v " $TOX\$" | grep -v " release.sh\$" || true)
[[ -z "$other" ]] || die "uncommitted changes besides $TOX:"$'\n'"$other"

if [[ -n "$src" ]]; then
  [[ -f "$src" ]] || die "no such file: $src"
  [[ "$src" == *.tox ]] || die "$src is not a .tox"
  ((dry)) || cp "$src" "$TOX"
  echo "copied $src -> $TOX"
fi
[[ -f "$TOX" ]] || die "$TOX not found"

if ((!dry)) && git diff --quiet HEAD -- "$TOX"; then
  die "$TOX is identical to the last release — save the new version from TouchDesigner first"
fi

last=$(git describe --tags --abbrev=0 2>/dev/null || echo none)
size=$(du -h "$TOX" | cut -f1 | tr -d ' ')
modified=$(stat -f '%Sm' "$TOX" 2>/dev/null || stat -c '%y' "$TOX")

if [[ -z "$notes" ]]; then
  notes=$(mktemp -t gnm-release-notes).md
  cat > "$notes" <<EOF
## Changes

-

**To update:** delete the old component from your network and drag in this .tox.
EOF
  ((dry)) || "${EDITOR:-nano}" "$notes"
fi
[[ -f "$notes" ]] || die "notes file not found: $notes"

echo
echo "  release:   $tag  (previous: $last)"
echo "  file:      $TOX  ($size, saved $modified)"
echo "  notes:     $notes"
echo
((dry)) && { echo "dry run — nothing committed or published."; exit 0; }

read -r -p "Commit, tag, push and publish $tag to GitHub? [y/N] " ok
[[ "$ok" == [yY] ]] || die "aborted"

git add "$TOX"
git commit -q -m "Release $tag"
git tag -a "$tag" -m "$tag"
git push -q origin main "$tag"
gh release create "$tag" "$TOX" --title "GNM Head for TouchDesigner $tag" --notes-file "$notes"
echo "done: $(gh release view "$tag" --json url -q .url)"
