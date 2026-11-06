#!/bin/bash
# Build the vendored wheels and place them in vendor/:
#   - sap-cloud-sdk: built from the git URL in the root-level requirements.txt.
#   - google-re2: downloaded as a linux/cp312 wheel to work around an sfw bug
#     (sfw crashes fetching PyPI alerts for google-re2); resolving it locally
#     via --find-links avoids that network call. Keep the pin in sync with
#     requirements.txt.
#
# Usage:
#   ./vendor.sh
#
# After running, the vendor/ directory will contain both wheels.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR/../../../.."
VENDOR_DIR="$SCRIPT_DIR/vendor"
ROOT_REQUIREMENTS="$ROOT_DIR/requirements.txt"

# Keep in sync with the google-re2 pin in requirements.txt.
GOOGLE_RE2_VERSION="1.1.20240702"

echo "Building sap-cloud-sdk wheel from git..."
echo "  Using requirements from: $ROOT_REQUIREMENTS"

# Extract the sap-cloud-sdk git URL from root requirements.txt
SAP_SDK_LINE=$(grep "^sap-cloud-sdk" "$ROOT_REQUIREMENTS" || true)

if [ -z "$SAP_SDK_LINE" ]; then
    echo "Error: Could not find sap-cloud-sdk in $ROOT_REQUIREMENTS"
    exit 1
fi

echo "  Found: $SAP_SDK_LINE"

# Clean existing vendor directory
rm -rf "$VENDOR_DIR"
mkdir -p "$VENDOR_DIR"

# Create a temporary directory for building
BUILD_DIR=$(mktemp -d)
trap "rm -rf $BUILD_DIR" EXIT

# Download and build wheel from git URL
echo "  Building wheel..."
pip3 wheel \
    "$SAP_SDK_LINE" \
    --wheel-dir "$VENDOR_DIR" \
    --no-deps \
    --no-cache-dir

# Download the google-re2 wheel for the builder platform (linux, cpython 3.12).
echo "  Downloading google-re2==${GOOGLE_RE2_VERSION} (linux cp312)..."
pip3 download "google-re2==${GOOGLE_RE2_VERSION}" \
    --dest "$VENDOR_DIR" \
    --no-deps \
    --no-cache-dir \
    --only-binary=:all: \
    --platform manylinux_2_28_x86_64 \
    --python-version 312 \
    --implementation cp \
    --abi cp312

# Count downloaded files
COUNT=$(ls -1 "$VENDOR_DIR"/*.whl 2>/dev/null | wc -l)
echo "✓ Built/downloaded $COUNT wheel(s) in vendor/"

if [ "$COUNT" -eq 0 ]; then
    echo "⚠ Warning: No wheels built. Check the git URL in requirements.txt."
    exit 1
fi

# List the wheels
echo ""
echo "Vendored wheels:"
ls -la "$VENDOR_DIR"/*.whl

echo ""
echo "✓ Done. Ensure requirements.txt references the vendored wheels:"
echo "  sap-cloud-sdk @ file:vendor/<wheel-filename>"
echo "  google-re2==${GOOGLE_RE2_VERSION}   (resolved from vendor/ via find-links)"
