#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[[ -f app/src/main/AndroidManifest.xml ]]
[[ -f app/src/main/cpp/director_vlm.cpp ]]
[[ -f app/src/main/assets/skills.json ]]
[[ -f scripts/compile_skills.py ]]
python3 scripts/compile_skills.py
python3 -m json.tool app/src/main/assets/skills.json >/dev/null
python3 - <<'PY'
import pathlib, xml.etree.ElementTree as ET
for p in pathlib.Path('app/src/main/res').rglob('*.xml'):
    ET.parse(p)
ET.parse('app/src/main/AndroidManifest.xml')
print('XML/skills validation passed')
PY
if [[ -n "${ANDROID_SDK_ROOT:-${ANDROID_HOME:-}}" ]]; then
  ./gradlew testDebugUnitTest lintDebug --stacktrace
else
  echo "Android SDK not present: static validation passed; Gradle Android tasks skipped."
fi
