#!/usr/bin/env bash

set -e
cd "./intermediate"

for zipfile in *.zip; do
  echo "Processing $zipfile"

  # Remove .zip
  base="${zipfile%.zip}"

  # Extract XXXXX
  if [[ "$base" =~ ^([0-9]{5})-([0-9]{4})-.*$ ]]; then
    XXXXX="${BASH_REMATCH[1]}"
    YYYY="${BASH_REMATCH[2]}"

    unzip -q "$zipfile"

    pdf="ICPSR_${XXXXX}/DS${YYYY}/${base}.pdf"

    if [[ -f "$pdf" ]]; then
      mv "$pdf" .
      rm -rf "ICPSR_${XXXXX}"
      echo "✔ Moved $base.pdf"
    else
      echo "✖ PDF not found: $pdf"
    fi

  elif [[ "$base" =~ ^([0-9]{5})-.*$ ]]; then
    XXXXX="${BASH_REMATCH[1]}"

    unzip -q "$zipfile"

    pdf="ICPSR_${XXXXX}/${base}.pdf"

    if [[ -f "$pdf" ]]; then
      mv "$pdf" .
      rm -rf "ICPSR_${XXXXX}"
      echo "✔ Moved $base.pdf"
    else
      echo "✖ PDF not found: $pdf"
    fi

  else
    echo "⚠ Skipping (unexpected name format): $zipfile"
  fi

done

echo "Done."