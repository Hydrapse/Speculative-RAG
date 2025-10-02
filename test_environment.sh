#!/usr/bin/env bash

echo "=== Environment Diagnostic Test ==="
echo ""

echo "1. Current Directory:"
pwd
echo ""

echo "2. Directory Contents:"
ls -la
echo ""

echo "3. Python Version:"
python --version 2>&1
echo ""

echo "4. Poetry Status:"
poetry --version 2>&1
echo ""

echo "5. Key Files Check:"
for file in README.md pyproject.toml speculative_rag.ipynb langgraph_speculative_rag.ipynb; do
    if [ -f "$file" ]; then
        echo "✓ $file exists ($(stat -c%s "$file") bytes)"
    else
        echo "✗ $file NOT FOUND"
    fi
done
echo ""

echo "6. Qdrant Data:"
if [ -d "qdrant_client" ]; then
    echo "✓ qdrant_client directory exists"
    du -sh qdrant_client 2>/dev/null || echo "Cannot read size"
else
    echo "✗ qdrant_client directory NOT FOUND"
fi
echo ""

echo "7. Git Status:"
git status 2>&1 | head -5
echo ""

echo "8. Environment Variables:"
echo "SHELL: $SHELL"
echo "USER: $USER"
echo "HOME: $HOME"
echo ""

echo "=== Test Complete ==="
