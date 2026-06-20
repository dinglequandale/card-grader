# Intentionally minimal. The live video pipeline imports phase3 submodules
# directly (e.g. `from phase3.grade_synthesis import aggregate`,
# `from phase3 import edge_inspect`), so the package must NOT eagerly import the
# VLM / torch modules here -- that would force google-genai + torch as runtime
# deps the video pipeline never uses. The retired experiments/ scripts that did
# `from phase3 import fetch_reference` should import `from phase3.reference ...`.
