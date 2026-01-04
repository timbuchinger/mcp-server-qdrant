Using the repository virtual environment (.venv)

1. Create the virtualenv (if not present):

   python3 -m venv .venv

2. Activate it:

   source .venv/bin/activate

3. Install the package and test deps (editable):

   pip install -e . --no-deps
   pip install pytest pytest-asyncio qdrant-client fastmcp

4. Run tests:

   pytest -q

Note: This project expects the .venv directory at the repository root; the above steps create and use that virtual environment.
