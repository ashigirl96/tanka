set -exvu

poetry install
poetry run pre-commit install
