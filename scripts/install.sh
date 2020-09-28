set -exvu

poetry shell
poetry install
poetry run pre-commit install
