FROM --platform=linux/x86_64 python:3.12.8

RUN pip install poetry
RUN poetry config virtualenvs.create false

COPY pyproject.toml poetry.lock main.py ./

RUN poetry install

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
