.PHONY: install train monitor mlflow-ui test serve docker lint

install:
	pip install -r requirements-dev.txt

train:
	cd src && python -m churn.train

monitor:
	cd src && python -m churn.simulate_production

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db

test:
	pytest -q

lint:
	ruff check src tests

serve:
	uvicorn churn.api:app --app-dir src --reload

docker:
	docker build -t churn-api . && docker run -p 8000:8000 churn-api
