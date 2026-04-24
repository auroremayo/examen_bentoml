.PHONY: serve token predict

PORT ?= 3001
# On pointe vers VOTRE classe de service dans src/service.py
SERVICE ?= src.service:AdmissionApiService
BASE_URL ?= http://127.0.0.1:$(PORT)

serve:
	uv run bentoml serve $(SERVICE) --port $(PORT)

token:
	@curl -s -X POST "$(BASE_URL)/login" \
		-H "Content-Type: application/json" \
		-d '{"credentials":{"username":"admin","password":"bentoml_2026"}}'

predict:
	@token=$$(curl -s -X POST "$(BASE_URL)/login" \
		-H "Content-Type: application/json" \
		-d '{"credentials":{"username":"admin","password":"bentoml_2026"}}' | jq -r '.access_token'); \
	curl -s -X POST "$(BASE_URL)/predict" \
		-H "Content-Type: application/json" \
		-H "Authorization: Bearer $$token" \
		-d '{"input_data": {"GRE Score": 320, "TOEFL Score": 110, "University Rating": 4, "SOP": 4.5, "LOR": 4.0, "CGPA": 8.8, "Research": 1}}'; \
	echo