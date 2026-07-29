OPENAI_API_KEY=""
API_BASE_URL=

# curl ${API_BASE_URL}/models --header "Authorization: Bearer ${OPENAI_API_KEY}" | jq

curl --location \
    --request POST \
    --header "Content-Type: application/json" \
    --header "Authorization: Bearer ${OPENAI_API_KEY}" \
    --data "@request_openai.json" \
    "${API_BASE_URL}/chat/completions" \
    | jq
