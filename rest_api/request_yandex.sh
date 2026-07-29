YANDEXGPT_API_KEY=
YANDEXGPT_FOLDER_ID=

curl \
    --request POST \
    --header "Content-Type: application/json" \
    --header "Authorization: Bearer ${YANDEXGPT_API_KEY}" \
    --header "x-folder-id: ${YANDEXGPT_FOLDER_ID}" \
    --data "@request_yandex.json" \
    "https://llm.api.cloud.yandex.net/foundationModels/v1/completion" \
    | jq
