GIGACHAT_AUTH_KEY=

RQUID=`uuidgen`

ACCESS_TOKEN=`curl --insecure \
    --location \
    --request POST \
    --header "Content-Type: application/x-www-form-urlencoded" \
    --header "RqUID: ${RQUID}" \
    --header "Authorization: Basic ${GIGACHAT_AUTH_KEY}" \
    --data-urlencode "scope=GIGACHAT_API_PERS" \
    "https://ngw.devices.sberbank.ru:9443/api/v2/oauth" \
    | jq -r ".access_token"`

curl --insecure \
    --request POST \
    --header "Content-Type: application/json" \
    --header "Authorization: Bearer ${ACCESS_TOKEN}" \
    --data "@request_sber.json" \
    "https://gigachat.devices.sberbank.ru/api/v1/chat/completions" \
    | jq
