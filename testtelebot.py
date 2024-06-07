import requests

TELE_TOKEN = "7277909972:AAFjj5meKrGqF5msuvmaxUx5prRxseIPWdE"
CHAT_ID = "5634015200"

message = "swiggity"

url = f"https://api.telegram.org/bot{TELE_TOKEN}/sendMessage?chat_id={CHAT_ID}&text={message}"

print(requests.get(url).json())