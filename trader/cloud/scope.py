import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]

SPREADSHEET_ID = "1CkQsRjpdVKGYWHQ_apgXyra4Fe-8UtkL1uzyWxTlqoM"
RANGE_NAME = "Debug!A2:E"
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "local_secrets")

def main():

  creds = None

  if os.path.exists(os.path.join(ROOT_DIR, "token.json")):
    creds = Credentials.from_authorized_user_file(os.path.join(ROOT_DIR, "token.json"), SCOPES)

  if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
      creds.refresh(Request())
    else:
      flow = InstalledAppFlow.from_client_secrets_file(
          os.path.join(ROOT_DIR, "cloud_trader_credentials.json"), SCOPES
      )
      creds = flow.run_local_server(port=0)
    with open(os.path.join(ROOT_DIR, "token.json"), "w") as token:
      token.write(creds.to_json())

  try:
    service = build("sheets", "v4", credentials=creds)

    sheet = service.spreadsheets()
    result = (
        sheet.values()
        .get(spreadsheetId=SPREADSHEET_ID, range=RANGE_NAME)
        .execute()
    )
    values = result.get("values", [])

    if not values:
      logging.warning("No data found.")
      return

    for row in values:
      logging.info(f"Row: {row}")
  except HttpError as err:
    logging.error(f"Error: {err}")


if __name__ == "__main__":
  main()