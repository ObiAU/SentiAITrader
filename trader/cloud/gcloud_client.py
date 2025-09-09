import json
import logging
import os
import tempfile
from typing import List, Any, Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from trader.config import Config
from trader.database.supa_client import execute_sql_query


SECRETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "local_secrets")
SPREADSHEET_ID = Config.SPREADSHEET_ID
POSITION_HISTORY_SHEET_NAME = "Position History"
TRADES_HISTORY_SHEET_NAME = "Trade History"

logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
class GoogleCloudClient:
    """
    Base class for Google Cloud clients, handles authentication.
    """
    def __init__(
                self,
                scopes: List[str] = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"],
                local: bool = True
                ) -> None:
        """
        Init gcloud client with scopes
        """
        self.scopes = scopes
        self.creds = None

        if not local:
            self.creds = self.authenticate_session_cloud()
        
        else:
            self.creds = self.authenticate_session_local()
        
    def authenticate_session_local(self) -> Optional[Credentials]:
        creds = None
        if os.path.exists(os.path.join(SECRETS_DIR, "token.json")):
            creds = Credentials.from_authorized_user_file(os.path.join(SECRETS_DIR, "token.json"), self.scopes)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    os.path.join(SECRETS_DIR, "cloud_trader_credentials.json"), self.scopes
                )
                creds = flow.run_local_server(port=0)
                with open(os.path.join(SECRETS_DIR, "token.json"), "w") as token:
                    token.write(creds.to_json())

        return creds

    def authenticate_session_cloud(self) -> Optional[Credentials]:
        # for local testing
        creds = None
        with open(os.path.join(SECRETS_DIR, "gtoken.txt"), "r") as f:
                self.token = f.read()
        with open(os.path.join(SECRETS_DIR, "grefresh_token.txt"), "r") as f:
                # self.refresh_token = f.read()
                self.refresh_token = None 
        with open(os.path.join(SECRETS_DIR, "gsecret.txt"), "r") as f:
                self.client_secret = f.read()

        # keep if want to automate -- probs not necessary as can just run it once a day via local script
        # keeping separate from config as will be its own cron
        # required_env_vars = ['GCLOUD_CLIENT_SECRET',
        #                     'GCLOUD_TOKEN',
        #                     'GCLOUD_REFRESH_TOKEN']
        # for var in required_env_vars:
        #     if not os.environ.get(var):
        #         raise ValueError(f"Environment variable {var} is required but not set.")
        # if not self.client_secret or not self.token or not self.refresh_token:
        #     raise ValueError("Missing required credentials")

        user_info = {
            "installed": {
                "client_id": Config.GCLOUD_CLIENT_ID,
                "project_id": "trade-cloud-client",
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "client_secret": self.client_secret,
                "redirect_uris": ["http://localhost"]
            },
            "token": self.token,
            "refresh_token": self.refresh_token or "",
            "token_uri": "https://oauth2.googleapis.com/token",
            "client_id": Config.GCLOUD_CLIENT_ID,
            "client_secret": self.client_secret,
            "scopes": ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"],
            "universe_domain": "googleapis.com",
            "account": "",
            "expiry": Config.GCLOUD_TOKEN_EXPIRY
        }

        try:
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".json") as temp_file:

                json.dump(user_info, temp_file)
                temp_file.flush()

                creds = Credentials.from_authorized_user_file(temp_file.name, self.scopes)
            return creds

        except Exception as e:
            logging.error(f"Error during authentication: {e}")
            return None

class GoogleSheetsClient(GoogleCloudClient):
    """
    sheets client
    """

    def __init__(self, scopes: Optional[List[str]] = None) -> None:

        if scopes is None:
            scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        super().__init__(scopes)

    def create_gsheet(
        self,
        title: str,
        content: Optional[List[List[Any]]] = None,
        sheet_name: str = "Sheet1",
    ) -> str:

        try:
            service = build("sheets", "v4", credentials=self.creds)

            spreadsheet_body = {
                "properties": {"title": title},
                "sheets": [],
            }

            if content:

                sheet = {
                    "properties": {"title": sheet_name},
                    "data": [
                        {
                            "startRow": 0,
                            "startColumn": 0,
                            "rowData": [
                                {
                                    "values": [
                                        {
                                            "userEnteredValue": (
                                                {"stringValue": str(cell)}
                                                if isinstance(cell, str)
                                                else {"numberValue": cell}
                                            )
                                        }
                                        for cell in row
                                    ]
                                }
                                for row in content
                            ],
                        }
                    ],
                }
                spreadsheet_body["sheets"].append(sheet)
            else:
                sheet = {"properties": {"title": sheet_name}}
                spreadsheet_body["sheets"].append(sheet)

            spreadsheet = (
                service.spreadsheets()
                .create(body=spreadsheet_body, fields="spreadsheetId")
                .execute()
            )
            spreadsheet_id = spreadsheet.get("spreadsheetId")

            hyperlink = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}"

            logging.info(f"Created Google Sheet at: {hyperlink}")

            return hyperlink

        except HttpError as error:
            logging.error(f"An error occurred: {error}")
            raise error
        

    def update_sheet(
        self, spreadsheet_id: str, range_name: str, values: List[List[Any]]
    ) -> None:
        """
        spreadsheet id = string after spreadsheets/d/ in endpoint
        """
        try:
            service = build("sheets", "v4", credentials=self.creds)
            body = {"values": values}
            result = (
                service.spreadsheets()
                .values()
                .update(
                    spreadsheetId=spreadsheet_id,
                    range=range_name,
                    valueInputOption="USER_ENTERED",
                    body=body,
                )
                .execute()
            )
            logging.info(f"{result.get('updatedCells')} cells updated.")
        except HttpError as err:
            logging.error(f"There was an HTTP error: {err}")
            raise err

    def append_row(
        self, spreadsheet_id: str, sheet_name: str, row: List[Any]
    ) -> None:
        try:
            service = build("sheets", "v4", credentials=self.creds)
            body = {
                "values": [row]
            }
            result = (
                service.spreadsheets()
                .values()
                .append(
                    spreadsheetId=spreadsheet_id,
                    range=sheet_name,
                    valueInputOption="USER_ENTERED",
                    insertDataOption="INSERT_ROWS",
                    body=body,
                )
                .execute()
            )
            logging.info(f"{result.get('updates').get('updatedCells')} cells appended.")
        except HttpError as err:
            logging.error(f"There was an HTTP error: {err}")
            raise err

    def get_from_sheet(
        self, spreadsheet_id: str, range_name: str
    ) -> List[List[Any]]:

        try:
            service = build("sheets", "v4", credentials=self.creds)
            result = (
                service.spreadsheets()
                .values()
                .get(spreadsheetId=spreadsheet_id, range=range_name)
                .execute()
            )
            values = result.get("values", [])
            if not values:
                logging.warning("No data found.")
            return values
        except HttpError as err:
            logging.error(f"There was an HTTP error: {err}")
            raise err
        
    def update_trades_history_sheet(self, spreadsheet_id: str, sheet_name: str) -> None:
        """
        qol update trades history sheet -- identifier is trade_id
        """
        query = "SELECT * FROM trades ORDER BY timestamp ASC"
        trades_records = execute_sql_query(query)
        if not trades_records:
            logging.info("No trades records found.")
            return

        try:
            sheet_data = self.get_from_sheet(spreadsheet_id, sheet_name)
        except Exception as e:
            logging.error(f"Error retrieving sheet data: {e}")
            sheet_data = []

        if not sheet_data:
            first_record = trades_records[0]
            data = first_record.get("result", first_record)
            header = list(data.keys())
            logging.info("Appending header row for trades history:")
            self.append_row(spreadsheet_id, sheet_name, header)
            existing_ids = set()
        else:
            header = sheet_data[0]
            try:
                trade_id_index = header.index("trade_id")
            except ValueError:
                logging.error("trade_id column not found in existing sheet header.")
                return
            # Convert all recorded trade IDs to strings
            existing_ids = {str(row[trade_id_index]) for row in sheet_data[1:] if len(row) > trade_id_index}

        for record in trades_records:
            data = record.get("result", record)
            trade_id = data.get("trade_id")
            if trade_id is None:
                logging.warning("Record without trade_id encountered. Skipping.")
                continue
            if str(trade_id) in existing_ids:
                logging.info(f"Trade id {trade_id} already recorded. Skipping.")
                continue
 
            row = [data.get(col, "") for col in header]
            self.append_row(spreadsheet_id, sheet_name, row)


    def update_positions_history_sheet(self, spreadsheet_id: str, sheet_name: str) -> None:
        """
        qol update positions history sheet -- identifier is position_id
        """
        query = "SELECT * FROM positions ORDER BY entry_time ASC"
        positions_records = execute_sql_query(query)
        if not positions_records:
            logging.info("No positions records found.")
            return

        try:
            sheet_data = self.get_from_sheet(spreadsheet_id, sheet_name)
        except Exception as e:
            logging.error(f"Error retrieving sheet data: {e}")
            sheet_data = []

        if not sheet_data:
            first_record = positions_records[0]
            data = first_record.get("result", first_record)
            header = list(data.keys())
            logging.info("Appending header row for positions history:")
            self.append_row(spreadsheet_id, sheet_name, header)
            existing_ids = set()
        else:
            header = sheet_data[0]
            try:
                position_id_index = header.index("position_id")
            except ValueError:
                logging.error("position_id column not found in existing sheet header.")
                return
            existing_ids = {str(row[position_id_index]) for row in sheet_data[1:] if len(row) > position_id_index}

        for record in positions_records:
            data = record.get("result", record)
            position_id = data.get("position_id")
            if position_id is None:
                logging.warning("Record without position_id encountered. Skipping.")
                continue
            if str(position_id) in existing_ids:
                logging.info(f"Position id {position_id} already recorded. Skipping.")
                continue
            row = [data.get(col, "") for col in header]
            self.append_row(spreadsheet_id, sheet_name, row)


gsheet = GoogleSheetsClient()

def main():

    logging.info("Recording latest trades and positions to Sheets..")

    try:
        gsheet.update_trades_history_sheet(SPREADSHEET_ID, TRADES_HISTORY_SHEET_NAME)
    except Exception as e:
        logging.error(f"Error updating trades history sheet: {e}")

    try:
        gsheet.update_positions_history_sheet(SPREADSHEET_ID, POSITION_HISTORY_SHEET_NAME)
    except Exception as e:
        logging.error(f"Error updating positions history sheet: {e}")



if __name__ == "__main__":
    main()
    