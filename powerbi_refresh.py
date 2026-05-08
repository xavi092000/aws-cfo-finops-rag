from __future__ import annotations

from typing import Any
import requests
import time

# =========================
# POWER BI CONFIG
# =========================
TENANT_ID = ""
CLIENT_ID = ""
CLIENT_SECRET = ""

WORKSPACE_ID = ""
DATASET_ID = ""

POWERBI_SCOPE = "https://analysis.windows.net/powerbi/api/.default"


# =========================
# TOKEN
# =========================
def get_powerbi_access_token() -> str:
    token_url = f"https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/token"

    data = {
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "scope": POWERBI_SCOPE,
    }

    response = requests.post(token_url, data=data, timeout=30)

    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to get Power BI token. "
            f"Status={response.status_code}. Response={response.text}"
        )

    payload: dict[str, Any] = response.json()
    access_token = payload.get("access_token")

    if not access_token:
        raise RuntimeError("Power BI token response did not contain access_token.")

    return access_token


# =========================
# TRIGGER REFRESH
# =========================
def trigger_powerbi_refresh(access_token: str) -> None:
    refresh_url = (
        f"https://api.powerbi.com/v1.0/myorg/groups/{WORKSPACE_ID}"
        f"/datasets/{DATASET_ID}/refreshes"
    )

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    response = requests.post(refresh_url, headers=headers, timeout=30)

    if response.status_code != 202:
        raise RuntimeError(
            f"Power BI refresh trigger failed. "
            f"Status={response.status_code}. Response={response.text}"
        )

    print("Power BI refresh triggered")


# =========================
# CHECK STATUS
# =========================
def wait_for_refresh(access_token: str, timeout: int = 120) -> str:
    url = (
        f"https://api.powerbi.com/v1.0/myorg/groups/{WORKSPACE_ID}"
        f"/datasets/{DATASET_ID}/refreshes?$top=1"
    )

    headers = {
        "Authorization": f"Bearer {access_token}",
    }

    start_time = time.time()

    while True:
        response = requests.get(url, headers=headers, timeout=30)

        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to get refresh status. "
                f"Status={response.status_code}. Response={response.text}"
            )

        data = response.json()
        status = data["value"][0]["status"]

        print(f"Refresh status: {status}")

        if status in ["Completed", "Failed"]:
            return status

        if time.time() - start_time > timeout:
            return "Timeout"

        time.sleep(5)


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    token = get_powerbi_access_token()

    trigger_powerbi_refresh(token)

    final_status = wait_for_refresh(token)

    print(f"Final refresh status: {final_status}")