from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import json
from datetime import datetime
from pathlib import Path

app = FastAPI()

# Folder untuk menyimpan data
output_folder = Path("received_json")
output_folder.mkdir(exist_ok=True)

@app.post("/upload")
async def upload_data(request: Request):
    try:
        data = await request.json()

        # Buat nama file berdasarkan timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_folder / f"data_{timestamp}.json"

        # Simpan data ke file JSON dengan indentasi
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)

        print(f"[INFO] Data diterima dan disimpan: {filename}")
        return JSONResponse(content={"status": True, "message": "Data received successfully"})
    
    except Exception as e:
        print(f"[ERROR] Gagal memproses data: {e}")
        return JSONResponse(content={"status": False, "message": str(e)})
