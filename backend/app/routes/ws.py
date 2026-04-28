from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.utils.audio import decode_webm_chunk 
import json 

router = APIRouter() 

@router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        model_name = ""
        la = "en"
        while True:
            msg = await ws.receive()
            if "text" in msg: # JSON 
                data = json.loads(msg["text"])
                msg_type = data.get("type")

                if msg_type == "start":
                    model_name = data.get("model_name")
                    print("Start config:", model_name)
                    la = data.get("la")

                elif msg_type == "stop":
                    print("Stop stream")
                    break

            # 🔹 2. Nếu là audio binary
            elif "bytes" in msg:
                
                audio_chunk = msg["bytes"]

                # xử lý audio ở đây
                # print(model_name, la, audio_chunk)
                audio_np = decode_webm_chunk(audio_chunk)
                print(audio_np)
                # a , _, _  = inferenceText(model_name, la, audio_chunk)
                # print(a)
                text = "hello"

                await ws.send_json({ # await ws.send_text(text)
                    "type": "transcript",
                    "text": text,
                    "is_final": False
                })

    except WebSocketDisconnect:
        print("Client disconnected")

    finally:
        print("Cleanup here")