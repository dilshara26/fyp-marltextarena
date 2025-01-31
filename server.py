from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import asyncio
from implementation import ActorCriticLoop, ActorOnlyLoop, WebSocketActorCriticLoop, WebSocketActorOnlyLoop
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional
import uvicorn

# Load environment variables
load_dotenv()

app = FastAPI()

class GameConfig(BaseModel):
    actor_model1_name: str
    actor_model2_name: str
    critic_model_name: str
    game_seed: Optional[int] = None

class ActorOnlyGameConfig(BaseModel):
    actor_model1_name: str
    actor_model2_name: str
    game_seed: Optional[int] = None

@app.get("/api/game")
async def get_game():
    return JSONResponse(
        content={
            "status": "success",
            "message": "The game server is running..."
        }
    )

@app.post("/api/game")
async def run_game(config: GameConfig):
    try:
        print("Post request received...")
        print(f"Actor Model 1: {config.actor_model1_name}")
        print(f"Actor Model 2: {config.actor_model2_name}")
        print(f"Critic Model: {config.critic_model_name}")
        print(f"Seed: {config.game_seed}")
        game_loop = ActorCriticLoop(
            env_id="Negotiation-v0",
            actor_model1_name=config.actor_model1_name,
            actor_model2_name=config.actor_model2_name,
            critic_model_name=config.critic_model_name,
            isHuman=False
        )

        step_outputs = game_loop.run_episode(seed=config.game_seed)
        
        return JSONResponse(
            content={
                "status": "success",
                "game_history": step_outputs
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )

@app.post("/api/game/only_actor")
async def run_game_only_actor(config: ActorOnlyGameConfig):
    try:
        print("Post request received...")
        print(f"Actor Model 1: {config.actor_model1_name}")
        print(f"Actor Model 2: {config.actor_model2_name}")
        print(f"Seed: {config.game_seed}")
        game_loop = ActorOnlyLoop(
            env_id="Negotiation-v0",
            actor_model1_name=config.actor_model1_name,
            actor_model2_name=config.actor_model2_name,
            isHuman=False
        )

        step_outputs = game_loop.run_episode(seed=config.game_seed)
        
        return JSONResponse(
            content={
                "status": "success",
                "game_history": step_outputs
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_message(self, message: Dict[str, Any], websocket: WebSocket):
        await websocket.send_json(message)

manager = ConnectionManager()

@app.websocket("/ws/game")
async def websocket_game(websocket: WebSocket, actor_model1: str, actor_model2: str, critic_model: str, seed: int = 1):
    await manager.connect(websocket)
    try:
        print("Websocket connection established!")
        print(f"Actor Model 1: {actor_model1}")
        print(f"Actor Model 2: {actor_model2}")
        print(f"Critic Model: {critic_model}")
        print(f"Seed: {seed}")
        game_loop = WebSocketActorCriticLoop(
            env_id="Negotiation-v0",
            actor_model1_name=actor_model1,
            actor_model2_name=actor_model2,
            critic_model_name=critic_model,
        )

        for step_output in game_loop.run_episode(seed=seed):
            await manager.send_message(step_output, websocket)
            await asyncio.sleep(0.1)  # Prevent flooding

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        error_message = {"error": str(e)}
        await manager.send_message(error_message, websocket)
        manager.disconnect(websocket)
        
@app.websocket("/ws/game/only_actor")
async def websocket_game_only_actor(websocket: WebSocket, actor_model1: str, actor_model2: str, seed: int = 1):
    await manager.connect(websocket)
    try:
        print("Websocket connection established!")
        print(f"Actor Model 1: {actor_model1}")
        print(f"Actor Model 2: {actor_model1}")
        print(f"Seed: {seed}")
        game_loop = WebSocketActorOnlyLoop(
            env_id="Negotiation-v0",
            actor_model1_name=actor_model1,
            actor_model2_name=actor_model2,
        )

        for step_output in game_loop.run_episode(seed=seed):
            await manager.send_message(step_output, websocket)
            await asyncio.sleep(0.1)  # Prevent flooding

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        error_message = {"error": str(e)}
        await manager.send_message(error_message, websocket)
        manager.disconnect(websocket)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
