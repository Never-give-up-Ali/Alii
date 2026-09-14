from fastapi import FastAPI
from pydantic import BaseModel
import base64, io
import torch
from diffusers import StableDiffusionPipeline

app = FastAPI(title="SD V1.5 Private API")


MODEL_DIR = r""

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_DIR,
    safety_checker=None,
    requires_safety_checker=False,
    torch_dtype=torch.float16
).to("cuda")

class Txt2ImgRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    steps: int = 20
    guidance_scale: float = 7.5
    width: int = 512
    height: int = 512

@app.post("/sdapi/v1/txt2img")
async def txt2img(req: Txt2ImgRequest):
    with torch.no_grad():
        result = pipe(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            num_inference_steps=req.steps,
            guidance_scale=req.guidance_scale,
            width=req.width,
            height=req.height
        )
    img = result.images[0]
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return {"images": [b64]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("sd_server:app", host="0.0.0.0", port=7860)