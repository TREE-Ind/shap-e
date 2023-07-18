import uvicorn
import torch
import trimesh
import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import decode_latent_mesh

import boto3
from botocore.exceptions import NoCredentialsError

app = FastAPI()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

xm = load_model('transmitter', device=device)
model = load_model('text300M', device=device)
diffusion = diffusion_from_config(load_config('diffusion'))


class Prompt(BaseModel):
    prompt: str


@app.post('/generate')
async def generate_point_cloud(prompt: Prompt):
    batch_size = 4
    guidance_scale = 15.0

    latents = sample_latents(
        batch_size=batch_size,
        model=model,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=dict(texts=[prompt.prompt] * batch_size),
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=32,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )

    # Example of saving the latents as meshes.
    for i, latent in enumerate(latents):
        with open(f'mesh.ply', 'wb') as f:
            decode_latent_mesh(xm, latent).tri_mesh().write_ply(f)

    mesh = trimesh.load('mesh.ply')

    transformation_matrix = trimesh.transformations.rotation_matrix(
        np.radians(-90),  # convert degrees to radians
        (1, 0, 0)  # rotate around x-axis
    )

    mesh.apply_transform(transformation_matrix)

    mesh = mesh.simplify_quadric_decimation(70000)

    mesh = mesh.smoothed()

    mesh.export("output.obj")

    s3 = boto3.client('s3')

    filename = 'output.obj'  # this can be any file on your local machine
    bucket_name = 'collodi'  # replace with your bucket name

    try:
        s3.upload_file(filename, bucket_name, filename)
        print("Upload Successful")
        url = f"https://{bucket_name}.s3.amazonaws.com/{filename}"
        return {'url': url}
    except FileNotFoundError:
        print("The file was not found")
        return {'error': 'The file was not found'}
    except NoCredentialsError:
        print("Credentials not available")
        return {'error': 'Credentials not available'}


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8080)
