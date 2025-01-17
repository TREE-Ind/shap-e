import torch
from flask import Flask, request, jsonify, make_response, send_file
import trimesh
import numpy as np

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget

import boto3
from botocore.exceptions import NoCredentialsError

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

xm = load_model('transmitter', device=device)
model = load_model('text300M', device=device)
diffusion = diffusion_from_config(load_config('diffusion'))

@app.route('/generate', methods=['POST'])
def generate_point_cloud():

    batch_size = 4
    guidance_scale = 15.0
    # Get the prompt from the request
    prompt = request.json['prompt']

    latents = sample_latents(
        batch_size=batch_size,
        model=model,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=dict(texts=[prompt] * batch_size),
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=32,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )

    render_mode = 'nerf' # you can change this to 'stf'
    size = 64 # this is the size of the renders; higher values take longer to render.

    # Example of saving the latents as meshes.
    from shap_e.util.notebooks import decode_latent_mesh

    for i, latent in enumerate(latents):
        with open(f'mesh.ply', 'wb') as f:
            decode_latent_mesh(xm, latent).tri_mesh().write_ply(f)

    mesh = trimesh.load('mesh.ply')

    # Extract the vertex colors from the mesh
    #vertex_colors = mesh.visual.vertex_colors

    #material = trimesh.visual.material.from_color(vertex_colors)

    # Assign the PBR materials to the mesh
    #mesh.visual.material = material
    
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
        return jsonify({'url': url})
    except FileNotFoundError:
        print("The file was not found")
        return jsonify({'error': 'The file was not found'}), 400
    except NoCredentialsError:
        print("Credentials not available")
        return jsonify({'error': 'Credentials not available'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
