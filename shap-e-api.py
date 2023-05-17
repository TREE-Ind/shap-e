import torch
from flask import Flask, request, jsonify, make_response, send_file
import trimesh

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget

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
        karras_steps=64,
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
    vertex_colors = mesh.visual.vertex_colors

    material = trimesh.visual.material.from_color(vertex_colors)

    # Assign the PBR materials to the mesh
    mesh.visual.material = material

    mesh = mesh.smoothed()

    mesh.export('output.glb')


    # Return the generated point cloud as a response
    return send_file('output.glb')

if __name__ == '__main__':
    app.run(host='0.0.0.0')
