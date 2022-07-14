import os
import torch
import pytorch3d
from pytorch3d.io import load_obj, save_obj, load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)
import numpy as np
from tqdm.notebook import tqdm

# Set the device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("WARNING: CPU only, this will be slow!")

src_obj = ['/scratch1/CMH/SPIN/demo_tog/registered_data_new/top_006.obj',
'/scratch1/CMH/SPIN/demo_tog/registered_data_new/bot_tmp.obj']
trg_obj = '/scratch1/CMH/SPIN/demo_tog/ours_cloth2.obj'
src_mesh = pytorch3d.structures.join_meshes_as_scene(load_objs_as_meshes(src_obj).cuda())
src_mesh = pytorch3d.ops.SubdivideMeshes().forward(src_mesh)
print(src_mesh.num_faces_per_mesh().shape)
vs=[]
with open(trg_obj, 'r') as f:
    for lines in f:
        vs.append([float(k) for k in lines[2:-1].split(' ')])
vs = torch.tensor(vs).cuda()

deform_verts = torch.full(src_mesh.verts_packed().shape, 0.0, device=device, requires_grad=True)
off = torch.tensor([0.0,-0.0,0],device=device,dtype=torch.float32)
optimizer = torch.optim.SGD([deform_verts], lr=1.0, momentum=0.9)

# Number of optimization steps
Niter = 2000
# Weight for the chamfer loss
w_chamfer = 1.0 
# Weight for mesh edge loss
w_edge = 2.0 
# Weight for mesh normal consistency
w_normal = 0.1 
# Weight for mesh laplacian smoothing
w_laplacian = 0.2
# Plot period for the losses
plot_period = 250
loop = tqdm(range(Niter))

chamfer_losses = []
laplacian_losses = []
edge_losses = []
normal_losses = []


for i in loop:
    # Initialize optimizer
    optimizer.zero_grad()
    
    # Deform the mesh
    new_src_mesh = src_mesh.offset_verts(deform_verts+off)
    del_src_mesh = Meshes(verts=[deform_verts],faces=src_mesh.faces_list())
    
    # We sample 5k points from the surface of each mesh 
    sample_trg = vs[torch.randint(32768, [5000])].unsqueeze(0)
    sample_src = sample_points_from_meshes(new_src_mesh, 5000)
    
    # We compare the two sets of pointclouds by computing (a) the chamfer loss
    loss_chamfer, _ = chamfer_distance(sample_trg, sample_src)
    
    # and (b) the edge length of the predicted mesh
    loss_edge = mesh_edge_loss(del_src_mesh)
    
    # mesh normal consistency
    loss_normal = mesh_normal_consistency(new_src_mesh)
    
    # mesh laplacian smoothing
    loss_laplacian = mesh_laplacian_smoothing(del_src_mesh, method="uniform")
    
    # Weighted sum of the losses
    loss = loss_chamfer * w_chamfer + loss_edge * w_edge + loss_normal * w_normal + loss_laplacian * w_laplacian
    
    # Print the losses
    loop.set_description('total_loss = %.6f' % loss)
    
    # Save the losses for plotting
    chamfer_losses.append(loss_chamfer)
    edge_losses.append(loss_edge)
    normal_losses.append(loss_normal)
    laplacian_losses.append(loss_laplacian)
    
        
    # Optimization step
    loss.backward()
    optimizer.step()
    if i % 100 == 0:
        print("step",i,'loss',loss.data)

final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)

# Scale normalize back to the original target size
final_verts = final_verts

# Store the predicted mesh using save_obj
final_obj = os.path.join('./', 'final_model.obj')
save_obj(final_obj, final_verts, final_faces)