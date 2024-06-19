from mesh_util import SMPLX
from tqdm.auto import tqdm

def smplx_fitting(data):
    
    # SMPLX object
    SMPLX_object = SMPLX()
    # The optimizer and variables
    optimed_pose = data["body_pose"].requires_grad_(True)
    optimed_trans = data["trans"].requires_grad_(True)
    optimed_betas = data["betas"].requires_grad_(True)
    optimed_orient = data["global_orient"].requires_grad_(True)

    optimizer_smpl = torch.optim.Adam([optimed_pose, optimed_trans, optimed_betas, optimed_orient],
                                        lr=1e-2,
                                        amsgrad=True)
    scheduler_smpl = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_smpl,
        mode="min",
        factor=0.5,
        verbose=0,
        min_lr=1e-5,
        patience=args.patience,
    )
    # smpl optimization
    loop_smpl = tqdm(range(50))
    for i in loop_smpl:

        per_loop_lst = []

        optimizer_smpl.zero_grad()

        N_body, N_pose = optimed_pose.shape[:2]

        # 6d_rot to rot_mat
        optimed_orient_mat = rot6d_to_rotmat(optimed_orient.view(-1,
                                                                    6)).view(N_body, 1, 3, 3)
        optimed_pose_mat = rot6d_to_rotmat(optimed_pose.view(-1,
                                                                6)).view(N_body, N_pose, 3, 3)

        smpl_verts, smpl_landmarks, smpl_joints = dataset.smpl_model(
            shape_params=optimed_betas,
            expression_params=tensor2variable(data["exp"], device),
            body_pose=optimed_pose_mat,
            global_pose=optimed_orient_mat,
            jaw_pose=tensor2variable(data["jaw_pose"], device),
            left_hand_pose=tensor2variable(data["left_hand_pose"], device),
            right_hand_pose=tensor2variable(data["right_hand_pose"], device),
        )

        smpl_verts = (smpl_verts + optimed_trans) * data["scale"]
        smpl_joints = (smpl_joints + optimed_trans) * data["scale"] * torch.tensor([
            1.0, 1.0, -1.0
        ]).to(device)

        # landmark errors
        smpl_joints_3d = (
            smpl_joints[:, dataset.smpl_data.smpl_joint_ids_45_pixie, :] + 1.0
        ) * 0.5
        in_tensor["smpl_joint"] = smpl_joints[:,
                                                dataset.smpl_data.smpl_joint_ids_24_pixie, :]

        ghum_lmks = data["landmark"][:, SMPLX_object.ghum_smpl_pairs[:, 0], :2].to(device)
        ghum_conf = data["landmark"][:, SMPLX_object.ghum_smpl_pairs[:, 0], -1].to(device)
        smpl_lmks = smpl_joints_3d[:, SMPLX_object.ghum_smpl_pairs[:, 1], :2]

        # render optimized mesh as normal [-1,1]
        in_tensor["T_normal_F"], in_tensor["T_normal_B"] = dataset.render_normal(
            smpl_verts * torch.tensor([1.0, -1.0, -1.0]).to(device),
            in_tensor["smpl_faces"],
        )

        T_mask_F, T_mask_B = dataset.render.get_image(type="mask")

        with torch.no_grad():
            in_tensor["normal_F"], in_tensor["normal_B"] = normal_net.netG(in_tensor)

        diff_F_smpl = torch.abs(in_tensor["T_normal_F"] - in_tensor["normal_F"])
        diff_B_smpl = torch.abs(in_tensor["T_normal_B"] - in_tensor["normal_B"])

        # silhouette loss
        smpl_arr = torch.cat([T_mask_F, T_mask_B], dim=-1)
        gt_arr = in_tensor["mask"].repeat(1, 1, 2)
        diff_S = torch.abs(smpl_arr - gt_arr)
        losses["silhouette"]["value"] = diff_S.mean()

        # large cloth_overlap --> big difference between body and cloth mask
        # for loose clothing, reply more on landmarks instead of silhouette+normal loss
        cloth_overlap = diff_S.sum(dim=[1, 2]) / gt_arr.sum(dim=[1, 2])
        cloth_overlap_flag = cloth_overlap > cfg.cloth_overlap_thres
        losses["joint"]["weight"] = [50.0 if flag else 5.0 for flag in cloth_overlap_flag]

        # small body_overlap --> large occlusion or out-of-frame
        # for highly occluded body, reply only on high-confidence landmarks, no silhouette+normal loss

        # BUG: PyTorch3D silhouette renderer generates dilated mask
        bg_value = in_tensor["T_normal_F"][0, 0, 0, 0]
        smpl_arr_fake = torch.cat([
            in_tensor["T_normal_F"][:, 0].ne(bg_value).float(),
            in_tensor["T_normal_B"][:, 0].ne(bg_value).float()
        ],
                                    dim=-1)

        body_overlap = (gt_arr * smpl_arr_fake.gt(0.0)
                        ).sum(dim=[1, 2]) / smpl_arr_fake.gt(0.0).sum(dim=[1, 2])
        body_overlap_mask = (gt_arr * smpl_arr_fake).unsqueeze(1)
        body_overlap_flag = body_overlap < cfg.body_overlap_thres

        losses["normal"]["value"] = (
            diff_F_smpl * body_overlap_mask[..., :512] +
            diff_B_smpl * body_overlap_mask[..., 512:]
        ).mean() / 2.0

        losses["silhouette"]["weight"] = [0 if flag else 1.0 for flag in body_overlap_flag]
        occluded_idx = torch.where(body_overlap_flag)[0]
        ghum_conf[occluded_idx] *= ghum_conf[occluded_idx] > 0.95
        losses["joint"]["value"] = (torch.norm(ghum_lmks - smpl_lmks, dim=2) *
                                    ghum_conf).mean(dim=1)

        # Weighted sum of the losses
        smpl_loss = 0.0
        pbar_desc = "Body Fitting -- "
        for k in ["normal", "silhouette", "joint"]:
            per_loop_loss = (
                losses[k]["value"] * torch.tensor(losses[k]["weight"]).to(device)
            ).mean()
            pbar_desc += f"{k}: {per_loop_loss:.3f} | "
            smpl_loss += per_loop_loss
        pbar_desc += f"Total: {smpl_loss:.3f}"
        loose_str = ''.join([str(j) for j in cloth_overlap_flag.int().tolist()])
        occlude_str = ''.join([str(j) for j in body_overlap_flag.int().tolist()])
        pbar_desc += colored(f"| loose:{loose_str}, occluded:{occlude_str}", "yellow")
        loop_smpl.set_description(pbar_desc)

        # save intermediate results
        if (i == args.loop_smpl - 1) and (not args.novis):

            per_loop_lst.extend([
                in_tensor["image"],
                in_tensor["T_normal_F"],
                in_tensor["normal_F"],
                diff_S[:, :, :512].unsqueeze(1).repeat(1, 3, 1, 1),
            ])
            per_loop_lst.extend([
                in_tensor["image"],
                in_tensor["T_normal_B"],
                in_tensor["normal_B"],
                diff_S[:, :, 512:].unsqueeze(1).repeat(1, 3, 1, 1),
            ])
            per_data_lst.append(
                get_optim_grid_image(per_loop_lst, None, nrow=N_body * 2, type="smpl")
            )

        smpl_loss.backward()
        optimizer_smpl.step()
        scheduler_smpl.step(smpl_loss)

    in_tensor["smpl_verts"] = smpl_verts * torch.tensor([1.0, 1.0, -1.0]).to(device)
    in_tensor["smpl_faces"] = in_tensor["smpl_faces"][:, :, [0, 2, 1]]

    if not args.novis:
        per_data_lst[-1].save(
            osp.join(args.out_dir, cfg.name, f"png/{data['name']}_smpl.png")
        )

