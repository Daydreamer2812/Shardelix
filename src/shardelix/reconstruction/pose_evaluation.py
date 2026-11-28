"""Independent evaluator, never imported by the blind solver."""
import numpy as np
from shardelix.reconstruction.common import Pose,render
from shardelix.reconstruction.evaluation import adjacency,canvas_align
from shardelix.reconstruction.rectification import normalize_fragments

def evaluate_pose(pieces,poses,truth,meta,result,truth_labels):
    shape=(meta["height"],meta["width"]);grid=tuple(meta["grid"]) if meta.get("grid") else None
    mode=meta.get("angle_mode","quarter");sampling=meta.get("sampling",1)
    norm,angles=normalize_fragments(pieces,shape,grid,mode,sampling,cubic_a=result.get("diagnostics",{}).get("normalization_cubic_a",-1.))
    gt={}
    for i,p in truth["poses"].items():
        k=int(round((p["input_ccw_deg"]+angles[i])/90))%4 if mode=="continuous" else p["k"]
        gt[i]=Pose(p["x"],p["y"],k)
    raw_conflicts=core_conflicts=0
    if result.get("diagnostics",{}).get("raw_boundary_conflicts") is not None:
        from shardelix.reconstruction.irregular.tolerant import repair_boundary_ownership
        norm,raw_conflicts,core_conflicts=repair_boundary_ownership(norm,poses,shape)
    aligned,k=canvas_align(norm,poses,gt,shape,meta["rotations"],return_rotation=True)
    _,raw,collision=render(norm,poses,shape)
    _,pred,_=render(norm,aligned,shape)
    lookup={p.id:j for j,p in enumerate(norm)}
    remap=np.array([lookup[i] for i in truth["label_ids"]],np.int32)
    lt=np.full(truth_labels.shape,-1,np.int32);sel=truth_labels>=0;lt[sel]=remap[truth_labels[sel]]
    tp=len(adjacency(pred)&adjacency(lt));ap=len(adjacency(pred));at=len(adjacency(lt))
    precision=tp/max(1,ap);recall=tp/max(1,at);f1=2*tp/max(1,ap+at) if ap+at else 1.
    errors=[];correct=0;correct_relaxed=0;exact=0
    for i,p in aligned.items():
        q=gt[i];err=abs(((truth["poses"][i]["input_ccw_deg"]+angles[i]-p.k*90+180)%360)-180) if mode=="continuous" else (0. if p.k==q.k else 90.)
        errors.append(err)
        dx=abs(p.x-q.x);dy=abs(p.y-q.y);exact+=p==q
        correct+=int(dx==dy==0 and err<=2)
        correct_relaxed+=int(max(dx,dy)<=1 and err<=2)
    raw_correct=sum(p.x==gt[i].x and p.y==gt[i].y and
                    (abs(((truth["poses"][i]["input_ccw_deg"]+angles[i]-p.k*90+180)%360)-180)<=2
                     if mode=="continuous" else p.k==gt[i].k) for i,p in poses.items())
    n=len(norm);observed_area=int(sel.sum());found=int((pred>=0).sum());available_recovery=len(poses)/n
    area_recovery=min(1.,found/max(1,observed_area))
    dpa=correct/n;label_recovery=float(((pred==lt)&sel).sum()/max(1,observed_area))
    missing_gt=~sel;missing_pred=pred<0
    missing_tp=int((missing_gt&missing_pred).sum());missing_den=int(missing_gt.sum()+missing_pred.sum())
    return dict(piece_count=n,expected_count=meta["expected_count"],missing_count=meta["expected_count"]-n,placed=len(poses),direct_position_accuracy=dpa,
                exact_discrete_pose_accuracy=exact/n,position_accuracy_1px=correct_relaxed/n,
                raw_position_accuracy=raw_correct/n,evaluation_rotation_cw_deg=k*90,
                adjacency_precision=precision,adjacency_recall=recall,adjacency_f1=f1,coverage=float((raw>=0).mean()),
                expected_observed_coverage=float(sel.mean()),available_piece_recovery=available_recovery,available_area_recovery=area_recovery,
                observed_pixel_label_accuracy=label_recovery,missing_pixel_f1=2*missing_tp/missing_den if missing_den else 1.,
                angle_mae_deg=float(np.mean(errors)) if errors else None,
                angle_mode=mode,sampling=sampling,collision_pixels=collision,raw_boundary_conflicts=raw_conflicts,core_conflicts=core_conflicts,
                pass_95=bool(dpa>.95 and f1>.95 and available_recovery>.95 and area_recovery>.95 and collision==0 and core_conflicts==0 and raw_conflicts/max(1,observed_area)<.01))
