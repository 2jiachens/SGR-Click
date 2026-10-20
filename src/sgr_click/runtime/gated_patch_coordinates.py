"""Align existing root-cell selection with nearest-neighbor mask upsampling.

Patch only the two audited expressions in an instance-local build function.
Frozen runtime files, physical user clicks and model parameters stay unchanged.
"""
import ast
import hashlib
import inspect
import textwrap
from types import MethodType
import numpy as np

PATCH_VARIANTS={
    'base_pruned_aligned_tau60_pixel_arc_quadratic':(.60,True),
    'base_pruned_aligned_tau75_pixel_arc_quadratic':(.75,True),
    'base_pruned_aligned_global_tau60_pixel_arc_quadratic':(.60,False),
    'base_pruned_aligned_global_tau75_pixel_arc_quadratic':(.75,False),
}


def configure(holder,variant):
    tau,gate=PATCH_VARIANTS.get(variant,(.60,True))
    holder.update(enabled=variant in PATCH_VARIANTS,tau=tau,original_single_scope_gate=gate)


def coordinate(click_xy,image_shape,grid_shape):
    x,y=click_xy;h,w=image_shape;gh,gw=grid_shape
    return (int(np.clip(np.floor(x*gw/w),0,gw-1)),
            int(np.clip(np.floor(y*gh/h),0,gh-1)))


def compile_aligned_build(method):
    original=method.__func__;text=textwrap.dedent(inspect.getsource(original))
    tree=ast.parse(text);assert len(tree.body)==1 and isinstance(tree.body[0],ast.FunctionDef)
    function=tree.body[0];assert function.name=='build';changed=[]
    for node in ast.walk(function):
        if not isinstance(node,ast.Assign) or len(node.targets)!=1 or not isinstance(node.targets[0],ast.Name):continue
        name=node.targets[0].id
        if name not in ['click_x','click_y']:continue
        expected=('round(click_xy[0] * grid_w / image.shape[1])' if name=='click_x'
                  else 'round(click_xy[1] * grid_h / image.shape[0])')
        calls=[n for n in ast.walk(node.value) if isinstance(n,ast.Call) and ast.unparse(n)==expected]
        assert len(calls)==1,('Frozen build expression changed',name)
        calls[0].func=ast.Attribute(value=ast.Name(id='np',ctx=ast.Load()),attr='floor',ctx=ast.Load())
        changed.append(name)
    assert sorted(changed)==['click_x','click_y']
    ast.fix_missing_locations(tree);namespace=dict(original.__globals__)
    exec(compile(tree,inspect.getfile(compile_aligned_build)+'::aligned_build','exec'),namespace)
    return namespace['build'],dict(replaced_expressions=changed,
        original_build_source_sha256=hashlib.sha256(text.encode()).hexdigest(),
        aligned_build_ast_sha256=hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest())


def install(controller,source,holder):
    base=type(controller);trajectory=source._external_trajectory
    original_build=trajectory.build;original_coordinate=source._patch_coordinate
    original_radius=float(source.prototype_radius)
    function,proof=compile_aligned_build(original_build)
    aligned_build=MethodType(function,trajectory)

    class AlignedPatchController(base):
        def _rebuild_hierarchy(self,simple_pred):
            direct=holder['enabled'] and not holder.get('original_single_scope_gate',True)
            if direct:
                trajectory.build=aligned_build;source._patch_coordinate=coordinate
                if holder.get('prototype_radius') is not None:
                    source.prototype_radius=float(holder['prototype_radius'])
                assert holder['tau'] in [.60,.75];trajectory.set_cluster_tau(holder['tau'])
            feature=source._prepared_semantic_feature;calls=getattr(source,'student_calls',None)
            result=super()._rebuild_hierarchy(simple_pred)
            if direct:
                assert source._prepared_semantic_feature is feature and getattr(source,'student_calls',None)==calls
            return result

        def add_click(self,x,y,is_positive):
            first=len(self.clicker)==0 and is_positive and self._init_mask is None
            if first:
                trajectory.build=original_build;source._patch_coordinate=original_coordinate
                source.prototype_radius=original_radius
                holder['details']=dict(enabled=False,single_scope_preserved=False)
            result=super().add_click(x,y,is_positive)
            if not first or not self.granularity_ready or not holder['enabled']:return result
            path=self.granularity_refiner.path
            if not holder.get('original_single_scope_gate',True):
                holder['details']=dict(enabled=True,single_scope_preserved=False,original_single_scope_gate=False,
                    cluster_tau=holder['tau'],physical_user_click_unchanged=True,descriptor_tensor_reused=True,
                    parameters_added=0,additional_hierarchy_rebuilds=0,resulting_stages=len(path.masks)-1,
                    projection_count_may_change_with_semantic_stages=True,**proof)
                return result
            if len(path.masks)<=2:
                holder['details']=dict(enabled=False,single_scope_preserved=True)
                return result
            feature=source._prepared_semantic_feature;calls=getattr(source,'student_calls',None)
            radius=holder.get('prototype_radius')
            if radius is not None:
                assert float(radius)==1.0 and original_radius==2.5
            if radius is None and holder.get('skip_aligned_rebuild',False) and holder['tau']==.60:
                assert trajectory._prepared_semantic_feature is feature
                grid=tuple(map(int,feature.shape[-2:]));shape=self.image.shape[:2]
                old_root=original_coordinate((x,y),shape,grid);new_root=coordinate((x,y),shape,grid)
                if old_root==new_root:
                    assert float(trajectory.cluster_tau)==.60
                    holder['details']=dict(enabled=False,single_scope_preserved=False,
                        coordinates_already_aligned=True,original_single_scope_gate=True,cluster_tau=.60,
                        root_cell=list(new_root),grid_shape=list(grid),physical_user_click_unchanged=True,
                        additional_hierarchy_rebuilds=0,unnecessary_hierarchy_rebuilds_avoided=1,
                        descriptor_tensor_reused=True,parameters_added=0,resulting_stages=len(path.masks)-1)
                    return result
            trajectory.build=aligned_build;source._patch_coordinate=coordinate
            if radius is not None:source.prototype_radius=float(radius)
            assert holder['tau'] in [.60,.75];trajectory.set_cluster_tau(holder['tau'])
            self.semantic_cache.clear();self.edge_cache.clear()
            self._rebuild_hierarchy(self.simple_probs_history[-1]);self._finalize_display_path()
            assert source._prepared_semantic_feature is feature and getattr(source,'student_calls',None)==calls
            holder['details']=dict(enabled=True,single_scope_preserved=False,cluster_tau=holder['tau'],
                original_single_scope_gate=True,additional_hierarchy_rebuilds=1,
                physical_user_click_unchanged=True,descriptor_tensor_reused=True,parameters_added=0,
                original_stages=len(path.masks)-1,resulting_stages=len(self.granularity_refiner.path.masks)-1,
                hierarchy_rebuild_adds_existing_projection_cost=True,**proof)
            holder['details'].update(original_prototype_radius=original_radius,
                prototype_radius=float(source.prototype_radius),
                prototype_radius_changed=float(source.prototype_radius)!=original_radius)
            self.set_granularity(self.requested_granularity)
            return result

    controller.__class__=AlignedPatchController
    return controller
