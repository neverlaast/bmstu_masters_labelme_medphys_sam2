"""Class mutations keep stable IDs and never leave dangling references."""
from __future__ import annotations

from project_format.models import ClassDefinition, Project, now


def find_class(project, class_id):
    for definition in project.classes:
        if definition.id == class_id:
            return definition
    raise ValueError("Unknown class ID")


def _finish(result, action, **details):
    result.journal.append({"time": now(), "action": action, **details})
    return Project.model_validate(result.model_dump(mode="json"))


def add_class(project, definition):
    result = project.model_copy(deep=True)
    result.classes.append(definition)
    return _finish(result, "class_added", class_id=definition.id)


def rename_class(project, class_id, name, selected_ids=None):
    result = project.model_copy(deep=True)
    definition = find_class(result, class_id)
    name = name.strip()
    if selected_ids is None:
        definition.name = name
        return _finish(result, "class_renamed", class_id=class_id, name=name)
    selected_ids = set(selected_ids)
    linked = [a for image in result.images for a in image.annotations
              if a.class_id == class_id and a.id in selected_ids]
    if not linked:
        raise ValueError("No selected annotations belong to this class")
    new_definition = ClassDefinition(
        **{k: v for k, v in definition.model_dump().items() if k not in {"id", "name"}},
        name=name)
    result.classes.append(new_definition)
    for annotation in linked:
        annotation.class_id = new_definition.id
        annotation.modified_at = now()
    return _finish(result, "class_renamed_for_selection", class_id=class_id,
                   new_class_id=new_definition.id, object_ids=sorted(selected_ids))


def remove_class(project, class_id, policy="forbid", target_id=None):
    result = project.model_copy(deep=True)
    definition = find_class(result, class_id)
    linked = [a for image in result.images for a in image.annotations if a.class_id == class_id]
    if policy not in {"forbid", "deactivate", "transfer", "delete_objects"}:
        raise ValueError("Unknown removal policy")
    if policy == "deactivate":
        definition.active = False
    else:
        if linked and policy == "forbid":
            raise ValueError("Class is in use: transfer/delete objects or deactivate it")
        if policy == "transfer":
            target = find_class(result, target_id)
            if target.id == class_id or not target.active:
                raise ValueError("Choose a different active target class")
            for annotation in linked:
                annotation.class_id = target.id
                annotation.modified_at = now()
        if policy == "delete_objects":
            for image in result.images:
                image.annotations = [a for a in image.annotations if a.class_id != class_id]
        result.classes = [c for c in result.classes if c.id != class_id]
    return _finish(result, "class_removed", class_id=class_id, policy=policy,
                   target_id=target_id, affected_objects=len(linked))
