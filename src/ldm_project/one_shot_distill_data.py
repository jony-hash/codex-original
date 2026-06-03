from __future__ import annotations

from pathlib import Path

from .train_utils import validate_class_label_values


def _load_distill_payload(torch_module, distill_dataset: str | Path) -> dict[str, object]:
    path = Path(distill_dataset)
    if not path.is_file():
        raise FileNotFoundError(f"Distillation dataset does not exist: {path}")
    payload = torch_module.load(path, map_location="cpu", weights_only=False)
    required = ("codes", "class_labels", "teacher_camera")
    for key in required:
        if key not in payload:
            raise ValueError(f"Distillation dataset is missing required tensor: {key}")
    return payload

def _validated_distill_tensors(torch_module, payload: dict[str, object], train_cfg):
    codes = payload["codes"].float()
    labels = payload["class_labels"].long()
    targets = payload["teacher_camera"].float()
    if codes.ndim != 2:
        raise ValueError(f"distill codes must be [N, code_dim], got {tuple(codes.shape)}")
    if codes.shape[1] != train_cfg.one_shot.code_dim:
        raise ValueError(
            "distill code_dim does not match one_shot.code_dim: "
            f"{codes.shape[1]} != {train_cfg.one_shot.code_dim}"
        )
    if labels.ndim != 1 or labels.shape[0] != codes.shape[0] or targets.shape[0] != codes.shape[0]:
        raise ValueError("distill codes, class_labels, and teacher_camera must share the same batch dimension.")
    expected_target_shape = (1, train_cfg.first_stage.image_size, train_cfg.first_stage.image_size)
    if tuple(targets.shape[1:]) != expected_target_shape:
        raise ValueError(
            "distill teacher_camera shape does not match configured one-shot image size: "
            f"{tuple(targets.shape[1:])} != {expected_target_shape}"
        )
    validate_class_label_values(tuple(int(label) for label in labels.tolist()), train_cfg.one_shot.num_classes, "class_labels")
    return codes, labels, targets

def _filter_distill_payload_by_indices(torch_module, payload: dict[str, object], indices, *, source_size: int):
    index_tensor = torch_module.tensor(indices, dtype=torch_module.long)
    filtered: dict[str, object] = {}
    for key, value in payload.items():
        if isinstance(value, torch_module.Tensor) and value.ndim >= 1 and int(value.shape[0]) == source_size:
            filtered[key] = value.index_select(0, index_tensor).clone()
        else:
            filtered[key] = value
    return filtered, index_tensor

def _subset_distill_payload_for_training(torch_module, payload: dict[str, object], train_cfg):
    codes, _, _ = _validated_distill_tensors(torch_module, payload, train_cfg)
    source_size = int(codes.shape[0])
    subset_size = train_cfg.distillation.train_subset_size
    if subset_size is None:
        return payload, source_size, source_size
    if subset_size <= 0:
        raise ValueError(f"distillation.train_subset_size must be positive, got {subset_size}")
    if subset_size > source_size:
        raise ValueError(
            "distillation.train_subset_size must be <= distillation dataset size "
            f"({source_size}), got {subset_size}"
        )
    generator = torch_module.Generator().manual_seed(train_cfg.data.subset_seed)
    indices = torch_module.randperm(source_size, generator=generator)[:subset_size].tolist()
    filtered, _ = _filter_distill_payload_by_indices(
        torch_module,
        payload,
        indices,
        source_size=source_size,
    )
    return filtered, source_size, subset_size

def _dataset_labels(torch_module, dataset) -> list[int]:
    if isinstance(dataset, torch_module.utils.data.Subset):
        source = dataset.dataset
        if isinstance(source, torch_module.utils.data.TensorDataset) and len(source.tensors) >= 2:
            labels = source.tensors[1].detach().long().cpu()
            return [int(labels[int(index)].item()) for index in dataset.indices]
    labels: list[int] = []
    for index in range(len(dataset)):
        item = dataset[index]
        labels.append(int(item[1]))
    return labels

def _paired_class_batch_indices(torch_module, dataset, batch_size: int, seed: int) -> list[list[int]]:
    labels = _dataset_labels(torch_module, dataset)
    if len(labels) < 2:
        raise ValueError("distillation.paired_class_batches requires at least two training samples.")
    effective_batch_size = max(2, int(batch_size))
    by_class: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        by_class.setdefault(label, []).append(index)
    pairable = {label: indices for label, indices in by_class.items() if len(indices) >= 2}
    if not pairable:
        raise ValueError("distillation.paired_class_batches requires at least one class with two training samples.")

    generator = torch_module.Generator().manual_seed(seed)
    pairs: list[list[int]] = []
    leftovers: list[int] = []
    for indices in pairable.values():
        order = torch_module.randperm(len(indices), generator=generator).tolist()
        shuffled = [indices[int(item)] for item in order]
        for start in range(0, len(shuffled) - 1, 2):
            pairs.append([shuffled[start], shuffled[start + 1]])
        if len(shuffled) % 2 == 1:
            leftovers.append(shuffled[-1])
    for indices in by_class.values():
        if len(indices) < 2:
            leftovers.extend(indices)

    pair_order = torch_module.randperm(len(pairs), generator=generator).tolist()
    pairs = [pairs[int(item)] for item in pair_order]
    leftover_order = torch_module.randperm(len(leftovers), generator=generator).tolist() if leftovers else []
    leftovers = [leftovers[int(item)] for item in leftover_order]

    batches: list[list[int]] = []
    leftover_cursor = 0
    pair_cursor = 0
    pairs_per_batch = max(1, effective_batch_size // 2)
    while pair_cursor < len(pairs):
        batch: list[int] = []
        while pair_cursor < len(pairs) and len(batch) + 2 <= effective_batch_size and len(batch) // 2 < pairs_per_batch:
            batch.extend(pairs[pair_cursor])
            pair_cursor += 1
        while leftover_cursor < len(leftovers) and len(batch) < effective_batch_size:
            batch.append(leftovers[leftover_cursor])
            leftover_cursor += 1
        batches.append(batch)
    if leftover_cursor < len(leftovers) and batches:
        for index in leftovers[leftover_cursor:]:
            if len(batches[-1]) < effective_batch_size:
                batches[-1].append(index)
            else:
                batches.append([pairs[0][0], pairs[0][1], index])
    return batches

def _multiclass_batch_indices(torch_module, dataset, batch_size: int, seed: int) -> list[list[int]]:
    labels = _dataset_labels(torch_module, dataset)
    if len(labels) < 2:
        raise ValueError("distillation.multiclass_batches requires at least two training samples.")
    effective_batch_size = max(2, int(batch_size))
    by_class: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        by_class.setdefault(label, []).append(index)
    if len(by_class) < 2:
        raise ValueError("distillation.multiclass_batches requires at least two classes.")

    generator = torch_module.Generator().manual_seed(seed)
    shuffled_by_class: dict[int, list[int]] = {}
    for label, indices in by_class.items():
        order = torch_module.randperm(len(indices), generator=generator).tolist()
        shuffled_by_class[label] = [indices[int(item)] for item in order]
    class_order = [int(item) for item in torch_module.randperm(len(shuffled_by_class), generator=generator).tolist()]
    class_labels = list(shuffled_by_class.keys())
    ordered_labels = [class_labels[index] for index in class_order]

    cursors = {label: 0 for label in ordered_labels}
    remaining = sum(len(indices) for indices in shuffled_by_class.values())
    batches: list[list[int]] = []
    start_offset = 0
    while remaining > 0:
        batch: list[int] = []
        visited = 0
        cursor = start_offset
        while len(batch) < effective_batch_size and visited < len(ordered_labels) * 2 and remaining > 0:
            label = ordered_labels[cursor % len(ordered_labels)]
            cursor += 1
            visited += 1
            label_cursor = cursors[label]
            if label_cursor >= len(shuffled_by_class[label]):
                continue
            batch.append(shuffled_by_class[label][label_cursor])
            cursors[label] = label_cursor + 1
            remaining -= 1
        if not batch:
            break
        batches.append(batch)
        start_offset = (start_offset + 1) % len(ordered_labels)
    return batches

def _build_distill_loaders(torch_module, payload: dict[str, object], train_cfg, use_pin_memory: bool):
    codes, labels, targets = _validated_distill_tensors(torch_module, payload, train_cfg)
    dataset = torch_module.utils.data.TensorDataset(codes, labels, targets)
    total_size = len(dataset)
    validation_size = int(total_size * train_cfg.data.validation_fraction)
    if validation_size >= total_size:
        validation_size = total_size - 1
    train_size = total_size - validation_size
    if train_size <= 0:
        raise ValueError("Distillation dataset must leave at least one training sample.")
    generator = torch_module.Generator().manual_seed(train_cfg.data.validation_split_seed)
    if validation_size > 0:
        train_dataset, validation_dataset = torch_module.utils.data.random_split(
            dataset,
            (train_size, validation_size),
            generator=generator,
        )
    else:
        train_dataset = dataset
        validation_dataset = None
    if train_cfg.distillation.multiclass_batches:
        train_batches = _multiclass_batch_indices(
            torch_module,
            train_dataset,
            train_cfg.batch_size,
            train_cfg.data.subset_seed,
        )
        train_loader = torch_module.utils.data.DataLoader(
            train_dataset,
            batch_sampler=train_batches,
            num_workers=train_cfg.data.num_workers,
            pin_memory=use_pin_memory,
        )
    elif train_cfg.distillation.paired_class_batches:
        train_batches = _paired_class_batch_indices(
            torch_module,
            train_dataset,
            train_cfg.batch_size,
            train_cfg.data.subset_seed,
        )
        train_loader = torch_module.utils.data.DataLoader(
            train_dataset,
            batch_sampler=train_batches,
            num_workers=train_cfg.data.num_workers,
            pin_memory=use_pin_memory,
        )
    else:
        train_loader = torch_module.utils.data.DataLoader(
            train_dataset,
            batch_size=train_cfg.batch_size,
            shuffle=True,
            num_workers=train_cfg.data.num_workers,
            pin_memory=use_pin_memory,
            drop_last=False,
        )
    validation_loader = None
    if validation_dataset is not None:
        validation_loader = torch_module.utils.data.DataLoader(
            validation_dataset,
            batch_size=train_cfg.batch_size,
            shuffle=False,
            num_workers=train_cfg.data.num_workers,
            pin_memory=use_pin_memory,
            drop_last=False,
        )
    return train_loader, validation_loader, total_size, train_size, validation_size

def _select_distill_eval_indices(torch_module, labels, train_cfg) -> tuple[list[int], dict[str, object]]:
    label_values = labels.detach().cpu()
    requested = tuple(train_cfg.one_shot.sample_class_labels)
    if requested:
        class_labels = validate_class_label_values(requested, train_cfg.one_shot.num_classes, "one_shot.sample_class_labels")
    else:
        class_labels = tuple(sorted({int(label) for label in label_values.tolist()}))
    samples_per_class = int(train_cfg.one_shot.sample_repeats_per_class)
    if samples_per_class <= 0:
        raise ValueError(
            "one_shot.sample_repeats_per_class must be positive for one_shot_distill_eval, "
            f"got {samples_per_class}"
        )
    total_size = int(label_values.shape[0])
    validation_size = int(total_size * train_cfg.data.validation_fraction)
    if validation_size >= total_size:
        validation_size = total_size - 1
    train_size = total_size - validation_size
    generator = torch_module.Generator().manual_seed(train_cfg.data.validation_split_seed)
    permutation = torch_module.randperm(total_size, generator=generator).tolist()
    validation_indices = set(permutation[train_size:]) if validation_size > 0 else set()

    selected: list[int] = []
    per_class: dict[str, dict[str, int]] = {}
    used_train_fallback = False
    for label in class_labels:
        all_for_label = [index for index, item in enumerate(label_values.tolist()) if int(item) == int(label)]
        if not all_for_label:
            raise ValueError(f"Distillation dataset has no samples for requested eval class label: {label}")
        validation_for_label = [index for index in all_for_label if index in validation_indices]
        chosen = validation_for_label[:samples_per_class]
        if len(chosen) < min(samples_per_class, len(all_for_label)):
            fallback = [index for index in all_for_label if index not in set(chosen)]
            chosen.extend(fallback[: samples_per_class - len(chosen)])
            used_train_fallback = True
        selected.extend(chosen)
        per_class[str(label)] = {
            "available": len(all_for_label),
            "selected": len(chosen),
            "validation_selected": min(len(validation_for_label), len(chosen)),
        }

    return selected, {
        "class_labels": list(class_labels),
        "requested_samples_per_class": samples_per_class,
        "selected_count": len(selected),
        "validation_size": validation_size,
        "used_train_fallback": used_train_fallback,
        "per_class": per_class,
    }
