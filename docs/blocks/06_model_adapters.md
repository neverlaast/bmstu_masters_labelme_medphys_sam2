# Блок 6. Общий контракт модели

Задача ТЗ: интеграция SAM2 «или аналогичных подходов».

## Что реализовано

- Контракт `SegmentationAdapter`: `load`, `unload`, `get_info`,
  `get_capabilities`, `clear_cache`, `prepare_image`, `compute_embedding`,
  `segment_prompt`, `generate_masks`, `process_sequence`.
- `AdapterCapabilities` описывает поддерживаемые режимы; неподдерживаемые
  отклоняются явно.
- SAM2 работает через этот контракт; другая модель подключается доверенной
  фабрикой (`--adapter module:factory`) без изменения GUI.
- Пример второго адаптера `threshold_demo` — пороговый алгоритм, не нейросеть.
- Выгрузка модели и просмотр её возможностей из GUI.

## Модули

`worker/adapters.py`, `worker/model_registry.py`, `worker/plugins/`.

## Ограничения

Готовых адаптеров других нейросетей нет; код фабрики исполняется
с правами worker.
