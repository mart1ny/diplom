export const INTERSECTIONS = [
  {
    value: "cross-a",
    label: "Площадь Мира",
    meta: "5 потоков",
    code: "PLAZA-04",
    saturation: "68%",
    flows: "5 потоков",
    notes: "PLAZA-04 — эталонный AI-сценарий центральной площади.",
  },
  {
    value: "cross-b",
    label: "Тестовая зона",
    meta: "3 потока",
    code: "TEST-17",
    saturation: "42%",
    flows: "3 потока",
    notes: "TEST-17 — экспериментальная площадка для ревизии фаз.",
  },
  {
    value: "cross-c",
    label: "МКАД · выезд",
    meta: "8 потоков",
    code: "MKAD-Exit",
    saturation: "83%",
    flows: "8 потоков",
    notes: "MKAD-Exit — пилот адаптивных фаз на развязке.",
  },
];

export const INTERSECTION_LOOKUP = INTERSECTIONS.reduce((acc, item) => {
  acc[item.value] = item;
  return acc;
}, {});

export const SIGNALS = [
  { value: "north-spine", label: "Северный рукав" },
  { value: "south-gate", label: "Южный въезд" },
  { value: "ped-core", label: "Пешеходный сегмент" },
];

export const SIGNAL_LOOKUP = SIGNALS.reduce((acc, item) => {
  acc[item.value] = item;
  return acc;
}, {});
