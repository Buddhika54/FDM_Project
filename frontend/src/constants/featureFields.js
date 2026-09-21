/**
 * Form field catalog — keep in sync with backend/models/prediction_schema.py
 * and docs/data-dictionary.md.
 *
 * Names match dataset/raw/train.csv predictors only.
 * Excluded: order_id (identifier), returned (target).
 *
 * Numeric min/max match Phase 2 / API live-traffic bounds (reject, do not clip).
 * delivery_delay_days is signed (negative = arrived early) and has no range cap.
 */
export const FEATURE_FIELDS = [
  {
    name: "customer_age",
    label: "Customer age",
    type: "number",
    integer: true,
    min: 1,
    max: 120,
    step: "1",
    hint: "1–120 years",
  },
  {
    name: "product_price",
    label: "Product price",
    type: "number",
    min: 0,
    step: "0.01",
    hint: "≥ 0",
  },
  {
    name: "discount_percent",
    label: "Discount percent",
    type: "number",
    min: 0,
    max: 100,
    step: "0.01",
    hint: "0–100 (percent, not a 0–1 rate)",
  },
  {
    name: "product_rating",
    label: "Product rating",
    type: "number",
    min: 1,
    max: 5,
    step: "0.1",
    hint: "1–5",
  },
  {
    name: "past_purchase_count",
    label: "Past purchase count",
    type: "number",
    integer: true,
    min: 0,
    step: "1",
    hint: "≥ 0",
  },
  {
    name: "past_return_rate",
    label: "Past return rate",
    type: "number",
    min: 0,
    max: 1,
    step: "0.01",
    hint: "0–1",
  },
  {
    name: "delivery_delay_days",
    label: "Delivery delay (days)",
    type: "number",
    step: "0.01",
    hint: "Signed: negative means arrived early",
  },
  {
    name: "session_length_minutes",
    label: "Session length (minutes)",
    type: "number",
    min: 0,
    step: "0.1",
    hint: "≥ 0",
  },
  {
    name: "num_product_views",
    label: "Product views",
    type: "number",
    integer: true,
    min: 0,
    step: "1",
    hint: "≥ 0",
  },
  {
    name: "device_type",
    label: "Device type",
    type: "select",
    options: [
      { value: "mobile", label: "mobile" },
      { value: "desktop", label: "desktop" },
      { value: "tablet", label: "tablet" },
    ],
  },
  {
    name: "product_category",
    label: "Product category",
    type: "select",
    options: [
      { value: "toys", label: "toys" },
      { value: "beauty", label: "beauty" },
      { value: "electronics", label: "electronics" },
      { value: "home", label: "home" },
      { value: "clothing", label: "clothing" },
      { value: "sports", label: "sports" },
    ],
  },
  {
    name: "shipping_method",
    label: "Shipping method",
    type: "select",
    options: [
      { value: "standard", label: "standard" },
      { value: "express", label: "express" },
      { value: "same_day", label: "same_day" },
    ],
  },
  {
    name: "payment_method",
    label: "Payment method",
    type: "select",
    options: [
      { value: "debit_card", label: "debit_card" },
      { value: "credit_card", label: "credit_card" },
      { value: "apple_pay", label: "apple_pay" },
      { value: "paypal", label: "paypal" },
    ],
  },
  {
    name: "used_coupon",
    label: "Used coupon",
    type: "checkbox",
    hint: "On = 1, off = 0",
  },
];

export const INITIAL_FORM_STATE = FEATURE_FIELDS.reduce((acc, field) => {
  acc[field.name] = field.type === "checkbox" ? false : "";
  return acc;
}, {});

export function valuesToPayload(values) {
  const payload = {};
  for (const field of FEATURE_FIELDS) {
    if (field.type === "checkbox") {
      payload[field.name] = values[field.name] ? 1 : 0;
      continue;
    }
    if (field.type === "number") {
      const n = Number(values[field.name]);
      payload[field.name] = field.integer ? parseInt(values[field.name], 10) : n;
      continue;
    }
    payload[field.name] = values[field.name];
  }
  return payload;
}

export function validateFormValues(values) {
  const errors = {};
  for (const field of FEATURE_FIELDS) {
    if (field.type === "checkbox") continue;

    const raw = values[field.name];
    if (raw === "" || raw === null || raw === undefined) {
      errors[field.name] = "This field is required";
      continue;
    }

    if (field.type === "select") {
      const allowed = (field.options || []).map((opt) => String(opt.value));
      if (!allowed.includes(String(raw))) {
        errors[field.name] = `Must be one of: ${allowed.join(", ")}`;
      }
      continue;
    }

    const n = Number(raw);
    if (Number.isNaN(n)) {
      errors[field.name] = "Must be a number";
      continue;
    }
    if (field.integer && !Number.isInteger(n)) {
      errors[field.name] = "Must be an integer";
      continue;
    }
    if (field.min != null && n < field.min) {
      errors[field.name] = `Must be ≥ ${field.min}`;
      continue;
    }
    if (field.max != null && n > field.max) {
      errors[field.name] = `Must be ≤ ${field.max}`;
    }
  }
  return errors;
}
